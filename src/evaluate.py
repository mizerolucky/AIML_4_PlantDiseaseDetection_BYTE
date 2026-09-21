"""Evaluate the trained model on the held-out test split.

    python -m src.evaluate

This is the only place the test split is read. Writes results/metrics.json,
results/plots/confusion_matrix.png and results/plots/training_curves.png.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from .config import BATCH_SIZE, CHECKPOINT, CLASSES, PLOTS_DIR, RESULTS_DIR
from .data import LeafDataset
from .model import FloraLensNet


def load_model(device: torch.device) -> FloraLensNet:
    if not CHECKPOINT.exists():
        raise SystemExit(f"No checkpoint at {CHECKPOINT}. Run `python -m src.train` first.")
    # pretrained=False: the fine-tuned checkpoint overwrites every backbone
    # weight anyway, so re-reading the ImageNet file would be wasted work.
    model = FloraLensNet(pretrained=False).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val macro F1 {ckpt['val_macro_f1']:.4f})")
    return model


def plot_confusion(cm: np.ndarray, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, normalise, title in (
        (axes[0], False, "Counts"),
        (axes[1], True, "Row-normalised (recall per class)"),
    ):
        data = cm.astype(float)
        if normalise:
            data = data / data.sum(axis=1, keepdims=True)
        disp = ConfusionMatrixDisplay(data, display_labels=CLASSES)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f" if normalise else ".0f")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    # The normalised panel matters because the test set holds 150 leaves for
    # each disease but only 23 healthy ones; raw counts make the rare class
    # look negligible when it is the one worth watching.
    fig.suptitle("FloraLens - test set confusion matrix", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(history: list[dict], path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="s", label="val")
    axes[0].set_title("Loss")

    axes[1].plot(epochs, [h["train_acc"] for h in history], marker="o", label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], marker="s", label="val")
    axes[1].set_title("Accuracy")

    axes[2].plot(epochs, [h["train_macro_f1"] for h in history], marker="o", label="train")
    axes[2].plot(epochs, [h["val_macro_f1"] for h in history], marker="s", label="val")
    axes[2].set_title("Macro F1 (checkpoint selection metric)")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("FloraLens - training history", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    test_ds = LeafDataset("test")
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Test images: {len(test_ds)}")

    preds: list[int] = []
    truth: list[int] = []
    confidences: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            preds.extend(pred.cpu().tolist())
            truth.extend(labels.tolist())
            confidences.extend(conf.cpu().tolist())

    preds_a, truth_a = np.array(preds), np.array(truth)
    accuracy = float((preds_a == truth_a).mean())
    cm = confusion_matrix(truth_a, preds_a, labels=list(range(len(CLASSES))))
    report = classification_report(
        truth_a, preds_a, target_names=CLASSES, output_dict=True, zero_division=0
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, PLOTS_DIR / "confusion_matrix.png")

    history_file = RESULTS_DIR / "training_history.json"
    if history_file.exists():
        plot_curves(json.loads(history_file.read_text())["history"], PLOTS_DIR / "training_curves.png")

    correct_conf = [c for c, p, t in zip(confidences, preds, truth) if p == t]
    wrong_conf = [c for c, p, t in zip(confidences, preds, truth) if p != t]

    metrics = {
        "test_accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
            for name in CLASSES
        },
        "confusion_matrix": {"labels": CLASSES, "rows_are_actual": cm.tolist()},
        "confidence": {
            "mean_when_correct": float(np.mean(correct_conf)) if correct_conf else None,
            "mean_when_wrong": float(np.mean(wrong_conf)) if wrong_conf else None,
            "n_wrong": len(wrong_conf),
        },
        "n_test_images": len(test_ds),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nTest accuracy : {accuracy:.4f}")
    print(f"Macro F1      : {metrics['macro_f1']:.4f}")
    print("\nPer class:")
    print(f"  {'class':<14}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for name in CLASSES:
        p = metrics["per_class"][name]
        print(
            f"  {name:<14}{p['precision']:>10.4f}{p['recall']:>9.4f}"
            f"{p['f1']:>8.4f}{p['support']:>9d}"
        )
    print(f"\nMisclassified: {len(wrong_conf)} of {len(test_ds)}")
    print(f"Wrote {RESULTS_DIR/'metrics.json'} and plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
