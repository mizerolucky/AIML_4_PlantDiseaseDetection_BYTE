"""Generate the sample inference figures required by the task.

    python -m src.sample_predictions

Produces results/samples/*.png, one per test image, each showing the original
leaf beside its Grad-CAM overlay with the predicted label, the confidence and
the full probability breakdown. Also writes a contact sheet and a CSV.

Selection is deliberate rather than random: the set covers every class, and
includes both the least-confident correct predictions and every mistake the
model made. A gallery of twelve easy wins would say nothing about where the
model actually stands.
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .config import CLASSES, IMAGE_SIZE, RESULTS_DIR, SAMPLES_DIR
from .data import LeafDataset
from .evaluate import load_model
from .gradcam import cam_to_overlay, gradcam_autograd

PER_CLASS = 4


def main() -> None:
    device = torch.device("cpu")
    model = load_model(device)
    ds = LeafDataset("test", train_mode=False)

    # One pass to score every test image, so selection can be informed.
    scored = []
    for idx in range(len(ds)):
        x, label = ds[idx]
        with torch.no_grad():
            probs = torch.softmax(model(x.unsqueeze(0)), dim=1)[0].numpy()
        pred = int(probs.argmax())
        scored.append(
            {
                "idx": idx,
                "true": int(label),
                "pred": pred,
                "conf": float(probs[pred]),
                "probs": probs,
                "correct": pred == int(label),
            }
        )

    chosen: list[dict] = []
    seen: set[int] = set()

    # Every mistake, since those are the informative cases.
    for rec in sorted((r for r in scored if not r["correct"]), key=lambda r: -r["conf"]):
        chosen.append(rec)
        seen.add(rec["idx"])

    # Then, per class, the least-confident correct predictions: the model's
    # own margin of doubt, which is where the heatmaps are worth reading.
    for cls in range(len(CLASSES)):
        pool = sorted(
            (r for r in scored if r["true"] == cls and r["correct"] and r["idx"] not in seen),
            key=lambda r: r["conf"],
        )
        for rec in pool[:PER_CLASS]:
            chosen.append(rec)
            seen.add(rec["idx"])

    # And one confident example per class, so the gallery is not all edge cases.
    for cls in range(len(CLASSES)):
        pool = sorted(
            (r for r in scored if r["true"] == cls and r["correct"] and r["idx"] not in seen),
            key=lambda r: -r["conf"],
        )
        if pool:
            chosen.append(pool[0])
            seen.add(pool[0]["idx"])

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in SAMPLES_DIR.glob("*.png"):
        old.unlink()

    rows = []
    print(f"Writing {len(chosen)} sample figures to {SAMPLES_DIR}\n")

    for n, rec in enumerate(chosen, start=1):
        row = ds.rows[rec["idx"]]
        raw = Image.open(ds.image_path(row)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        raw_np = np.array(raw)

        x, _ = ds[rec["idx"]]
        cam, pred_idx, probs = gradcam_autograd(model, x.unsqueeze(0))
        overlay = cam_to_overlay(cam, raw_np)

        correct = pred_idx == rec["true"]
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.9))
        axes[0].imshow(raw_np)
        axes[0].set_title(f"Input\nactual: {CLASSES[rec['true']]}", fontsize=10)
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM (regions driving the prediction)", fontsize=10)
        for ax in axes:
            ax.axis("off")

        breakdown = "   ".join(f"{c}: {p:.3f}" for c, p in zip(CLASSES, probs))
        verdict = "correct" if correct else "MISCLASSIFIED"
        fig.suptitle(
            f"{CLASSES[pred_idx]}  -  confidence {probs[pred_idx]:.3f}  ({verdict})",
            fontsize=12,
            color="#1a7f37" if correct else "#c02026",
        )
        fig.text(0.5, 0.035, breakdown, ha="center", fontsize=9, color="#444")
        fig.tight_layout(rect=(0, 0.07, 1, 0.97))

        tag = "ok" if correct else "err"
        out = SAMPLES_DIR / f"sample_{n:02d}_{tag}_{CLASSES[pred_idx].replace(' ','')}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)

        rows.append(
            {
                "figure": out.name,
                "source_image": row["filename"],
                "actual": CLASSES[rec["true"]],
                "predicted": CLASSES[pred_idx],
                "confidence": f"{probs[pred_idx]:.4f}",
                "correct": correct,
                **{f"p_{c.replace(' ','_')}": f"{p:.4f}" for c, p in zip(CLASSES, probs)},
            }
        )
        print(
            f"  {out.name:<44} actual {CLASSES[rec['true']]:<13} "
            f"pred {CLASSES[pred_idx]:<13} conf {probs[pred_idx]:.3f}"
        )

    with (RESULTS_DIR / "sample_predictions.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Contact sheet, so the README can show everything in one image.
    cols = 4
    n_rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(3.3 * cols, 3.5 * n_rows))
    for ax in np.array(axes).ravel():
        ax.axis("off")
    for ax, rec in zip(np.array(axes).ravel(), chosen):
        row = ds.rows[rec["idx"]]
        raw = Image.open(ds.image_path(row)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        x, _ = ds[rec["idx"]]
        cam, pred_idx, probs = gradcam_autograd(model, x.unsqueeze(0))
        ax.imshow(cam_to_overlay(cam, np.array(raw)))
        ok = pred_idx == rec["true"]
        ax.set_title(
            f"{CLASSES[pred_idx]}  {probs[pred_idx]:.2f}\nactual: {CLASSES[rec['true']]}",
            fontsize=8.5,
            color="#1a7f37" if ok else "#c02026",
        )
    fig.suptitle("FloraLens - Grad-CAM on held-out test images", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RESULTS_DIR / "plots" / "sample_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    n_err = sum(1 for r in rows if not r["correct"])
    print(f"\n{len(rows)} figures ({n_err} misclassified), CSV + contact sheet written.")


if __name__ == "__main__":
    main()
