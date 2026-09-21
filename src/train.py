"""Train the FloraLens classifier.

    python -m src.train

The test split is not touched here. Checkpoint selection uses validation macro
F1, and the test set is read exactly once, by src/evaluate.py, after training
has finished. Selecting on the test set is the most common way a project like
this ends up reporting a number it cannot reproduce on new data.

Macro F1 rather than accuracy is the selection metric because the classes are
imbalanced: a model that never predicts Healthy still scores about 93%
accuracy, while its macro F1 collapses.
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    CHECKPOINT,
    CLASSES,
    EPOCHS,
    LABELS_JSON,
    LR_BACKBONE,
    LR_HEAD,
    MEAN,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    STD,
    IMAGE_SIZE,
    WEIGHT_DECAY,
)
from .data import LeafDataset, class_weights
from .model import FloraLensNet


def run_epoch(model, loader, criterion, optimiser, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_preds: list[int] = []
    all_true: list[int] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(labels.cpu().tolist())

    n = len(all_true)
    accuracy = sum(int(p == t) for p, t in zip(all_preds, all_true)) / n
    macro_f1 = f1_score(all_true, all_preds, average="macro")
    return total_loss / n, accuracy, macro_f1


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = LeafDataset("train")
    val_ds = LeafDataset("val")
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = FloraLensNet(pretrained=True).to(device)

    weights = class_weights("train").to(device)
    print("Class weights: " + ", ".join(f"{c}={w:.3f}" for c, w in zip(CLASSES, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weights)

    # The backbone already knows edges, texture and colour from ImageNet, so it
    # only needs nudging. The head is random, so it gets the larger rate.
    optimiser = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    history = []
    best_f1 = -1.0
    best_epoch = -1
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimiser, device, True)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, optimiser, device, False)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "train_macro_f1": tr_f1,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "val_macro_f1": va_f1,
                "lr_backbone": optimiser.param_groups[0]["lr"],
            }
        )

        marker = ""
        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": IMAGE_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "epoch": epoch,
                    "val_macro_f1": va_f1,
                },
                CHECKPOINT,
            )
            marker = "  <- best so far, saved"

        print(
            f"epoch {epoch:2d}/{EPOCHS}  "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} macroF1 {va_f1:.4f}{marker}"
        )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min. Best val macro F1 {best_f1:.4f} at epoch {best_epoch}.")

    LABELS_JSON.write_text(json.dumps({"classes": CLASSES}, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "training_history.json").write_text(
        json.dumps(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "selection_metric": "val_macro_f1",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "elapsed_seconds": elapsed,
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
