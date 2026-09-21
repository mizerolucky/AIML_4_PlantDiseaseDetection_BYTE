"""Dataset and transforms, driven entirely by data/manifest.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import CLASSES, IMAGE_SIZE, MANIFEST, MEAN, RAW_DIR, STD


def build_transforms(train: bool) -> transforms.Compose:
    """Augmentation for training, plain resize for everything else.

    The augmentations are all label-preserving for this problem: a leaf is the
    same leaf flipped, rotated or photographed under warmer light. Nothing here
    changes lesion shape or colour enough to turn early blight into late blight,
    which would poison the labels.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


class LeafDataset(Dataset):
    """Reads the rows of manifest.csv belonging to one split."""

    def __init__(self, split: str, train_mode: bool | None = None, root: Path = RAW_DIR) -> None:
        if not MANIFEST.exists():
            raise SystemExit(
                f"{MANIFEST} not found. Run `python -m src.prepare_data --source ...` first."
            )
        self.root = root
        self.split = split
        self.transform = build_transforms(
            train_mode if train_mode is not None else (split == "train")
        )
        with MANIFEST.open() as fh:
            self.rows = [r for r in csv.DictReader(fh) if r["split"] == split]
        if not self.rows:
            raise SystemExit(f"No rows for split={split!r} in {MANIFEST}")

    def image_path(self, row: dict) -> Path:
        return self.root / row["source_dir"] / row["filename"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(self.image_path(row)).convert("RGB")
        return self.transform(img), int(row["label_index"])


def class_weights(split: str = "train") -> torch.Tensor:
    """Inverse-frequency weights, normalised to average 1.

    Healthy leaves are 152 of 2152 images. Without this, the cheapest way for
    the model to cut its loss is to under-predict the rare class, which is the
    opposite of what matters: a missed healthy leaf is a false disease alarm.
    """
    with MANIFEST.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == split]
    counts = torch.zeros(len(CLASSES))
    for row in rows:
        counts[int(row["label_index"])] += 1
    weights = counts.sum() / (len(CLASSES) * counts)
    return weights / weights.mean()
