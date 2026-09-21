"""Build the train/val/test manifest and audit the split for leakage.

Run this once, before training:

    python -m src.prepare_data --source /path/to/PlantVillage/raw/color

It writes data/manifest.csv, which is the single source of truth for which
image belongs to which split. Training never globs the data directory, so a
re-run of training on the same manifest always sees the same split.

Two leakage audits run before the manifest is written:

1. Exact duplicates (MD5 of the file bytes). If the same image appears twice
   under different names and lands in both train and test, the test score is
   inflated.
2. Near-duplicates (difference hash). PlantVillage sometimes holds several
   photographs of the same physical leaf. Those are not byte-identical, so the
   MD5 pass cannot see them, but they are close enough that a model which has
   memorised one will recognise the other.

Both audits are reported in data/split_audit.json rather than silently fixed,
so the numbers in the README can be traced back to a check that actually ran.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .config import (
    CLASSES,
    DATA_DIR,
    MANIFEST,
    SEED,
    SOURCE_DIRS,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dhash(path: Path, size: int = 8) -> str:
    """Difference hash: a 64-bit perceptual fingerprint.

    The image is reduced to a (size+1) x size grayscale thumbnail and each bit
    records whether a pixel is brighter than its right-hand neighbour. Two
    photographs of the same leaf under slightly different framing produce
    hashes that differ in only a handful of bits.
    """
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(img.tobytes())  # mode "L" -> one byte per pixel, row-major
    bits = 0
    idx = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits |= (1 if left > right else 0) << idx
            idx += 1
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def collect(source: Path) -> list[dict]:
    """Read every image under the three potato folders."""
    records = []
    for src_name, class_name in SOURCE_DIRS.items():
        folder = source / src_name
        if not folder.is_dir():
            raise SystemExit(
                f"Expected folder not found: {folder}\n"
                f"Point --source at the directory that contains "
                f"{', '.join(SOURCE_DIRS)}."
            )
        files = sorted(p for p in folder.iterdir() if p.suffix in IMAGE_SUFFIXES)
        if not files:
            raise SystemExit(f"No images found in {folder}")
        for path in files:
            records.append(
                {
                    "filename": path.name,
                    "source_dir": src_name,
                    "label": class_name,
                    "label_index": CLASSES.index(class_name),
                    "path": path,
                }
            )
    return records


def split_records(records: list[dict]) -> None:
    """Assign a split to each record, stratified by class.

    Stratifying matters here because the healthy class has 152 images against
    1000 for each disease. A plain random split could leave the test set with
    barely any healthy leaves, and per-class recall for that class would then
    be measured on a handful of images.
    """
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["label"]].append(rec)

    for label in sorted(by_class):
        group = sorted(by_class[label], key=lambda r: r["filename"])
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # Whatever rounding leaves over goes to test, so the three counts
        # always add back up to n.
        for i, rec in enumerate(group):
            if i < n_train:
                rec["split"] = "train"
            elif i < n_train + n_val:
                rec["split"] = "val"
            else:
                rec["split"] = "test"


def audit(records: list[dict], near_dup_threshold: int) -> dict:
    """Look for identical or near-identical images that straddle two splits."""
    by_md5 = defaultdict(list)
    for rec in records:
        by_md5[rec["md5"]].append(rec)

    exact_groups = [g for g in by_md5.values() if len(g) > 1]
    exact_cross = [
        sorted({r["split"] for r in g}) for g in exact_groups if len({r["split"] for r in g}) > 1
    ]

    # Near-duplicate scan. 2152 images is small enough for the full pairwise
    # comparison (~2.3M pairs), which avoids the false negatives you get from
    # bucketing by hash prefix.
    near_cross = []
    items = [(r["dhash"], r["split"], r["filename"], r["label"]) for r in records]
    for i in range(len(items)):
        h1, s1, f1, l1 = items[i]
        for j in range(i + 1, len(items)):
            h2, s2, f2, l2 = items[j]
            if s1 == s2:
                continue
            if hamming(h1, h2) <= near_dup_threshold:
                near_cross.append(
                    {
                        "a": f1,
                        "a_split": s1,
                        "a_label": l1,
                        "b": f2,
                        "b_split": s2,
                        "b_label": l2,
                        "distance": hamming(h1, h2),
                    }
                )

    return {
        "total_images": len(records),
        "unique_md5": len(by_md5),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicates_crossing_splits": len(exact_cross),
        "near_duplicate_threshold_bits": near_dup_threshold,
        "near_duplicate_pairs_crossing_splits": len(near_cross),
        "near_duplicate_examples": near_cross[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Directory containing Potato___Early_blight, Potato___healthy, Potato___Late_blight",
    )
    parser.add_argument(
        "--near-dup-threshold",
        type=int,
        default=4,
        help="Max Hamming distance between difference hashes to call two images near-duplicates",
    )
    args = parser.parse_args()

    records = collect(args.source)
    print(f"Found {len(records)} images across {len(CLASSES)} classes")

    for rec in records:
        rec["md5"] = md5_of(rec["path"])
        rec["dhash"] = dhash(rec["path"])

    split_records(records)

    report = audit(records, args.near_dup_threshold)
    print(
        f"Audit: {report['unique_md5']} unique files, "
        f"{report['exact_duplicates_crossing_splits']} exact duplicates across splits, "
        f"{report['near_duplicate_pairs_crossing_splits']} near-duplicate pairs across splits"
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["filename", "source_dir", "label", "label_index", "split", "md5", "dhash"],
        )
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r["label"], r["filename"])):
            writer.writerow({k: rec[k] for k in writer.fieldnames})

    counts = Counter((r["split"], r["label"]) for r in records)
    summary = {
        split: {label: counts[(split, label)] for label in CLASSES}
        for split in ("train", "val", "test")
    }
    report["split_counts"] = summary

    (DATA_DIR / "split_audit.json").write_text(json.dumps(report, indent=2))

    print(f"\nWrote {MANIFEST}")
    for split in ("train", "val", "test"):
        total = sum(summary[split].values())
        detail = ", ".join(f"{label}: {summary[split][label]}" for label in CLASSES)
        print(f"  {split:5s} {total:5d}  ({detail})")


if __name__ == "__main__":
    main()
