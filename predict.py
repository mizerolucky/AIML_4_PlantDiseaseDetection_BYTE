#!/usr/bin/env python3
"""FloraLens command-line inference.

Single image:

    python predict.py path/to/leaf.jpg

Whole folder, saving Grad-CAM overlays and a CSV:

    python predict.py path/to/folder --heatmap --out predictions/

The model was trained on potato leaves only (early blight, late blight,
healthy). Given a tomato leaf, a hand, or a photograph of a wall, it will still
return one of those three labels with a confidence attached, because a softmax
over three classes has nowhere else to go. Confidence is how strongly the image
resembles one of the three classes it knows, not a statement that the subject
is a potato leaf. --min-confidence flags the low-confidence cases.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CHECKPOINT, CLASSES, IMAGE_SIZE, MEAN, STD  # noqa: E402
from src.gradcam import cam_to_overlay, gradcam_autograd  # noqa: E402
from src.model import FloraLensNet  # noqa: E402

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(checkpoint: Path) -> FloraLensNet:
    if not checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}.\n"
            f"Train one with `python -m src.train`, or pass --checkpoint."
        )
    model = FloraLensNet(pretrained=False)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def preprocess(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Return the normalised tensor and the resized RGB image for overlaying."""
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    norm = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, np.asarray(img)


def gather(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.suffix.lower() in SUFFIXES)
        if not files:
            raise SystemExit(f"No images found under {target}")
        return files
    raise SystemExit(f"Not found: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify potato leaf images and optionally save Grad-CAM overlays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("target", type=Path, help="Image file or directory of images")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--heatmap", action="store_true", help="Save a Grad-CAM overlay per image")
    parser.add_argument("--out", type=Path, default=Path("predictions"), help="Output directory")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Confidence below which a prediction is flagged as uncertain",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    files = gather(args.target)
    if args.heatmap:
        args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    width = max((len(f.name) for f in files), default=10)
    width = min(width, 42)

    for path in files:
        tensor, raw = preprocess(path)
        cam, pred_idx, probs = gradcam_autograd(model, tensor)
        label = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        note = ""
        if confidence < args.min_confidence:
            note = "  [uncertain - may not be a potato leaf]"

        overlay_name = ""
        if args.heatmap:
            overlay_name = f"{path.stem}_gradcam.png"
            Image.fromarray(cam_to_overlay(cam, raw)).save(args.out / overlay_name)

        print(f"{path.name[:width]:<{width}}  {label:<13} {confidence:6.2%}{note}")
        rows.append(
            {
                "image": str(path),
                "predicted": label,
                "confidence": f"{confidence:.4f}",
                "uncertain": confidence < args.min_confidence,
                "gradcam": overlay_name,
                **{f"p_{c.replace(' ', '_')}": f"{p:.4f}" for c, p in zip(CLASSES, probs)},
            }
        )

    if len(files) > 1:
        args.out.mkdir(parents=True, exist_ok=True)
        csv_path = args.out / "predictions.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(rows)} images -> {csv_path}")
        if args.heatmap:
            print(f"Grad-CAM overlays -> {args.out}")


if __name__ == "__main__":
    main()
