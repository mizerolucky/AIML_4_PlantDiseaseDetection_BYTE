"""Check the closed-form Grad-CAM against the reference implementation.

    python -m src.verify_gradcam

The web demo cannot run a backward pass, so it computes the heatmap from the
classifier weights and the feature map (see src/gradcam.py). That shortcut is
only valid for a GAP -> Linear head. This script checks the claim rather than
asserting it, by running both implementations over real test images and
comparing the resulting maps pixel by pixel.

It also compares the PyTorch model against the exported ONNX model, so the
weights shipped to the browser are covered too.

What this script does NOT cover: web/app.js. Everything here is Python, so a
transcription error in the JavaScript would pass every check below. That is
what scripts/verify_browser_gradcam.py is for -- it runs the page's own code in
a real browser.

Writes results/gradcam_verification.json.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from .config import CLASSES, ONNX_MODEL, RESULTS_DIR
from .data import LeafDataset
from .evaluate import load_model
from .gradcam import gradcam_autograd, gradcam_closed_form, normalise

N_IMAGES = 60


def main() -> None:
    device = torch.device("cpu")
    model = load_model(device)

    ds = LeafDataset("test", train_mode=False)
    n = min(N_IMAGES, len(ds))
    print(f"Comparing Grad-CAM implementations over {n} test images\n")

    cam_diffs: list[float] = []
    prob_diffs: list[float] = []
    class_mismatches = 0

    for i in range(n):
        x, _ = ds[i]
        x = x.unsqueeze(0)
        cam_ref, cls_ref, probs_ref = gradcam_autograd(model, x)
        cam_cf, cls_cf, probs_cf = gradcam_closed_form(model, x)

        if cls_ref != cls_cf:
            class_mismatches += 1
        cam_diffs.append(float(np.abs(cam_ref - cam_cf).max()))
        prob_diffs.append(float(np.abs(probs_ref - probs_cf).max()))

    result = {
        "n_images": n,
        "autograd_vs_closed_form": {
            "max_abs_cam_difference": max(cam_diffs),
            "mean_abs_cam_difference": float(np.mean(cam_diffs)),
            "max_abs_probability_difference": max(prob_diffs),
            "predicted_class_mismatches": class_mismatches,
        },
    }

    print("Autograd Grad-CAM vs closed-form Grad-CAM (the method app.js uses):")
    print(f"  max  |difference| over all pixels : {max(cam_diffs):.3e}")
    print(f"  mean |difference| over all pixels : {np.mean(cam_diffs):.3e}")
    print(f"  predicted-class mismatches        : {class_mismatches} / {n}")

    # Second check: does the exported ONNX graph still agree with PyTorch?
    if ONNX_MODEL.exists():
        import onnxruntime as ort

        sess = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        w = model.classifier.weight.detach().numpy()

        onnx_cam_diffs: list[float] = []
        onnx_logit_diffs: list[float] = []

        for i in range(n):
            x, _ = ds[i]
            xb = x.unsqueeze(0)
            logits_onnx, feats_onnx = sess.run(None, {in_name: xb.numpy()})

            with torch.no_grad():
                logits_t, feats_t = model.forward_with_features(xb)
            onnx_logit_diffs.append(float(np.abs(logits_onnx - logits_t.numpy()).max()))

            # The same arithmetic web/app.js performs, written in Python. This
            # is a transcription of it, not the file itself; the browser script
            # is what actually exercises app.js.
            cls = int(np.argmax(logits_onnx[0]))
            cam_js = np.maximum((w[cls][:, None, None] * feats_onnx[0]).sum(axis=0), 0)
            cam_js = normalise(cam_js)

            cam_ref, _, _ = gradcam_autograd(model, xb)
            onnx_cam_diffs.append(float(np.abs(cam_ref - cam_js).max()))

        result["onnx_vs_pytorch"] = {
            "max_abs_logit_difference": max(onnx_logit_diffs),
            "max_abs_cam_difference": max(onnx_cam_diffs),
        }
        print("\nONNX graph vs PyTorch autograd Grad-CAM:")
        print(f"  max |logit difference| : {max(onnx_logit_diffs):.3e}")
        print(f"  max |CAM difference|   : {max(onnx_cam_diffs):.3e}")
    else:
        print(f"\n{ONNX_MODEL} not found; skipping the ONNX comparison.")
        print("Run `python -m src.export_onnx` then re-run this script.")

    # float32 arithmetic reorders differently across implementations, so exact
    # bitwise equality is not the right bar; 1e-4 on a [0,1] map is far below
    # anything visible in a rendered heatmap.
    tolerance = 1e-4
    checks = [max(cam_diffs) < tolerance, class_mismatches == 0]
    if "onnx_vs_pytorch" in result:
        checks.append(result["onnx_vs_pytorch"]["max_abs_cam_difference"] < tolerance)
    result["tolerance"] = tolerance
    result["passed"] = bool(all(checks))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gradcam_verification.json").write_text(json.dumps(result, indent=2))

    print(f"\nVerdict: {'PASS' if result['passed'] else 'FAIL'} (tolerance {tolerance:g})")
    print(f"Wrote {RESULTS_DIR/'gradcam_verification.json'}")

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
