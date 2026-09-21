"""Check the browser's Grad-CAM against PyTorch, by running the real JavaScript.

    pip install -r requirements-dev.txt
    python scripts/verify_browser_gradcam.py

src/verify_gradcam.py compares two Python implementations and the ONNX graph.
It does not execute web/app.js, so on its own it cannot say anything about the
code the browser actually runs -- a transcription error in the JavaScript would
pass every one of its checks.

This script closes that gap. It serves web/, drives a headless Chromium, waits
for the page to load the model, then calls the page's own computeCam() with a
feature map produced by PyTorch and compares the returned map against the
PyTorch autograd Grad-CAM for the same tensor.

Scope, stated precisely: this verifies the CAM computation given an identical
feature map. It deliberately does not compare end-to-end output for an uploaded
file, because image decoding and resizing genuinely differ between PIL and the
browser canvas -- see the note printed at the end.

Writes results/browser_gradcam_verification.json.
"""

from __future__ import annotations

import functools
import http.server
import json
import socketserver
import sys
import threading
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR  # noqa: E402
from src.data import LeafDataset  # noqa: E402
from src.evaluate import load_model  # noqa: E402
from src.gradcam import gradcam_autograd  # noqa: E402

N_IMAGES = 12
PORT = 8799
TOLERANCE = 1e-4


def serve(directory: Path) -> socketserver.TCPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "playwright is not installed.\n"
            "  pip install -r requirements-dev.txt && playwright install chromium"
        )

    model = load_model(torch.device("cpu"))
    ds = LeafDataset("test", train_mode=False)
    n = min(N_IMAGES, len(ds))

    httpd = serve(ROOT / "web")
    print(f"Serving web/ on port {PORT}; comparing {n} test images\n")

    cam_diffs: list[float] = []
    softmax_diffs: list[float] = []
    console_errors: list[str] = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.on("pageerror", lambda e: console_errors.append(str(e)))
            page.on(
                "console",
                lambda m: console_errors.append(m.text) if m.type == "error" else None,
            )

            page.goto(f"http://127.0.0.1:{PORT}/", wait_until="domcontentloaded")
            page.wait_for_selector("#status.ready", timeout=180_000)
            page.wait_for_function("window.__floralens !== undefined", timeout=30_000)
            print("Page reports the model is ready.\n")

            for i in range(n):
                x, _ = ds[i]
                xb = x.unsqueeze(0)

                # PyTorch: the reference map, and the feature tensor to hand over.
                cam_ref, cls_ref, probs_ref = gradcam_autograd(model, xb)
                with torch.no_grad():
                    logits, feats = model.forward_with_features(xb)

                dims = [1, *feats.shape[1:]]
                payload = {
                    "features": feats.flatten().tolist(),
                    "dims": [int(d) for d in dims],
                    "classIndex": int(cls_ref),
                    "logits": logits[0].tolist(),
                }

                # The page's own code, not a reimplementation of it.
                result = page.evaluate(
                    """(p) => {
                        const out = window.__floralens.computeCam(
                            Float32Array.from(p.features), p.dims, p.classIndex);
                        return {
                            cam: Array.from(out.cam),
                            height: out.height,
                            width: out.width,
                            probs: window.__floralens.softmax(p.logits),
                        };
                    }""",
                    payload,
                )

                cam_js = np.array(result["cam"], dtype=np.float64).reshape(
                    result["height"], result["width"]
                )
                cam_diffs.append(float(np.abs(cam_ref - cam_js).max()))
                softmax_diffs.append(
                    float(np.abs(np.array(result["probs"]) - probs_ref).max())
                )

                print(
                    f"  image {i + 1:2d}/{n}  max |CAM diff| {cam_diffs[-1]:.3e}   "
                    f"max |softmax diff| {softmax_diffs[-1]:.3e}"
                )

            browser.close()
    finally:
        httpd.shutdown()

    passed = (
        max(cam_diffs) < TOLERANCE
        and max(softmax_diffs) < TOLERANCE
        and not console_errors
    )

    report = {
        "what_this_checks": (
            "web/app.js computeCam() and softmax(), executed in Chromium, against "
            "PyTorch autograd Grad-CAM on the same feature tensor"
        ),
        "not_checked": (
            "end-to-end output for an uploaded file: PIL and the browser canvas "
            "resize images differently, and the 7x7 map is upscaled for display by "
            "the browser, so displayed pixels are not expected to match bit for bit"
        ),
        "n_images": n,
        "max_abs_cam_difference": max(cam_diffs),
        "mean_abs_cam_difference": float(np.mean(cam_diffs)),
        "max_abs_softmax_difference": max(softmax_diffs),
        "console_errors": console_errors,
        "tolerance": TOLERANCE,
        "passed": bool(passed),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "browser_gradcam_verification.json").write_text(json.dumps(report, indent=2))

    print(f"\nmax |CAM difference| over {n} images : {max(cam_diffs):.3e}")
    print(f"max |softmax difference|             : {max(softmax_diffs):.3e}")
    print(f"console errors                       : {len(console_errors)}")
    print(f"\nVerdict: {'PASS' if passed else 'FAIL'} (tolerance {TOLERANCE:g})")
    print("Wrote results/browser_gradcam_verification.json")
    print(
        "\nScope: this compares the CAM computation given an identical feature map.\n"
        "Image decoding and resizing differ between PIL and the browser canvas, so\n"
        "the heatmap shown for an uploaded file can differ slightly from the CLI's."
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
