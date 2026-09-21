"""Export the trained model to ONNX for the browser demo.

    python -m src.export_onnx

The exported graph has two outputs, logits and the 576x7x7 feature map,
because the browser needs both: the logits give the prediction, and the feature
map combined with the classifier weights gives the Grad-CAM (see src/model.py).

The classifier weights are written alongside as a JSON file, since the browser
needs them as plain numbers to compute the heatmap.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import torch

from .config import (
    CLASSES,
    IMAGE_SIZE,
    LABELS_JSON,
    MEAN,
    ONNX_MODEL,
    STD,
    WEB_MODEL_DIR,
)
from .evaluate import load_model
from .model import FloraLensExport


def main() -> None:
    device = torch.device("cpu")
    model = load_model(device)
    wrapper = FloraLensExport(model).eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    ONNX_MODEL.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(ONNX_MODEL),
        input_names=["input"],
        output_names=["logits", "features"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
            "features": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = ONNX_MODEL.stat().st_size / 1e6
    print(f"Exported {ONNX_MODEL} ({size_mb:.2f} MB)")

    # Parity check against PyTorch on random inputs. An export that silently
    # changes behaviour is the sort of thing you only notice in the demo.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    worst_logit = 0.0
    worst_feat = 0.0
    for _ in range(5):
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        onnx_logits, onnx_feats = sess.run(None, {in_name: x.numpy()})
        with torch.no_grad():
            t_logits, t_feats = model.forward_with_features(x)
        worst_logit = max(worst_logit, float(np.abs(onnx_logits - t_logits.numpy()).max()))
        worst_feat = max(worst_feat, float(np.abs(onnx_feats - t_feats.numpy()).max()))

    print(f"ONNX vs PyTorch: max |logit diff| {worst_logit:.3e}, max |feature diff| {worst_feat:.3e}")
    if worst_logit > 1e-4:
        raise SystemExit(f"ONNX export does not match PyTorch (logit diff {worst_logit:.3e})")

    # Everything the browser needs, in one file.
    meta = {
        "classes": CLASSES,
        "image_size": IMAGE_SIZE,
        "mean": MEAN,
        "std": STD,
        "feature_channels": int(model.classifier.weight.shape[1]),
        "classifier_weight": model.classifier.weight.detach().numpy().tolist(),
        "classifier_bias": model.classifier.bias.detach().numpy().tolist(),
    }

    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (WEB_MODEL_DIR / "meta.json").write_text(json.dumps(meta))
    shutil.copy(ONNX_MODEL, WEB_MODEL_DIR / "floralens.onnx")
    LABELS_JSON.write_text(json.dumps({"classes": CLASSES}, indent=2))

    print(f"Copied model and meta.json into {WEB_MODEL_DIR}")


if __name__ == "__main__":
    main()
