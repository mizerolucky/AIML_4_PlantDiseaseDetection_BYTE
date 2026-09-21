"""Grad-CAM, implemented twice on purpose.

`gradcam_autograd` is the textbook version: hook the last convolutional block,
backpropagate the class score, average the gradients spatially to get the
channel weights. It is the reference.

`gradcam_closed_form` skips the backward pass and reads the channel weights
straight out of the classifier layer. This is valid only because the head is
global average pooling into a single linear layer (see src/model.py), and it is
what the JavaScript in web/app.js reproduces.

Keeping both lets src/verify_gradcam.py prove that the map shown in the browser
is the same map the reference implementation produces, rather than a lookalike.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import FloraLensNet


def normalise(cam: np.ndarray) -> np.ndarray:
    """Scale a map to [0, 1].

    A flat map (which happens when ReLU zeroes everything) would divide by zero,
    so that case returns zeros rather than NaNs.
    """
    cam = cam - cam.min()
    peak = cam.max()
    if peak <= 1e-12:
        return np.zeros_like(cam)
    return cam / peak


def gradcam_autograd(
    model: FloraLensNet, x: torch.Tensor, class_idx: int | None = None
) -> tuple[np.ndarray, int, np.ndarray]:
    """Reference Grad-CAM via a real backward pass.

    Returns (cam HxW in [0,1], predicted class index, softmax probabilities).
    """
    model.eval()
    feats_store: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        feats_store["value"] = out

    handle = model.backbone.register_forward_hook(hook)
    try:
        logits, _ = model.forward_with_features(x)
        probs = F.softmax(logits, dim=1)
        target = int(logits.argmax(dim=1).item()) if class_idx is None else class_idx

        model.zero_grad(set_to_none=True)
        logits[0, target].backward()

        feats = feats_store["value"]            # (1, C, H, W)
        grads = feats.grad                      # (1, C, H, W)
        alpha = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = F.relu((alpha * feats).sum(dim=1)).squeeze(0)
    finally:
        handle.remove()

    return (
        normalise(cam.detach().cpu().numpy()),
        target,
        probs.detach().cpu().numpy()[0],
    )


def gradcam_closed_form(
    model: FloraLensNet, x: torch.Tensor, class_idx: int | None = None
) -> tuple[np.ndarray, int, np.ndarray]:
    """Grad-CAM without a backward pass, using the classifier weights directly.

    This is the exact computation web/app.js performs on the ONNX outputs.
    """
    model.eval()
    with torch.no_grad():
        logits, feats = model.forward_with_features(x)
        probs = F.softmax(logits, dim=1)
        target = int(logits.argmax(dim=1).item()) if class_idx is None else class_idx

        w = model.classifier.weight[target].view(1, -1, 1, 1)   # (1, C, 1, 1)
        cam = F.relu((w * feats).sum(dim=1)).squeeze(0)

    return (
        normalise(cam.cpu().numpy()),
        target,
        probs.cpu().numpy()[0],
    )


def cam_to_overlay(
    cam: np.ndarray, image: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Blend a CAM over an RGB image using a blue -> red colour ramp.

    `image` is uint8 HxWx3. The CAM is upsampled to the image size first, so the
    overlay lines up with the leaf rather than the 7x7 feature grid.
    """
    h, w = image.shape[:2]
    cam_t = torch.from_numpy(cam)[None, None].float()
    cam_up = F.interpolate(cam_t, size=(h, w), mode="bilinear", align_corners=False)
    cam_up = cam_up.squeeze().numpy()

    # Piecewise ramp: blue (cold) -> green -> red (hot). Matches web/app.js.
    r = np.clip(1.5 * cam_up - 0.5, 0, 1)
    g = np.clip(1.5 - np.abs(3.0 * cam_up - 1.5), 0, 1)
    b = np.clip(1.0 - 1.5 * cam_up, 0, 1)
    heat = np.stack([r, g, b], axis=-1) * 255.0

    # Weight the blend by CAM intensity so cold regions keep the original pixels
    # and the leaf stays readable underneath the hot ones.
    strength = (alpha * cam_up)[..., None]
    blended = image.astype(np.float32) * (1 - strength) + heat * strength
    return np.clip(blended, 0, 255).astype(np.uint8)
