"""The FloraLens classifier.

The head is deliberately plain: global average pooling straight into a single
linear layer. That choice is what makes the browser demo honest.

With a GAP -> Linear head, Grad-CAM has a closed form. For class c:

    y_c = sum_k w_ck * (1/HW) * sum_ij A_kij + b_c
    dy_c / dA_kij = w_ck / HW
    alpha_k       = (1/HW) * sum_ij dy_c/dA_kij = w_ck / HW
    cam           = relu(sum_k alpha_k * A_k) = relu(sum_k w_ck * A_k) / HW

The 1/HW is a positive constant, so once the map is normalised to [0, 1] it
drops out. The browser can therefore produce a pixel-identical Grad-CAM from
the classifier weights and the feature map alone, with no autograd and no
backward pass. src/verify_gradcam.py checks that equality numerically against
a real autograd Grad-CAM instead of asking the reader to trust the algebra.

The stock MobileNetV3 head (Linear -> Hardswish -> Dropout -> Linear) would
break the equivalence, which is why it is replaced rather than reused.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import BACKBONE_WEIGHTS, CLASSES
from .mobilenetv3 import mobilenetv3_small

FEATURE_CHANNELS = 576  # MobileNetV3-Small feature-map depth after the 1x1 expansion


class FloraLensNet(nn.Module):
    """ImageNet-pretrained MobileNetV3-Small backbone, GAP, one linear layer.

    The backbone is `features` followed by `conv`, which together take a
    224x224 image to a 576-channel 7x7 feature map. The original 1000-class
    head is discarded.

    forward() returns logits only, so it behaves like any classifier.
    forward_with_features() also returns the feature map, which is what both
    the Grad-CAM code and the exported ONNX graph need.
    """

    def __init__(
        self,
        num_classes: int = len(CLASSES),
        pretrained: bool = True,
        weights_path: Path = BACKBONE_WEIGHTS,
    ) -> None:
        super().__init__()
        base = mobilenetv3_small()

        if pretrained:
            if not weights_path.exists():
                raise SystemExit(
                    f"Backbone weights not found at {weights_path}.\n"
                    f"Run: python scripts/fetch_backbone.py"
                )
            state = torch.load(weights_path, map_location="cpu")
            # strict=True on purpose: a silent partial load would leave part of
            # the backbone randomly initialised while still looking pretrained.
            base.load_state_dict(state, strict=True)

        self.backbone = nn.Sequential(base.features, base.conv)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(FEATURE_CHANNELS, num_classes)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)                      # (B, 576, 7, 7)
        pooled = self.pool(feats).flatten(1)          # (B, 576)
        logits = self.classifier(self.dropout(pooled))
        return logits, feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x)[0]


class FloraLensExport(nn.Module):
    """Inference wrapper used only for the ONNX export.

    It returns (logits, features) so the browser receives everything it needs
    for both the prediction and the heatmap from a single session.run().
    """

    def __init__(self, net: FloraLensNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net.forward_with_features(x)
