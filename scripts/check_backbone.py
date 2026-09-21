"""Re-run the backbone provenance checks referenced in src/mobilenetv3.py.

    python scripts/check_backbone.py

Two things are checked:

1. The downloaded weights load into the vendored architecture with
   strict=True, so no tensor is left randomly initialised.
2. The same weights positionally remapped into torchvision's
   mobilenet_v3_small do NOT reproduce the same outputs. This is the check
   that justified vendoring: the shapes line up well enough that the remap
   looks correct, and the outputs show that it is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BACKBONE_WEIGHTS  # noqa: E402
from src.mobilenetv3 import mobilenetv3_small  # noqa: E402


def main() -> None:
    if not BACKBONE_WEIGHTS.exists():
        raise SystemExit(f"{BACKBONE_WEIGHTS} missing. Run scripts/fetch_backbone.py first.")

    state = torch.load(BACKBONE_WEIGHTS, map_location="cpu")
    vendored = mobilenetv3_small()
    vendored.load_state_dict(state, strict=True)
    vendored.eval()
    print(f"[1/2] Vendored architecture loads all {len(state)} tensors with strict=True.")

    try:
        from torchvision.models import mobilenet_v3_small
    except ImportError:
        print("[2/2] torchvision unavailable; skipping the comparison.")
        return

    tv = mobilenet_v3_small(weights=None)
    tsd = tv.state_dict()
    if len(tsd) != len(state):
        print(f"[2/2] Tensor counts differ ({len(tsd)} vs {len(state)}); remap not applicable.")
        return

    remapped = {}
    for tv_key, d_key in zip(tsd.keys(), state.keys()):
        tensor = state[d_key]
        if tsd[tv_key].shape != tensor.shape and tensor.dim() == 2:
            # squeeze-excitation: linear (out, in) -> 1x1 conv (out, in, 1, 1)
            tensor = tensor.view(*tsd[tv_key].shape)
        remapped[tv_key] = tensor

    tv.load_state_dict(remapped, strict=True)
    tv.eval()

    torch.manual_seed(0)
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        a, b = vendored(x), tv(x)

    diff = (a - b).abs().max().item()
    same_top1 = bool((a.argmax(1) == b.argmax(1)).all())
    print(f"[2/2] Positional remap into torchvision: max |logit diff| = {diff:.4f}, "
          f"top-1 agrees = {same_top1}")

    if diff < 1e-3:
        print("      Unexpected: the remap reproduces the vendored model. "
              "The vendoring note in src/mobilenetv3.py would need revisiting.")
    else:
        print("      As documented: the architectures are not weight-compatible, "
              "so the vendored implementation is the correct one to use.")


if __name__ == "__main__":
    main()
