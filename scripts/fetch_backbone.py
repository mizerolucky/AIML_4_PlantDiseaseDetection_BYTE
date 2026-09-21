"""Download the ImageNet-pretrained MobileNetV3-Small backbone weights.

    python scripts/fetch_backbone.py

The weights are a third-party artifact (MIT, see THIRD_PARTY_LICENSES.txt), so
they are fetched rather than committed. The download is checked against a
recorded MD5 so a truncated or substituted file fails loudly here instead of
quietly degrading training.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BACKBONE_MD5, BACKBONE_URL, BACKBONE_WEIGHTS  # noqa: E402


def main() -> None:
    BACKBONE_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    if BACKBONE_WEIGHTS.exists():
        digest = hashlib.md5(BACKBONE_WEIGHTS.read_bytes()).hexdigest()
        if digest == BACKBONE_MD5:
            print(f"Already present and verified: {BACKBONE_WEIGHTS}")
            return
        print(f"Existing file has wrong MD5 ({digest}); re-downloading.")

    print(f"Downloading {BACKBONE_URL}")
    urllib.request.urlretrieve(BACKBONE_URL, BACKBONE_WEIGHTS)

    digest = hashlib.md5(BACKBONE_WEIGHTS.read_bytes()).hexdigest()
    if digest != BACKBONE_MD5:
        BACKBONE_WEIGHTS.unlink(missing_ok=True)
        raise SystemExit(
            f"MD5 mismatch: expected {BACKBONE_MD5}, got {digest}. File deleted."
        )

    size_mb = BACKBONE_WEIGHTS.stat().st_size / 1e6
    print(f"Saved {BACKBONE_WEIGHTS} ({size_mb:.1f} MB), MD5 verified.")


if __name__ == "__main__":
    main()
