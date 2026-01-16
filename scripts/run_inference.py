"""Run minimal single-sequence inference.

PHASE 2: runs inference on one hard-coded example.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))


def main() -> None:
    _add_src_to_path()

    from infer import predict

    sequence = "ACGT" * 32
    logits = predict(sequence)

    print(f"Sequence length: {len(sequence)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Mean logit: {logits.float().mean().item():.6f}")


if __name__ == "__main__":
    main()
