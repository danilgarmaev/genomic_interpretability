"""Entry point for interpretability analyses.

Example:
  .venv/bin/python main.py --sequences results/sequences.jsonl

Optional:
  .venv/bin/python main.py --sequences results/sequences.jsonl --variant-id <ID>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sequences",
        type=str,
        default="results/sequences.jsonl",
        help="Path to results/sequences.jsonl produced by scripts/prepare_dataset.py",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory for CSVs and plots.",
    )
    p.add_argument(
        "--variant-id",
        type=str,
        default=None,
        help="Variant id to analyze (defaults to most impactful by impact_score).",
    )
    p.add_argument("--flank", type=int, default=50, help="bp flank around the variant")
    return p.parse_args()


def main() -> None:
    _add_src_to_path()

    from interpret import analyze_variant

    args = parse_args()
    outputs = analyze_variant(
        sequences_jsonl=Path(args.sequences),
        out_dir=Path(args.out_dir),
        variant_id=args.variant_id,
        flank=args.flank,
    )

    print("Wrote:")
    for k, v in outputs.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
