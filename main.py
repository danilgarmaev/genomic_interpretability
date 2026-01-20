"""Entry point for interpretability analyses.

Example:
  .venv/bin/python main.py --sequences results/sequences.jsonl

Mechanistic (causal) analyses:
    .venv/bin/python main.py --sequences results/sequences.jsonl --patching
    .venv/bin/python main.py --sequences results/sequences.jsonl --ablation

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
    p.add_argument(
        "--attention",
        action="store_true",
        help="Also generate attention heatmaps (ref/alt/diff) around the variant.",
    )
    p.add_argument(
        "--token-window",
        type=int,
        default=20,
        help="Token window radius for attention heatmaps.",
    )
    p.add_argument(
        "--patching",
        action="store_true",
        help="Run activation patching around the variant.",
    )
    p.add_argument(
        "--patch-token-window",
        type=int,
        default=5,
        help="Token window radius (around the variant token) for activation patching.",
    )
    p.add_argument(
        "--patch-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices for patching (default: last layer).",
    )
    p.add_argument(
        "--ablation",
        action="store_true",
        help="Run attention head ablation on ALT and measure effect.",
    )
    p.add_argument(
        "--ablation-layer",
        type=int,
        default=None,
        help="Layer index for head ablation (default: last layer).",
    )
    return p.parse_args()


def main() -> None:
    _add_src_to_path()

    from interpret import (
        activation_patching,
        analyze_variant,
        attention_head_ablation,
        attention_visualization,
    )

    args = parse_args()
    outputs = analyze_variant(
        sequences_jsonl=Path(args.sequences),
        out_dir=Path(args.out_dir),
        variant_id=args.variant_id,
        flank=args.flank,
    )

    if args.attention or args.patching or args.ablation:
        import json

        seq_path = Path(args.sequences)
        records = [json.loads(l) for l in seq_path.read_text().splitlines() if l.strip()]
        rec_by_id = {str(r.get("id")): r for r in records}
        vid = str(outputs["selected_variant_id"])
        record = rec_by_id[vid]

        if args.attention:
            attn_outputs = attention_visualization(
                record,
                out_dir=Path(args.out_dir),
                token_window=int(args.token_window),
            )
            outputs.update(attn_outputs)

        if args.patching:
            patch_layers = None
            if args.patch_layers:
                patch_layers = [int(x.strip()) for x in args.patch_layers.split(",") if x.strip()]
            patch_outputs = activation_patching(
                record,
                out_dir=Path(args.out_dir),
                token_window=int(args.patch_token_window),
                layers=patch_layers,
            )
            outputs.update(patch_outputs)

        if args.ablation:
            ablation_outputs = attention_head_ablation(
                record,
                out_dir=Path(args.out_dir),
                layer=args.ablation_layer,
            )
            outputs.update(ablation_outputs)

    print("Wrote:")
    for k, v in outputs.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
