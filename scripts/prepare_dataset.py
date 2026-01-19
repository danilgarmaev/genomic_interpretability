"""Prepare a small variant subset and build ref/alt sequences.

Outputs:
- results/variants_subset.csv
- results/sequences.jsonl

PHASE 3: dataset loading + variant-to-sequence preprocessing only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variants",
        required=True,
        help="Path or URL to variants file (.csv/.tsv/.parquet/.json/.jsonl/.vcf).",
    )
    p.add_argument(
        "--gene",
        default=None,
        help="Optional gene symbol filter (e.g., TP53) when variants come from ClinVar VCF.",
    )
    p.add_argument(
        "--vcf-limit",
        type=int,
        default=5000,
        help="Max number of VCF records to load after filtering (keeps runs fast).",
    )
    p.add_argument(
        "--genome-fasta",
        required=True,
        help="Local reference genome FASTA path (indexed .fai recommended).",
    )
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main() -> None:
    _add_src_to_path()

    from data import build_ref_alt_sequences, load_variants, select_subset, write_jsonl

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = load_variants(
        args.variants,
        gene=(str(args.gene) if args.gene else None),
        limit=(int(args.vcf_limit) if args.vcf_limit and str(args.variants).lower().endswith((".vcf", ".vcf.gz")) else None),
    )
    if not variants:
        raise SystemExit("No variants loaded. Check the input file.")

    subset = select_subset(variants, n=args.n, seed=args.seed, stratify_by="label")
    if not subset:
        raise SystemExit("Subset selection produced 0 variants.")

    seq_records = []
    meta_rows = []
    for v in subset:
        rec = build_ref_alt_sequences(v, genome_fasta_path=args.genome_fasta, window=args.window)
        seq_records.append(rec)
        meta_rows.append(
            {
                "id": rec["id"],
                "chrom": rec["chrom"],
                "pos": rec["pos"],
                "ref": rec["ref"],
                "alt": rec["alt"],
                "label": rec["label"],
                "variant_index": rec["variant_index"],
            }
        )

    pd.DataFrame(meta_rows).to_csv(out_dir / "variants_subset.csv", index=False)
    write_jsonl(seq_records, out_dir / "sequences.jsonl")

    print(f"Wrote {len(seq_records)} variants to {out_dir}/sequences.jsonl")


if __name__ == "__main__":
    main()
