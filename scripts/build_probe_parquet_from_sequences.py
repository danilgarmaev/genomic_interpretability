"""Build `data/processed/my_processed_clinvar.parquet` from `results/sequences.jsonl`.

This keeps the repo reproducible without committing derived datasets.

Input (from scripts/prepare_dataset.py):
- results/sequences.jsonl with fields including: ref_seq, alt_seq, ref, alt, label

Output schema:
- ref: 1024bp reference window sequence
- alt: 1024bp alternate window sequence
- variant_type: coarse type (single_nucleotide_variant/Insertion/Deletion/Indel)
- CLNSIG: label (typically Pathogenic/Benign)

Example:
  ./.venv/bin/python scripts/build_probe_parquet_from_sequences.py \
    --sequences results/sequences.jsonl \
    --out data/processed/my_processed_clinvar.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", type=str, default="results/sequences.jsonl")
    p.add_argument("--out", type=str, default="data/processed/my_processed_clinvar.parquet")
    p.add_argument(
        "--only",
        type=str,
        default="Pathogenic,Benign",
        help="Comma-separated labels to keep (exact match). Set empty to keep all.",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def infer_variant_type(ref_allele: str, alt_allele: str) -> str:
    ref_allele = str(ref_allele or "")
    alt_allele = str(alt_allele or "")
    if len(ref_allele) == 1 and len(alt_allele) == 1:
        return "single_nucleotide_variant"
    if len(ref_allele) < len(alt_allele):
        return "Insertion"
    if len(ref_allele) > len(alt_allele):
        return "Deletion"
    return "Indel"


def main() -> None:
    args = parse_args()
    seq_path = Path(args.sequences)
    if not seq_path.exists():
        raise SystemExit(f"Missing sequences.jsonl: {seq_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keep_labels = [x.strip() for x in str(args.only).split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for rec in read_jsonl(seq_path):
        ref_seq = rec.get("ref_seq")
        alt_seq = rec.get("alt_seq")
        label = rec.get("label")
        if ref_seq is None or alt_seq is None:
            continue
        if keep_labels and label not in keep_labels:
            continue

        ref_allele = rec.get("ref")
        alt_allele = rec.get("alt")
        vtype = rec.get("variant_type")
        if not vtype:
            vtype = infer_variant_type(str(ref_allele), str(alt_allele))

        rows.append(
            {
                "ref": str(ref_seq).upper(),
                "alt": str(alt_seq).upper(),
                "variant_type": str(vtype),
                "CLNSIG": str(label) if label is not None else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No records produced. Check input file and --only filter.")

    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
