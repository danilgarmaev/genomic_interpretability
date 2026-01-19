"""Dataset loading and variant-to-sequence preprocessing.

PHASE 3: load variants, sample a small subset, and construct ref/alt sequence windows
from a local reference genome FASTA.

Notes:
- This is intentionally minimal and designed for a time-constrained assessment.
- The default upstream source can be bowang-lab/genomic-FM, but formats may vary;
  the loader is flexible for common tabular formats and provides TODO guidance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd


@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int
    ref: str
    alt: str
    label: Optional[Any] = None
    id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _normalize_chrom(chrom: Any) -> str:
    chrom_str = str(chrom)
    return chrom_str.strip()


def _infer_format(path_or_url: str) -> str:
    lower = path_or_url.lower()
    for ext in (".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"):
        if lower.endswith(ext):
            return ext.lstrip(".")
    return ""


def load_variants(path_or_url: str) -> List[Variant]:
    """Load variants from a local path or URL.

    Supports: CSV/TSV/TXT/Parquet/JSON/JSONL.

    Expected columns (best effort):
      - chrom, pos, ref, alt
    Optional:
      - label, id

    If your dataset uses different column names, rename them before loading or
    edit the mapping below.

    TODO (genomic-FM): if you point this at bowang-lab/genomic-FM artifacts and the
    schema differs, adjust the column mapping here.
    """

    fmt = _infer_format(path_or_url)
    if not fmt:
        raise ValueError(
            "Unsupported file type for path_or_url. Supported: .csv .tsv .txt .parquet .json .jsonl"
        )

    if fmt in {"csv", "tsv", "txt"}:
        sep = "," if fmt == "csv" else "\t"
        df = pd.read_csv(path_or_url, sep=sep)
    elif fmt == "parquet":
        df = pd.read_parquet(path_or_url)
    elif fmt in {"json", "jsonl"}:
        df = pd.read_json(path_or_url, lines=(fmt == "jsonl"))
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if df.empty:
        return []

    # Common alternative column names
    rename_map = {
        "chr": "chrom",
        "chromosome": "chrom",
        "position": "pos",
        "start": "pos",
        "reference": "ref",
        "ref_allele": "ref",
        "alternate": "alt",
        "alt_allele": "alt",
        "variant_id": "id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"chrom", "pos", "ref", "alt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
            + ". Present columns: "
            + ", ".join(map(str, df.columns))
            + ".\n"
            + "If using genomic-FM and the schema differs, update the rename_map in src/data.py."
        )

    variants: List[Variant] = []
    for _, row in df.iterrows():
        chrom = _normalize_chrom(row["chrom"])
        pos = int(row["pos"])
        ref = str(row["ref"]).strip().upper()
        alt = str(row["alt"]).strip().upper()
        label = row["label"] if "label" in df.columns else None
        var_id = str(row["id"]) if "id" in df.columns and pd.notna(row["id"]) else None

        extra = {}
        for col in df.columns:
            if col in {"chrom", "pos", "ref", "alt", "label", "id"}:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            extra[col] = val

        variants.append(
            Variant(
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                label=label,
                id=var_id,
                extra=extra,
            )
        )

    return variants


def select_subset(
    variants: Sequence[Variant],
    n: int = 30,
    seed: int = 42,
    stratify_by: str = "label",
) -> List[Variant]:
    """Select a small representative subset.

    If stratify_by exists (currently only 'label' supported via Variant.label) and has
    multiple classes, sample approximately balanced across classes.
    """

    if n <= 0:
        return []

    if not variants:
        return []

    rng = pd.RandomState(seed)

    if stratify_by == "label":
        labeled = [v for v in variants if v.label is not None and str(v.label) != "nan"]
        if labeled:
            df = pd.DataFrame(
                {
                    "idx": list(range(len(labeled))),
                    "label": [v.label for v in labeled],
                }
            )
            classes = list(df["label"].unique())
            if len(classes) > 1:
                per_class = max(1, n // len(classes))
                picked: List[Variant] = []
                for cls in classes:
                    cls_rows = df[df["label"] == cls]
                    take = min(per_class, len(cls_rows))
                    chosen = cls_rows.sample(n=take, random_state=rng)
                    picked.extend([labeled[int(i)] for i in chosen["idx"].tolist()])

                # Fill any remaining slots from the labeled pool.
                if len(picked) < n:
                    remaining = [v for v in labeled if v not in picked]
                    if remaining:
                        extra_take = min(n - len(picked), len(remaining))
                        extra_idx = rng.choice(len(remaining), size=extra_take, replace=False)
                        picked.extend([remaining[int(i)] for i in extra_idx])

                return picked[:n]

    # Fallback: random sample
    take = min(n, len(variants))
    idx = rng.choice(len(variants), size=take, replace=False)
    return [variants[int(i)] for i in idx]


def build_ref_alt_sequences(
    variant: Variant,
    genome_fasta_path: Union[str, Path],
    window: int = 1000,
) -> Dict[str, Any]:
    """Construct reference and alternate sequence windows for one variant.

    Returns a dict:
      {
        "id", "chrom", "pos", "ref", "alt", "label",
        "ref_seq", "alt_seq", "variant_index"
      }

    Assumptions:
    - variant.pos is 1-based.
    - window is the desired final length for both sequences.

    For indels, the alternate sequence is trimmed/padded on the right to keep a fixed
    window length.
    """

    if window <= 0:
        raise ValueError("window must be a positive integer")

    fasta_path = Path(genome_fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(
            f"Reference genome FASTA not found at: {fasta_path}. "
            "Provide a local FASTA path via --genome-fasta."
        )

    try:
        from pyfaidx import Fasta  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyfaidx is required for FASTA access. Install it with `pip install pyfaidx`."
        ) from exc

    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

    chrom = variant.chrom
    if chrom not in fasta:
        # Try common normalization: add/remove 'chr'
        if chrom.startswith("chr") and chrom[3:] in fasta:
            chrom = chrom[3:]
        elif ("chr" + chrom) in fasta:
            chrom = "chr" + chrom
        else:
            raise KeyError(
                f"Chromosome '{variant.chrom}' not found in FASTA. "
                f"Available examples: {list(fasta.keys())[:5]}"
            )

    chrom_len = len(fasta[chrom])

    half = window // 2
    start_1 = variant.pos - half
    if start_1 < 1:
        start_1 = 1
    end_1 = start_1 + window - 1
    if end_1 > chrom_len:
        end_1 = chrom_len
        start_1 = max(1, end_1 - window + 1)

    start0 = start_1 - 1
    end0 = start0 + window

    ref_seq = str(fasta[chrom][start0:end0])
    if len(ref_seq) != window:
        # Near contig boundaries: pad with Ns
        if len(ref_seq) < window:
            ref_seq = ref_seq + ("N" * (window - len(ref_seq)))
        else:
            ref_seq = ref_seq[:window]

    variant_index = variant.pos - start_1
    if variant_index < 0 or variant_index >= window:
        raise ValueError(
            f"Variant position {variant.pos} is outside the extracted window [{start_1}, {end_1}]."
        )

    ref_len = len(variant.ref)
    ref_in_window = ref_seq[variant_index : variant_index + ref_len]
    if ref_in_window != variant.ref:
        raise ValueError(
            "Reference allele does not match genome at position. "
            f"chrom={chrom} pos={variant.pos} expected_ref={variant.ref} genome_ref={ref_in_window} "
            f"window_start={start_1}"
        )

    alt_seq = ref_seq[:variant_index] + variant.alt + ref_seq[variant_index + ref_len :]

    # Enforce fixed window length
    if len(alt_seq) < window:
        alt_seq = alt_seq + ("N" * (window - len(alt_seq)))
    elif len(alt_seq) > window:
        alt_seq = alt_seq[:window]

    return {
        "id": variant.id or f"{chrom}:{variant.pos}:{variant.ref}>{variant.alt}",
        "chrom": chrom,
        "pos": variant.pos,
        "ref": variant.ref,
        "alt": variant.alt,
        "label": variant.label,
        "ref_seq": ref_seq,
        "alt_seq": alt_seq,
        "variant_index": int(variant_index),
    }


def write_jsonl(records: Iterable[Dict[str, Any]], path: Union[str, Path]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
