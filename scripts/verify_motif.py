"""Verify a candidate splice motif at the centered position for a given variant.

This script loads ref/alt sequences for a variant (default: 1121868), extracts the
centered 10-nt window (variant windows are centered by construction), prints the
10-mers with the center 2 bases capitalized, and checks for canonical splice motifs:
- Splice Donor: GT
- Splice Acceptor: AG

Example:
  .venv/bin/python scripts/verify_motif.py --variant-id 1121868 --sequences results/sequences.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant-id",
        type=str,
        default="1121868",
        help="Variant id to inspect (matches the 'id' field in sequences.jsonl).",
    )
    p.add_argument(
        "--sequences",
        type=str,
        default="results/sequences.jsonl",
        help="Path to sequences.jsonl produced by scripts/prepare_dataset.py.",
    )
    p.add_argument(
        "--center-window",
        type=int,
        default=10,
        help="Number of nucleotides to extract around the center (default: 10).",
    )
    return p.parse_args()


def _load_record(sequences_jsonl: Path, variant_id: str) -> Dict[str, Any]:
    for line in sequences_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if str(rec.get("id")) == str(variant_id):
            return rec
    raise ValueError(f"Variant id {variant_id} not found in {sequences_jsonl}")


def _center_index(rec: Dict[str, Any], seq_len: int) -> int:
    # Prefer the explicit index written during window construction.
    if "variant_index" in rec and rec["variant_index"] is not None:
        try:
            idx = int(rec["variant_index"])
            if 0 <= idx < seq_len:
                return idx
        except Exception:
            pass
    return seq_len // 2


def _extract_center_window(seq: str, center: int, window: int) -> str:
    if window <= 0 or window % 2 != 0:
        raise ValueError("--center-window must be a positive even number (e.g., 10).")
    half = window // 2
    start = max(0, center - half)
    end = min(len(seq), center + half)
    s = seq[start:end]

    # If we hit sequence boundaries, pad with N to keep length fixed.
    if len(s) < window:
        left_pad = max(0, (center - half) - 0)
        # left_pad will be 0 in our windowed data, but keep logic explicit.
        left_needed = max(0, window - len(s))
        # Prefer padding on the side that was clipped.
        if start == 0:
            s = ("N" * left_needed) + s
        elif end == len(seq):
            s = s + ("N" * left_needed)
        else:
            s = s.ljust(window, "N")

    # Ensure exact size.
    if len(s) != window:
        s = s[:window].ljust(window, "N")
    return s


def _capitalize_center2(tenmer: str) -> str:
    if len(tenmer) < 2:
        return tenmer.upper()
    mid = len(tenmer) // 2
    # For 10-mer, this uppercases positions 4 and 5 (0-based), matching atgcGTatgc.
    a = mid - 1
    b = mid + 1
    return tenmer[:a] + tenmer[a:b].upper() + tenmer[b:]


def _center2(tenmer: str) -> str:
    mid = len(tenmer) // 2
    return tenmer[mid - 1 : mid + 1].upper()


def _motif_type(two_mer: str) -> Optional[str]:
    if two_mer == "GT":
        return "Splice Donor"
    if two_mer == "AG":
        return "Splice Acceptor"
    return None


def main() -> None:
    args = parse_args()

    sequences_path = Path(args.sequences)
    rec = _load_record(sequences_path, str(args.variant_id))

    ref = str(rec["ref_seq"]).upper()
    alt = str(rec["alt_seq"]).upper()

    center = _center_index(rec, seq_len=len(ref))

    ref_10 = _extract_center_window(ref, center=center, window=int(args.center_window))
    alt_10 = _extract_center_window(alt, center=center, window=int(args.center_window))

    print(f"variant_id: {args.variant_id}")
    print(f"center_index: {center} (0-based)")
    print()

    print("REF:", _capitalize_center2(ref_10.lower()))
    print("ALT:", _capitalize_center2(alt_10.lower()))
    print()

    ref_center2 = _center2(ref_10)
    alt_center2 = _center2(alt_10)

    ref_type = _motif_type(ref_center2)
    if ref_type is None:
        print(f"Reference center2 = {ref_center2} (not GT/AG).")
        return

    if alt_center2 != ref_center2:
        print(f"Reference center2 = {ref_center2} ({ref_type}).")
        print(f"Alternate center2  = {alt_center2} (changed).")
        print(f"MECHANISM FOUND: Mutation disrupts a {ref_type} Site.")
        return

    print(f"Reference center2 = {ref_center2} ({ref_type}).")
    print(f"Alternate center2  = {alt_center2} (unchanged).")
    print("No disruption detected at the center2 motif.")


if __name__ == "__main__":
    main()
