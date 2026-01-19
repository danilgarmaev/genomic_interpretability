"""Run inference on prepared variant sequences.

Reads results/sequences.jsonl and runs inference on the first 3 variants
(ref vs alt) using the default model in src/infer.py.

PHASE 3: inference over prepared ref/alt windows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    _add_src_to_path()

    from infer import predict

    repo_root = Path(__file__).resolve().parents[1]
    seq_path = repo_root / "results" / "sequences.jsonl"
    if not seq_path.exists():
        raise SystemExit(
            f"Missing {seq_path}. Run scripts/prepare_dataset.py first to generate it."
        )

    records = _read_jsonl(seq_path)
    if not records:
        raise SystemExit(f"No records found in {seq_path}.")

    print("id\tlabel\tref_mean_logit\talt_mean_logit\tdelta")
    for rec in records[:3]:
        ref_logits = predict(rec["ref_seq"])
        alt_logits = predict(rec["alt_seq"])

        ref_score = float(ref_logits.float().mean().item())
        alt_score = float(alt_logits.float().mean().item())
        delta = alt_score - ref_score

        print(
            f"{rec.get('id')}\t{rec.get('label')}\t"
            f"{ref_score:.6f}\t{alt_score:.6f}\t{delta:.6f}"
        )


if __name__ == "__main__":
    main()
