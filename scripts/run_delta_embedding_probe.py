"""Train/evaluate probes on multiple representation types (incl. delta embeddings).

Implements the required probe evaluation suite:
- center-token raw embedding
- center-token delta embedding (alt - ref)
- mean-pooled raw embedding (baseline)
- random-token embedding (negative control)

Uses LogisticRegression with CV over C, reports:
- AUC + bootstrap 95% CI
- Accuracy
- PR-AUC

Saves:
- results/<out-name>/embeddings.npz
- results/<out-name>/metrics.json
- results/<out-name>/probe_<rep>.joblib

Example:
  .venv/bin/python scripts/run_delta_embedding_probe.py \
    --parquet data/processed/my_processed_clinvar.parquet \
    --model InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --out results/delta_probe_500m \
    --n-total 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer


def _add_src() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=str, default="data/processed/my_processed_clinvar.parquet")
    p.add_argument(
        "--model",
        type=str,
        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    )
    p.add_argument("--out", "--outdir", dest="out", type=str, default="results/delta_probe_500m")
    p.add_argument("--n-total", "--max_n", dest="n_total", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--seq-len-bases", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    _add_src()

    from mech_eval import (
        extract_representations,
        groupwise_auc,
        load_balanced_clinvar_parquet,
        make_train_test_indices,
        pick_device,
        save_metrics_json,
        save_probe_artifacts,
        set_global_seed,
        train_probe_cv,
        variant_group,
    )

    args = parse_args()
    set_global_seed(int(args.seed))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_balanced_clinvar_parquet(Path(args.parquet), n_total=int(args.n_total), seed=int(args.seed))
    y = df["y"].to_numpy().astype(np.int64)
    groups = df.get("variant_type", pd.Series([""] * len(df))).map(variant_group).tolist()
    train_idx, test_idx = make_train_test_indices(y, seed=int(args.seed), test_size=0.2)

    device = pick_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True)
    model.to(device)

    reps = extract_representations(
        model=model,
        tokenizer=tokenizer,
        device=device,
        ref_seqs=df["ref"].tolist(),
        alt_seqs=df["alt"].tolist(),
        seq_len_bases=int(args.seq_len_bases),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    meta: Dict[str, Any] = {
        "parquet": str(args.parquet),
        "model": str(args.model),
        "n_total": int(args.n_total),
        "seq_len_bases": int(args.seq_len_bases),
        "seed": int(args.seed),
        "device": str(device),
        "variant_type_counts": df["variant_type"].value_counts().to_dict() if "variant_type" in df.columns else {},
    }

    npz_path = save_probe_artifacts(out_dir=out_dir, reps=reps, y=y, meta=meta)

    metrics_payload: Dict[str, Any] = {"meta": meta, "representations": {}}

    rep_map = {
        "center_raw": reps.center_raw,
        "center_delta": reps.center_delta,
        "mean_raw": reps.mean_raw,
        "random_raw": reps.random_raw,
    }

    for name, X in rep_map.items():
        print(f"\nTraining probe for {name}...")
        pipe, metrics, extra = train_probe_cv(X, y, seed=int(args.seed))

        # Groupwise AUC on held-out test split.
        y_score_test = pipe.predict_proba(X[test_idx])[:, 1]
        y_test = y[test_idx]
        group_auc = groupwise_auc(y_true=y_test, y_score=y_score_test, groups=[groups[i] for i in test_idx])

        metrics_payload["representations"][name] = {
            "auc": metrics.auc,
            "auc_ci95": [metrics.auc_ci_low, metrics.auc_ci_high],
            "pr_auc": metrics.pr_auc,
            "accuracy": metrics.accuracy,
            "best_C": metrics.best_C,
            "groupwise_auc_test": group_auc,
            "extra": extra,
        }
        joblib.dump(pipe, out_dir / f"probe_{name}.joblib")
        print(
            f"{name}: AUC={metrics.auc:.4f} (95% {metrics.auc_ci_low:.4f}-{metrics.auc_ci_high:.4f}) "
            f"PR-AUC={metrics.pr_auc:.4f} Acc={metrics.accuracy:.4f} best_C={metrics.best_C}"
        )

    save_metrics_json(out_dir / "metrics.json", metrics_payload)
    print("\nWrote:")
    print(f"- {npz_path}")
    print(f"- {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
