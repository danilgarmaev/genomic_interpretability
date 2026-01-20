"""Attention head ablation evaluation with controls.

Implements:
- Rank heads by mutation-token attention mass (mean over variants).
- Evaluate performance vs K for:
  1) top-K heads
  2) layer-matched random heads
  3) fully random heads

Metric: probe AUC on an evaluation subset.

Outputs:
- results/<out>/head_ablation_vs_k.csv
- reports/figures/<out>_head_ablation_vs_k.png

Example:
  .venv/bin/python scripts/run_head_ablation_eval.py \
    --parquet data/processed/my_processed_clinvar.parquet \
    --probe results/delta_probe_500m/probe_center_raw.joblib \
    --out results/head_ablation_500m
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
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
    p.add_argument("--probe", type=str, default=None)
    p.add_argument("--probe_dir", type=str, default=None)
    p.add_argument("--out", "--outdir", dest="out", type=str, default="results/head_ablation_500m")
    p.add_argument("--fig", type=str, default="reports/figures/head_ablation_vs_k_mean_std.png")
    p.add_argument("--n-total", "--max_n", dest="n_total", type=int, default=400, help="Eval subset size (balanced)")
    p.add_argument("--rank-samples", type=int, default=128, help="How many samples to rank heads")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--n_random_repeats",
        type=int,
        default=20,
        help="Number of random repeats for random baselines (layer-matched and fully-random).",
    )
    p.add_argument("--seq-len-bases", type=int, default=1024)
    p.add_argument("--ks", type=str, default="1,2,4,8,16")
    return p.parse_args()


def _probe_scores(pipe, X: np.ndarray) -> np.ndarray:
    # Use probability of class 1
    return pipe.predict_proba(X)[:, 1]


def _head_ablation_hook(attn_self: torch.nn.Module, heads_to_zero: Sequence[int]):
    num_heads = int(getattr(attn_self, "num_attention_heads"))
    head_dim = int(getattr(attn_self, "attention_head_size"))
    heads_set = set(int(h) for h in heads_to_zero)

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, (tuple, list))
        if is_tuple:
            context = output[0]
            rest = tuple(output[1:])
        else:
            context = output
            rest = ()
        bsz, seqlen, hidden_dim = context.shape
        if hidden_dim != num_heads * head_dim:
            return output
        x = context.reshape(bsz, seqlen, num_heads, head_dim).clone()
        for h in heads_set:
            if 0 <= h < num_heads:
                x[:, :, h, :] = 0.0
        patched = x.reshape(bsz, seqlen, hidden_dim)
        if is_tuple:
            return (patched,) + rest
        return patched

    return hook


def main() -> None:
    _add_src()

    from mech_eval import (
        center_token_index_for_bases,
        load_balanced_clinvar_parquet,
        pick_device,
        set_global_seed,
    )

    args = parse_args()
    set_global_seed(int(args.seed))

    if args.probe is None:
        if args.probe_dir is None:
            raise SystemExit("Provide --probe or --probe_dir")
        probe_path = Path(args.probe_dir) / "probe_center_raw.joblib"
        if not probe_path.exists():
            raise SystemExit(f"Probe not found: {probe_path}")
        args.probe = str(probe_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(args.probe)

    device = pick_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True)
    model.to(device)
    model.eval()

    df_eval = load_balanced_clinvar_parquet(Path(args.parquet), n_total=int(args.n_total), seed=int(args.seed))
    df_rank = df_eval.sample(n=min(int(args.rank_samples), len(df_eval)), random_state=int(args.seed)).reset_index(drop=True)

    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.esm.encoder.layer[0].attention.self.num_attention_heads)

    # Rank heads by attention mass to center token.
    scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    count = 0

    with torch.no_grad():
        for _, r in df_rank.iterrows():
            alt_seq = str(r["alt"]).upper()
            enc = tokenizer(alt_seq, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.esm(**enc, output_attentions=True, return_dict=True)
            center_tok = center_token_index_for_bases(
                tokenizer=tokenizer,
                input_ids_1d=enc["input_ids"][0],
                seq_len_bases=int(args.seq_len_bases),
                center_base_index=int(args.seq_len_bases // 2),
            )
            # attentions: tuple[L] of (B, H, T, T)
            for l in range(n_layers):
                att = out.attentions[l][0]  # (H, T, T)
                # sum over queries attending to center key token
                mass = att[:, :, center_tok].sum(dim=1).detach().float().cpu().numpy()  # (H,)
                scores[l, :] += mass
            count += 1

    scores /= max(1, count)

    flat: List[Tuple[int, int, float]] = []
    for l in range(n_layers):
        for h in range(n_heads):
            flat.append((l, h, float(scores[l, h])))
    flat.sort(key=lambda x: x[2], reverse=True)

    ks = [int(x.strip()) for x in str(args.ks).split(",") if x.strip()]
    base_seed = int(args.seed)

    # Pre-tokenize evaluation set once.
    eval_seqs = df_eval["alt"].astype(str).str.upper().tolist()
    y = df_eval["y"].to_numpy().astype(int)

    def make_layer_matched_random(topk: List[Tuple[int, int, float]], rng: np.random.Generator) -> Dict[int, List[int]]:
        lm_by_layer: Dict[int, List[int]] = {}
        for l, h, _s in topk:
            candidates = [x for x in heads_by_layer[l] if x != h]
            if not candidates:
                continue
            lm_by_layer.setdefault(l, []).append(int(rng.choice(candidates)))
        return lm_by_layer

    def make_fully_random(k: int, rng: np.random.Generator) -> Dict[int, List[int]]:
        fr_by_layer: Dict[int, List[int]] = {}
        for _ in range(k):
            l = int(rng.integers(0, n_layers))
            h = int(rng.integers(0, n_heads))
            fr_by_layer.setdefault(l, []).append(h)
        return fr_by_layer

    def compute_auc_with_hooks(head_sets_by_layer: Dict[int, List[int]]) -> float:
        hooks = []
        try:
            for l, heads in head_sets_by_layer.items():
                attn_self = model.esm.encoder.layer[l].attention.self
                hooks.append(attn_self.register_forward_hook(_head_ablation_hook(attn_self, heads)))

            # Run embeddings in batches (center token embedding)
            reps: List[np.ndarray] = []
            for i in range(0, len(eval_seqs), int(args.batch_size)):
                batch = eval_seqs[i : i + int(args.batch_size)]
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = model.esm(**enc, return_dict=True)
                h = out.last_hidden_state  # (B,T,D)

                # center token indices per element
                centers: List[int] = []
                for b in range(h.shape[0]):
                    centers.append(
                        center_token_index_for_bases(
                            tokenizer=tokenizer,
                            input_ids_1d=enc["input_ids"][b],
                            seq_len_bases=int(args.seq_len_bases),
                            center_base_index=int(args.seq_len_bases // 2),
                        )
                    )
                b_idx = torch.arange(h.shape[0], device=device)
                rep = h[b_idx, torch.tensor(centers, device=device), :].detach().float().cpu().numpy()
                reps.append(rep)

            X = np.concatenate(reps, axis=0)
            score = _probe_scores(pipe, X)
            return float(roc_auc_score(y, score))
        finally:
            for h in hooks:
                h.remove()

    # Baseline
    baseline_auc = compute_auc_with_hooks({})

    raw_rows: List[Dict[str, Any]] = []
    raw_rows.append({"K": 0, "condition": "baseline", "auc": baseline_auc, "repeat_id": 0})

    # Pre-build pools for random head selection
    heads_by_layer = {l: list(range(n_heads)) for l in range(n_layers)}

    for K in ks:
        topk = flat[:K]

        # top-K (deterministic)
        top_by_layer: Dict[int, List[int]] = {}
        for l, h, _s in topk:
            top_by_layer.setdefault(l, []).append(h)
        auc_top = compute_auc_with_hooks(top_by_layer)
        raw_rows.append({"K": K, "condition": "topK", "auc": auc_top, "repeat_id": 0})

        # Random baselines with repeats
        n_rep = int(args.n_random_repeats)
        for repeat_id in range(n_rep):
            # Stable per-repeat seed
            rep_seed = base_seed + 10_000 * (K + 1) + repeat_id
            rng = np.random.default_rng(rep_seed)
            lm_by_layer = make_layer_matched_random(topk, rng)
            fr_by_layer = make_fully_random(K, rng)

            auc_lm = compute_auc_with_hooks(lm_by_layer)
            auc_fr = compute_auc_with_hooks(fr_by_layer)
            raw_rows.append({"K": K, "condition": "layer_matched_random", "auc": auc_lm, "repeat_id": repeat_id})
            raw_rows.append({"K": K, "condition": "fully_random", "auc": auc_fr, "repeat_id": repeat_id})

        print(
            f"K={K}: baseline={baseline_auc:.4f} topK={auc_top:.4f} "
            f"(random repeats={n_rep}, seed={base_seed})"
        )

    raw_df = pd.DataFrame(raw_rows)
    raw_csv_path = out_dir / "head_ablation_raw.csv"
    raw_df.to_csv(raw_csv_path, index=False)

    summary_df = (
        raw_df.groupby(["K", "condition"], as_index=False)["auc"]
        .agg(auc_mean="mean", auc_std=lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)
        .sort_values(["condition", "K"], ascending=[True, True])
        .reset_index(drop=True)
    )
    summary_csv_path = out_dir / "head_ablation_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # Back-compat: keep the previous filename as the summary.
    legacy_csv_path = out_dir / "head_ablation_vs_k.csv"
    summary_df.rename(columns={"auc_mean": "auc"}).drop(columns=["auc_std"]).to_csv(legacy_csv_path, index=False)

    # Plot
    plt.figure(figsize=(7.8, 4.2))
    for cond in ["topK", "layer_matched_random", "fully_random"]:
        sub = summary_df[summary_df["condition"] == cond].sort_values("K")
        plt.plot(sub["K"], sub["auc_mean"], marker="o", label=cond)
        if cond in {"layer_matched_random", "fully_random"}:
            x = sub["K"].to_numpy()
            m = sub["auc_mean"].to_numpy()
            s = sub["auc_std"].to_numpy()
            plt.fill_between(x, m - s, m + s, alpha=0.15)

    plt.axhline(baseline_auc, color="black", linestyle="--", linewidth=1, label="baseline")
    plt.xlabel("K heads ablated")
    plt.ylabel("Probe AUC")
    plt.title("Attention head ablation: mean ± std vs K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print("Wrote:")
    print(f"- {raw_csv_path}")
    print(f"- {summary_csv_path}")
    print(f"- {legacy_csv_path}")
    print(f"- {fig_path}")


if __name__ == "__main__":
    main()
