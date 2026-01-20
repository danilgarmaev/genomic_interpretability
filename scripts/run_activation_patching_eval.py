"""Activation patching evaluation using a trained probe.

For each layer, patch the *center token* activation from REF into ALT and measure
probe logit recovery:
  recovery = (L_patched - L_alt) / (L_ref - L_alt)

Controls:
- random_token: patch a random token (same layer)
- random_layer: patch center token but at a random layer
- full_sequence: patch ALL token activations at that layer

Outputs:
- results/<out>/patching_recovery.csv
- reports/figures/<out>_patching_recovery.png

Example:
  .venv/bin/python scripts/run_delta_embedding_probe.py --out results/delta_probe_500m
  .venv/bin/python scripts/run_activation_patching_eval.py \
    --parquet data/processed/my_processed_clinvar.parquet \
    --probe results/delta_probe_500m/probe_center_raw.joblib \
    --out results/patching_500m
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    p.add_argument("--probe", type=str, default=None, help="Path to probe_*.joblib")
    p.add_argument("--probe_dir", type=str, default=None, help="Directory containing probe_*.joblib")
    p.add_argument("--probe-rep", type=str, default="center_raw", choices=["center_raw", "mean_raw", "random_raw", "center_delta"])
    p.add_argument("--out", "--outdir", dest="out", type=str, default="results/patching_500m")
    p.add_argument("--fig", type=str, default="reports/figures/patching_recovery.png")
    p.add_argument("--n-total", "--max_n", dest="n_total", type=int, default=400, help="Eval subset size (balanced).")
    p.add_argument("--batch-size", type=int, default=1, help="Kept at 1 for patching correctness.")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--seq-len-bases", type=int, default=1024)
    return p.parse_args()


def _probe_logit(pipe, X: np.ndarray) -> float:
    # Pipeline ends with LogisticRegression; use decision_function for raw logit.
    if hasattr(pipe, "decision_function"):
        return float(pipe.decision_function(X.reshape(1, -1))[0])
    # Fallback to logit from prob
    p = float(pipe.predict_proba(X.reshape(1, -1))[:, 1][0])
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(math.log(p / (1 - p)))


def main() -> None:
    _add_src()

    from mech_eval import (
        center_token_index_for_bases,
        load_balanced_clinvar_parquet,
        mean_pool,
        pick_device,
        random_token_index,
        set_global_seed,
        variant_group,
    )

    args = parse_args()
    set_global_seed(int(args.seed))

    if args.probe is None:
        if args.probe_dir is None:
            raise SystemExit("Provide --probe or --probe_dir")
        probe_dir = Path(args.probe_dir)
        probe_path = probe_dir / f"probe_{args.probe_rep}.joblib"
        if not probe_path.exists():
            # Back-compat: some runs may use the default rep.
            fallback = probe_dir / "probe_center_raw.joblib"
            if fallback.exists():
                probe_path = fallback
            else:
                raise SystemExit(f"Probe not found: {probe_path}")
        args.probe = str(probe_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_balanced_clinvar_parquet(Path(args.parquet), n_total=int(args.n_total), seed=int(args.seed))
    pipe = joblib.load(args.probe)

    device = pick_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True)
    model.to(device)
    model.eval()

    n_layers = int(model.config.num_hidden_layers)
    rng = np.random.default_rng(int(args.seed))

    rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        ref_seq = str(row["ref"]).upper()
        alt_seq = str(row["alt"]).upper()
        y = int(row["y"])
        vtype = str(row.get("variant_type", ""))
        group = variant_group(vtype)

        enc_ref = tokenizer(ref_seq, return_tensors="pt", padding=True, truncation=True)
        enc_alt = tokenizer(alt_seq, return_tensors="pt", padding=True, truncation=True)
        enc_ref = {k: v.to(device) for k, v in enc_ref.items()}
        enc_alt = {k: v.to(device) for k, v in enc_alt.items()}

        center_base = int(args.seq_len_bases // 2)
        center_tok = center_token_index_for_bases(
            tokenizer=tokenizer,
            input_ids_1d=enc_alt["input_ids"][0],
            seq_len_bases=int(args.seq_len_bases),
            center_base_index=center_base,
        )
        rand_tok = random_token_index(tokenizer=tokenizer, input_ids_1d=enc_alt["input_ids"][0], rng=rng)

        # Baselines: ref / alt logits
        with torch.inference_mode():
            out_ref = model.esm(**enc_ref, return_dict=True)
            out_alt = model.esm(**enc_alt, return_dict=True)

        h_ref_final = out_ref.last_hidden_state[0]  # (T, D)
        h_alt_final = out_alt.last_hidden_state[0]

        def rep_from_final(h_final: torch.Tensor, which: str) -> np.ndarray:
            if which == "center_raw":
                return h_final[center_tok].detach().float().cpu().numpy()
            if which == "random_raw":
                return h_final[rand_tok].detach().float().cpu().numpy()
            if which == "mean_raw":
                pooled = mean_pool(h_final.unsqueeze(0), enc_alt.get("attention_mask"))[0]
                return pooled.detach().float().cpu().numpy()
            raise ValueError(which)

        ref_rep = rep_from_final(h_ref_final, "center_raw" if args.probe_rep == "center_delta" else args.probe_rep)
        alt_rep = rep_from_final(h_alt_final, "center_raw" if args.probe_rep == "center_delta" else args.probe_rep)
        if args.probe_rep == "center_delta":
            ref_delta = np.zeros_like(ref_rep)
            alt_delta = alt_rep - ref_rep
            ref_logit = _probe_logit(pipe, ref_delta)
            alt_logit = _probe_logit(pipe, alt_delta)
        else:
            ref_logit = _probe_logit(pipe, ref_rep)
            alt_logit = _probe_logit(pipe, alt_rep)

        denom = (ref_logit - alt_logit)
        denom = denom if abs(denom) > 1e-6 else (1e-6 if denom >= 0 else -1e-6)

        # Random layer control for this variant
        rand_layer = int(rng.integers(0, n_layers))

        for layer_idx in range(n_layers):
            layer = model.esm.encoder.layer[layer_idx]

            # Capture reference activation at this layer
            cache: Dict[str, torch.Tensor] = {}

            def cap_hook(_m, _inp, out):
                hidden = out[0] if isinstance(out, (tuple, list)) else out
                cache["h"] = hidden.detach()
                return out

            h_cap = layer.register_forward_hook(cap_hook)
            try:
                with torch.inference_mode():
                    _ = model.esm(**enc_ref, return_dict=True)
            finally:
                h_cap.remove()

            ref_layer_h = cache["h"]  # (1, T, D)

            def _run_patched(kind: str) -> float:
                def patch_hook(_m, _inp, out):
                    is_tuple = isinstance(out, (tuple, list))
                    hidden = out[0] if is_tuple else out
                    patched = hidden.clone()
                    if kind == "center":
                        patched[0, center_tok, :] = ref_layer_h[0, center_tok, :]
                    elif kind == "random_token":
                        patched[0, rand_tok, :] = ref_layer_h[0, rand_tok, :]
                    elif kind == "full_sequence":
                        patched[0, :, :] = ref_layer_h[0, : patched.shape[1], :]
                    else:
                        return out
                    if is_tuple:
                        return (patched,) + tuple(out[1:])
                    return patched

                h_patch = layer.register_forward_hook(patch_hook)
                try:
                    with torch.inference_mode():
                        out_p = model.esm(**enc_alt, return_dict=True)
                    h_final = out_p.last_hidden_state[0]
                    rep = rep_from_final(h_final, "center_raw" if args.probe_rep == "center_delta" else args.probe_rep)
                    if args.probe_rep == "center_delta":
                        rep = rep - ref_rep
                    return _probe_logit(pipe, rep)
                finally:
                    h_patch.remove()

            patched_logit = _run_patched("center")
            randtok_logit = _run_patched("random_token")
            fullseq_logit = _run_patched("full_sequence")

            # Random-layer control: only patch if this is the chosen random layer.
            randlayer_logit = float("nan")
            if layer_idx == rand_layer:
                randlayer_logit = patched_logit

            rows.append(
                {
                    "i": int(idx),
                    "y": y,
                    "variant_type": vtype,
                    "group": group,
                    "layer": int(layer_idx),
                    "ref_logit": float(ref_logit),
                    "alt_logit": float(alt_logit),
                    "patched_logit": float(patched_logit),
                    "recovery": float((patched_logit - alt_logit) / denom),
                    "randtok_recovery": float((randtok_logit - alt_logit) / denom),
                    "fullseq_recovery": float((fullseq_logit - alt_logit) / denom),
                    "rand_layer": int(rand_layer),
                    "randlayer_recovery": float((randlayer_logit - alt_logit) / denom) if not math.isnan(randlayer_logit) else float("nan"),
                }
            )

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx+1}/{len(df)} variants")

    out_df = pd.DataFrame(rows)
    csv_path = out_dir / "patching_recovery.csv"
    out_df.to_csv(csv_path, index=False)

    # Aggregate mean +/- SEM helper
    def _mean_sem(x: pd.Series) -> Tuple[float, float]:
        x = x.dropna().astype(float)
        if x.empty:
            return float("nan"), float("nan")
        return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))

    # Groupwise layer aggregates (mean +/- SEM)
    agg_rows: List[Dict[str, Any]] = []
    for group_name, gdf in out_df.groupby("group"):
        for layer, ldf in gdf.groupby("layer"):
            for col, label in [
                ("recovery", "center"),
                ("randtok_recovery", "random_token"),
                ("fullseq_recovery", "full_sequence"),
            ]:
                mean_v, sem_v = _mean_sem(ldf[col])
                agg_rows.append(
                    {
                        "group": group_name,
                        "layer": int(layer),
                        "condition": label,
                        "mean": mean_v,
                        "sem": sem_v,
                        "n": int(ldf[col].dropna().shape[0]),
                    }
                )
    group_csv = out_dir / "patching_recovery_groupwise.csv"
    pd.DataFrame(agg_rows).to_csv(group_csv, index=False)

    layers = sorted(out_df["layer"].unique().tolist())
    series = {
        "center": out_df.groupby("layer")["recovery"],
        "random_token": out_df.groupby("layer")["randtok_recovery"],
        "full_sequence": out_df.groupby("layer")["fullseq_recovery"],
    }

    plt.figure(figsize=(9, 4))
    for name, grp in series.items():
        means, sems = zip(*[_mean_sem(grp.get_group(l)) if l in grp.groups else (float('nan'), float('nan')) for l in layers])
        x = np.array(layers)
        yv = np.array(means)
        se = np.array(sems)
        plt.plot(x, yv, label=name)
        plt.fill_between(x, yv - se, yv + se, alpha=0.15)

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Logit recovery")
    plt.title("Activation patching recovery vs layer (mean ± SEM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print("Wrote:")
    print(f"- {csv_path}")
    print(f"- {group_csv}")
    print(f"- {fig_path}")


if __name__ == "__main__":
    main()
