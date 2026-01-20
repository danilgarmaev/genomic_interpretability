"""Mechanistic interpretability via a lightweight probe on NTv2 (500M).

Goal
- Efficiently extract mean-pooled embeddings for a small balanced dataset.
- Train a simple probe (LogisticRegression) to predict CLNSIG.
- For one high-confidence Pathogenic example:
  - Visualize last-layer attention (center 100 tokens)
  - Compute token saliency via a linear attribution derived from the probe

This is intentionally NOT fine-tuning.

Example
  .venv/bin/python scripts/probe_mechanistic_analysis.py \
    --parquet data/processed/my_processed_clinvar.parquet \
    --out-dir results/probe_500m \
    --n-total 2000 \
    --batch-size 4

Outputs
- out_dir/embeddings.npy
- out_dir/labels.npy
- out_dir/probe_metrics.txt
- out_dir/attention_last_layer_center.png
- out_dir/saliency_center.png
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parquet",
        type=str,
        default="data/processed/my_processed_clinvar.parquet",
        help="Parquet file with columns: Sequence, CLNSIG.",
    )
    p.add_argument(
        "--sequence-col",
        type=str,
        default=None,
        help=(
            "Name of the sequence column. If omitted, tries 'Sequence', then 'alt', then 'ref'."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/probe_500m",
        help="Output directory for artifacts.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        help="HF model id to load.",
    )
    p.add_argument(
        "--n-total",
        type=int,
        default=2000,
        help="Total samples after balancing (default 2000 = 1000/1000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling and splitting.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Embedding extraction batch size. For 500M on MPS, 2-8 is typical.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Tokenizer truncation length (tokens).",
    )
    p.add_argument(
        "--center-tokens",
        type=int,
        default=100,
        help="How many tokens to show for attention/saliency (center crop).",
    )
    return p.parse_args()


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_label(clnsig: Any) -> Optional[int]:
    if clnsig is None or (isinstance(clnsig, float) and np.isnan(clnsig)):
        return None
    s = str(clnsig).strip().lower()
    # ClinVar-like values can contain multiple labels; keep it simple.
    if "pathogenic" in s and "benign" not in s:
        return 1
    if "benign" in s and "pathogenic" not in s:
        return 0
    return None


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: (B, S, D), attention_mask: (B, S)
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (last_hidden * mask).sum(dim=1) / denom
def _infer_sequence_column(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"--sequence-col={requested} not found in parquet columns: {list(df.columns)}")
        return requested
    for cand in ("Sequence", "sequence", "seq", "alt_seq", "ref_seq", "alt", "ref"):
        if cand in df.columns:
            return cand
    raise ValueError(
        "Could not infer a sequence column. Provide --sequence-col. "
        f"Available columns: {list(df.columns)}"
    )


def load_balanced(
    df: pd.DataFrame,
    *,
    n_total: int,
    seed: int,
    sequence_col: str,
) -> Tuple[List[str], np.ndarray]:
    if "CLNSIG" not in df.columns:
        raise ValueError("Parquet must contain a CLNSIG column")

    tmp = df[[sequence_col, "CLNSIG"]].copy()
    tmp = tmp.rename(columns={sequence_col: "Sequence"})
    tmp["y"] = tmp["CLNSIG"].map(normalize_label)
    tmp = tmp.dropna(subset=["y", "Sequence"])

    n_per = n_total // 2
    pos = tmp[tmp["y"] == 1].sample(n=min(n_per, (tmp["y"] == 1).sum()), random_state=seed)
    neg = tmp[tmp["y"] == 0].sample(n=min(n_per, (tmp["y"] == 0).sum()), random_state=seed)

    if len(pos) < n_per or len(neg) < n_per:
        raise ValueError(
            f"Not enough labeled samples for a balanced set of {n_total}. "
            f"Found pos={len(pos)}, neg={len(neg)} after filtering."
        )

    sampled = pd.concat([pos, neg], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sequences = sampled["Sequence"].astype(str).str.upper().tolist()
    labels = sampled["y"].astype(int).to_numpy()
    return sequences, labels


def extract_embeddings(
    *,
    sequences: List[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()

    embeddings: List[np.ndarray] = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
            batch = sequences[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            # Use the base model hidden states (not the LM head outputs).
            base_out = model.esm(**enc, return_dict=True)
            last_hidden = base_out.last_hidden_state
            pooled = mean_pool(last_hidden, enc["attention_mask"])  # (B, D)
            embeddings.append(pooled.detach().float().cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def pick_high_conf_pathogenic(
    *,
    clf,
    X: np.ndarray,
    y: np.ndarray,
    sequences: List[str],
) -> int:
    probs = clf.predict_proba(X)[:, 1]
    idxs = np.where(y == 1)[0]
    if len(idxs) == 0:
        raise ValueError("No pathogenic examples available to choose from.")
    best = idxs[np.argmax(probs[idxs])]
    return int(best)


def center_crop_indices(total: int, center_tokens: int, center_idx: int) -> Tuple[int, int]:
    half = center_tokens // 2
    start = max(0, center_idx - half)
    end = min(total, center_idx + half)
    # Adjust to requested width when possible
    if end - start < center_tokens:
        if start == 0:
            end = min(total, start + center_tokens)
        elif end == total:
            start = max(0, end - center_tokens)
    return start, end


def visualize_attention_and_saliency(
    *,
    sequence: str,
    clf_pipeline,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    center_tokens: int,
    out_dir: Path,
) -> None:
    model.eval()

    enc = tokenizer(
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.esm(**enc, output_attentions=True, return_dict=True)

    attns = out.attentions
    if attns is None:
        raise RuntimeError("No attentions returned; model may not support output_attentions.")

    # Last layer attention: (B, heads, S, S)
    last = attns[-1][0].detach().float().cpu()  # (heads, S, S)
    attn_avg = last.mean(dim=0)  # (S, S)

    input_ids = enc["input_ids"][0].detach().cpu()
    tokens_all = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    special_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    non_special = [i for i, m in enumerate(special_mask) if m == 0]
    if non_special:
        center_idx = non_special[len(non_special) // 2]
    else:
        center_idx = int(attn_avg.shape[0] // 2)

    s, e = center_crop_indices(total=int(attn_avg.shape[0]), center_tokens=center_tokens, center_idx=int(center_idx))
    crop = attn_avg[s:e, s:e].numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(crop, aspect="auto", cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"Last-layer attention (avg heads)\ncenter {center_tokens} tokens")
    plt.xlabel("Key token")
    plt.ylabel("Query token")
    plt.tight_layout()
    plt.savefig(out_dir / "attention_last_layer_center.png", dpi=160)
    plt.close()

    # Saliency: use linear attribution from the trained sklearn probe.
    # For a linear probe with mean pooling, per-token contribution to the logit is:
    #   contribution_i = (w_eff · h_i) / N
    # where w_eff accounts for StandardScaler if present.
    with torch.no_grad():
        out2 = model.esm(**enc, return_dict=True)
    hidden = out2.last_hidden_state[0].detach().float().cpu().numpy()  # (S, D)
    mask = enc.get("attention_mask")
    if mask is None:
        valid = np.ones((hidden.shape[0],), dtype=np.float32)
    else:
        valid = mask[0].detach().float().cpu().numpy()
    n_valid = max(1.0, float(valid.sum()))

    scaler: StandardScaler = clf_pipeline.named_steps["standardscaler"]
    lr: LogisticRegression = clf_pipeline.named_steps["logisticregression"]

    w = lr.coef_.reshape(-1).astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    w_eff = w / np.clip(scale, 1e-12, None)

    # contribution per token to the logit (before sigmoid)
    contrib = (hidden @ w_eff) / n_valid
    contrib = contrib * valid

    s2, e2 = center_crop_indices(total=int(contrib.shape[0]), center_tokens=center_tokens, center_idx=int(center_idx))
    contrib_crop = contrib[s2:e2]

    # Slightly taller figure to accommodate token labels + annotations.
    plt.figure(figsize=(10, 4.2))
    x = np.arange(s2, e2)
    plt.plot(x, contrib_crop, linewidth=1.5)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Token saliency (linear contribution to probe logit)", pad=12)
    plt.xlabel("Token index")
    plt.ylabel("Contribution")

    # Light labeling: show token strings on sparse ticks.
    crop_tokens = tokens_all[s2:e2]
    tick_every = 5 if (e2 - s2) <= 120 else 10
    tick_idx = list(range(0, len(crop_tokens), tick_every))
    plt.xticks(
        [x[i] for i in tick_idx],
        [crop_tokens[i].replace(" ", "") for i in tick_idx],
        rotation=90,
        fontsize=7,
    )

    # Annotate the top positive contributors (helps interpret motifs quickly).
    # Avoid annotating special tokens and keep a small K to prevent clutter.
    k = 6
    vals = contrib_crop.copy()
    # Mask special tokens (in the cropped window) so we don't annotate them.
    special_crop = np.array(special_mask[s2:e2], dtype=np.int32)
    vals = np.where(special_crop == 1, -np.inf, vals)
    top_pos = np.argsort(vals)[-k:][::-1]
    for idx in top_pos:
        if not np.isfinite(vals[idx]):
            continue
        tok = crop_tokens[int(idx)].replace(" ", "")
        plt.annotate(
            tok,
            (x[int(idx)], contrib_crop[int(idx)]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            rotation=90,
        )

    # Reserve some top margin so annotations don't collide with the title.
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(out_dir / "saliency_center.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device: {device}")

    df = pd.read_parquet(args.parquet)
    seq_col = _infer_sequence_column(df, args.sequence_col)
    sequences, labels = load_balanced(
        df,
        n_total=int(args.n_total),
        seed=int(args.seed),
        sequence_col=seq_col,
    )
    print(f"Loaded balanced set: {len(sequences)} sequences (pos={int(labels.sum())}, neg={int((labels==0).sum())})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True)
    model.to(device)

    embeddings = extract_embeddings(
        sequences=sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )

    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    np.save(out_dir / "labels.npy", labels.astype(np.int64))

    # Step 2: probe
    X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
        embeddings,
        labels,
        sequences,
        test_size=0.2,
        random_state=int(args.seed),
        stratify=labels,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            n_jobs=-1,
            class_weight=None,
            solver="lbfgs",
        ),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    print(f"Probe accuracy: {acc:.4f}")

    (out_dir / "probe_metrics.txt").write_text(f"accuracy\t{acc:.6f}\n")

    # Free memory if needed (then reload for 1 example).
    del model
    gc.collect()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # Step 3: pick one high-confidence pathogenic example (from test split for sanity)
    idx = pick_high_conf_pathogenic(clf=clf, X=X_test, y=y_test, sequences=seq_test)
    example_seq = seq_test[idx]
    p = float(clf.predict_proba(X_test[idx : idx + 1])[:, 1][0])
    print(f"Selected example: high-confidence Pathogenic with p={p:.4f}")

    # Reload model for attention + token-level states
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True)
    model.to(device)

    visualize_attention_and_saliency(
        sequence=example_seq,
        clf_pipeline=clf,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=int(args.max_length),
        center_tokens=int(args.center_tokens),
        out_dir=out_dir,
    )

    print("Wrote:")
    print(f"- {out_dir/'embeddings.npy'}")
    print(f"- {out_dir/'labels.npy'}")
    print(f"- {out_dir/'probe_metrics.txt'}")
    print(f"- {out_dir/'attention_last_layer_center.png'}")
    print(f"- {out_dir/'saliency_center.png'}")


if __name__ == "__main__":
    # Reduce tokenizer parallelism noise on macOS.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
