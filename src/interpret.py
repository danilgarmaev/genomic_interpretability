"""Interpretability utilities for genomic variants.

Goal (mechanistic interpretability for drug discovery):
- Identify which variants likely have functional impact (the "what").
- Localize which nucleotides in a regulatory window are load-bearing (the "why").

Why this matters for drug discovery:
- Many disease-associated variants act by perturbing regulatory "switches" (enhancers,
  promoters, transcription factor binding sites). Finding the most sensitive positions
  around a variant can suggest candidate motifs, TFs, and intervention targets.

This module intentionally stays minimal and CPU-friendly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ImpactScore:
    variant_id: str
    label: Optional[Any]
    cosine_similarity: float
    l2_distance: float
    impact_score: float


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = (a.norm(p=2) * b.norm(p=2)).clamp_min(1e-12)
    return float(torch.dot(a, b) / denom)


def _l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm(p=2))


def compute_impact_scores(
    sequences_jsonl: Path,
) -> List[ImpactScore]:
    """Compute zero-shot impact scores for all variants in sequences.jsonl.

    We embed ref and alt windows with mean-pooled last hidden state and then compute:
    - cosine similarity
    - L2 distance

    impact_score is defined as (1 - cosine_similarity), i.e., larger means more shift.
    """

    from infer import embed

    records = _read_jsonl(sequences_jsonl)
    scores: List[ImpactScore] = []

    for rec in records:
        variant_id = str(rec.get("id"))
        label = rec.get("label")

        ref_emb = embed(rec["ref_seq"])  # (hidden_dim,)
        alt_emb = embed(rec["alt_seq"])  # (hidden_dim,)

        cos = _cosine_similarity(ref_emb, alt_emb)
        l2 = _l2_distance(ref_emb, alt_emb)
        impact = 1.0 - cos

        scores.append(
            ImpactScore(
                variant_id=variant_id,
                label=label,
                cosine_similarity=cos,
                l2_distance=l2,
                impact_score=impact,
            )
        )

    return scores


def save_impact_scores(scores: Iterable[ImpactScore], out_csv: Path) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "id": s.variant_id,
                "label": s.label,
                "cosine_similarity": s.cosine_similarity,
                "l2_distance": s.l2_distance,
                "impact_score": s.impact_score,
            }
            for s in scores
        ]
    )
    df.to_csv(out_csv, index=False)
    return df


def _token_index_for_nucleotide(
    *,
    seq_len: int,
    nucleotide_pos: int,
    token_count: int,
    kmer_size: int,
) -> int:
    """Map a nucleotide position to a token position in the non-special token space.

    If tokenization is k-mer based, we approximate by masking the k-mer whose last base
    is the nucleotide position.
    """

    if nucleotide_pos < 0:
        nucleotide_pos = 0
    if nucleotide_pos >= seq_len:
        nucleotide_pos = seq_len - 1

    if kmer_size <= 1:
        return min(max(nucleotide_pos, 0), token_count - 1)

    # Token i corresponds roughly to seq[i : i + kmer_size]. Mask token ending at nucleotide_pos.
    min_token = kmer_size - 1
    max_token = token_count - 1
    return min(max(nucleotide_pos, min_token), max_token)


def scanning_masking(
    record: Dict[str, Any],
    *,
    flank: int = 50,
) -> Tuple[pd.DataFrame, Path]:
    """In-silico mutagenesis by scanning masking around the variant.

    We mask (one token at a time) across +/- flank bp around variant_index, run the
    model, and measure embedding shift vs the baseline sequence embedding.

    Output:
      - DataFrame with columns: offset_bp, abs_pos, score
      - Path to saved plot
    """

    from infer import load_model

    seq = str(record["ref_seq"]).upper()
    variant_index = int(record["variant_index"])

    tokenizer, model, device = load_model()

    encoded = tokenizer(seq, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    attention_mask = attention_mask.to(device) if attention_mask is not None else None

    special_mask = tokenizer.get_special_tokens_mask(
        input_ids[0].tolist(), already_has_special_tokens=True
    )
    non_special_indices = [i for i, m in enumerate(special_mask) if m == 0]
    token_count = len(non_special_indices)

    # Heuristic: infer k-mer size if possible.
    # If token_count ~= seq_len, treat as character-level.
    if token_count <= 0:
        raise ValueError("Tokenizer produced no non-special tokens; cannot run masking.")

    seq_len = len(seq)
    if token_count == seq_len:
        kmer_size = 1
    else:
        kmer_size = max(1, seq_len - token_count + 1)

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no mask_token_id; cannot perform scanning masking.")

    def mean_pool(last_hidden: torch.Tensor) -> torch.Tensor:
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (last_hidden * mask).sum(dim=1) / denom

    with torch.no_grad():
        base_out = model(**encoded, output_hidden_states=True, return_dict=True)
        if hasattr(base_out, "last_hidden_state") and base_out.last_hidden_state is not None:
            base_hidden = base_out.last_hidden_state.to(device)
        else:
            base_hidden = base_out.hidden_states[-1].to(device)
        base_emb = mean_pool(base_hidden)[0].detach()

    offsets = list(range(-flank, flank + 1))
    rows: List[Dict[str, Any]] = []

    for off in offsets:
        pos = variant_index + off
        tok_pos_non_special = _token_index_for_nucleotide(
            seq_len=seq_len,
            nucleotide_pos=pos,
            token_count=token_count,
            kmer_size=kmer_size,
        )
        tok_idx = non_special_indices[tok_pos_non_special]

        masked_ids = input_ids.clone()
        masked_ids[0, tok_idx] = mask_id

        masked_encoded = {"input_ids": masked_ids}
        if attention_mask is not None:
            masked_encoded["attention_mask"] = attention_mask

        with torch.no_grad():
            out = model(**masked_encoded, output_hidden_states=True, return_dict=True)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                hidden = out.last_hidden_state
            else:
                hidden = out.hidden_states[-1]
            emb = mean_pool(hidden)[0].detach()

        # Embedding shift score (larger = more load-bearing position)
        score = 1.0 - float(
            torch.dot(base_emb.float(), emb.float())
            / (base_emb.norm(p=2) * emb.norm(p=2)).clamp_min(1e-12)
        )

        rows.append({"offset_bp": off, "abs_pos": pos, "score": score})

    df = pd.DataFrame(rows)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_id = str(record.get("id"))
    plot_path = out_dir / f"mutagenesis_{variant_id.replace(':', '_').replace('>', '_')}.png"

    plt.figure(figsize=(10, 3))
    plt.plot(df["offset_bp"], df["score"], linewidth=1.5)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"Scanning masking around variant: {variant_id}")
    plt.xlabel("Position relative to variant (bp)")
    plt.ylabel("Embedding shift (1 - cosine)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return df, plot_path


def analyze_variant(
    *,
    sequences_jsonl: Path,
    out_dir: Path = Path("results"),
    variant_id: Optional[str] = None,
    flank: int = 50,
) -> Dict[str, Any]:
    """Run impact scoring (all) + scanning masking (one variant).

    If variant_id is None, picks the variant with the highest impact_score.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    scores = compute_impact_scores(sequences_jsonl)
    scores_df = save_impact_scores(scores, out_dir / "impact_scores.csv")

    records = _read_jsonl(sequences_jsonl)
    rec_by_id = {str(r.get("id")): r for r in records}

    if variant_id is None:
        top = scores_df.sort_values("impact_score", ascending=False).iloc[0]
        variant_id = str(top["id"])

    if variant_id not in rec_by_id:
        raise KeyError(
            f"variant_id '{variant_id}' not found in sequences.jsonl. Available examples: {list(rec_by_id.keys())[:5]}"
        )

    mut_df, plot_path = scanning_masking(rec_by_id[variant_id], flank=flank)
    mut_df.to_csv(out_dir / "mutagenesis_scores.csv", index=False)

    return {
        "impact_scores_csv": str(out_dir / "impact_scores.csv"),
        "selected_variant_id": variant_id,
        "mutagenesis_scores_csv": str(out_dir / "mutagenesis_scores.csv"),
        "mutagenesis_plot": str(plot_path),
    }
