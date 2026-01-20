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
    bases_per_token: int,
) -> int:
    """Map a nucleotide position to a token position in the non-special token space.

    For Nucleotide Transformer style tokenization, each non-special token typically
    corresponds to a fixed-size chunk (often ~6 bases). This mapping converts a base
    index into a token index.
    """

    if nucleotide_pos < 0:
        nucleotide_pos = 0
    if nucleotide_pos >= seq_len:
        nucleotide_pos = seq_len - 1

    if bases_per_token <= 1:
        return min(max(nucleotide_pos, 0), token_count - 1)

    tok = nucleotide_pos // bases_per_token
    return min(max(int(tok), 0), token_count - 1)


def _variant_token_index(
    *,
    tokenizer: Any,
    encoded: Dict[str, torch.Tensor],
    nucleotide_index: int,
    seq_len: int,
) -> int:
    input_ids = encoded["input_ids"][0]
    special_mask = tokenizer.get_special_tokens_mask(
        input_ids.tolist(), already_has_special_tokens=True
    )
    non_special_indices = [i for i, m in enumerate(special_mask) if m == 0]
    token_count = len(non_special_indices)
    if token_count <= 0:
        raise ValueError("Tokenizer produced no non-special tokens.")

    bases_per_token = max(1, int(round(seq_len / token_count)))
    tok_pos_non_special = _token_index_for_nucleotide(
        seq_len=seq_len,
        nucleotide_pos=nucleotide_index,
        token_count=token_count,
        bases_per_token=bases_per_token,
    )
    return non_special_indices[tok_pos_non_special]


def _mean_pool_hidden(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def _embed_encoded(model: torch.nn.Module, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True, return_dict=True)
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        hidden = out.last_hidden_state
    else:
        hidden = out.hidden_states[-1]
    attn_mask = encoded.get("attention_mask")
    pooled = _mean_pool_hidden(hidden, attn_mask)[0]
    return pooled


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

    # Heuristic: infer an effective bases-per-token.
    # For nucleotide-transformer-v2, this is typically ~6.
    if token_count <= 0:
        raise ValueError("Tokenizer produced no non-special tokens; cannot run masking.")

    seq_len = len(seq)
    bases_per_token = max(1, int(round(seq_len / token_count)))

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
            bases_per_token=bases_per_token,
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


def attention_visualization(
    record: Dict[str, Any],
    *,
    out_dir: Path = Path("results"),
    token_window: int = 20,
    layers: Optional[List[int]] = None,
) -> Dict[str, str]:
    """Generate attention heatmaps for ref vs alt around the variant.

    This is a mechanistic probe: attention can highlight which tokens the model
    condition on when processing a locus. Differences (alt - ref) can suggest
    altered context usage near putative regulatory motifs (e.g., TF binding sites).

    We save:
    - ref attention (mean over heads)
    - alt attention (mean over heads)
    - diff attention (alt - ref)

    Notes:
    - Attention is token-level; with Nucleotide Transformer v2 the tokenization is
      typically ~6bp per token, so fine-grained 1bp resolution is not expected.
    """

    from infer import load_model

    out_dir.mkdir(parents=True, exist_ok=True)

    variant_id = str(record.get("id"))
    ref_seq = str(record["ref_seq"]).upper()
    alt_seq = str(record["alt_seq"]).upper()
    variant_index = int(record["variant_index"])  # nucleotide index in the 1000bp window

    tokenizer, model, device = load_model()

    def encode(seq: str) -> Dict[str, torch.Tensor]:
        enc = tokenizer(seq, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    ref_enc = encode(ref_seq)
    alt_enc = encode(alt_seq)

    # Map nucleotide position -> token index.
    input_ids = ref_enc["input_ids"][0]
    variant_tok_idx = _variant_token_index(
        tokenizer=tokenizer,
        encoded=ref_enc,
        nucleotide_index=variant_index,
        seq_len=len(ref_seq),
    )

    # Select layers to plot.
    with torch.no_grad():
        ref_out = model(**ref_enc, output_attentions=True, return_dict=True)
        alt_out = model(**alt_enc, output_attentions=True, return_dict=True)

    ref_atts = list(ref_out.attentions or [])
    alt_atts = list(alt_out.attentions or [])
    if not ref_atts or not alt_atts:
        raise ValueError(
            "Model did not return attentions. Try a different model or ensure trust_remote_code is enabled."
        )

    n_layers = min(len(ref_atts), len(alt_atts))
    if layers is None:
        layers = [n_layers - 1]

    # Define token slice around the variant token.
    start = max(0, variant_tok_idx - token_window)
    end = min(int(input_ids.shape[0]), variant_tok_idx + token_window + 1)

    def mean_head(att: torch.Tensor) -> np.ndarray:
        # att: (batch=1, heads, seq, seq)
        a = att[0].float().mean(dim=0)  # (seq, seq)
        return a.detach().cpu().numpy()

    outputs: Dict[str, str] = {}

    for layer in layers:
        if layer < 0 or layer >= n_layers:
            continue

        ref_mat = mean_head(ref_atts[layer])[start:end, start:end]
        alt_mat = mean_head(alt_atts[layer])[start:end, start:end]
        diff_mat = alt_mat - ref_mat

        def save_heatmap(mat: np.ndarray, title: str, fname: str, *, diverging: bool = False) -> str:
            plt.figure(figsize=(6, 5))
            if diverging:
                vmax = float(np.max(np.abs(mat))) if mat.size else 1.0
                plt.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            else:
                plt.imshow(mat, aspect="auto", cmap="viridis")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(title)
            plt.xlabel("Key token index (local window)")
            plt.ylabel("Query token index (local window)")
            plt.tight_layout()
            out_path = out_dir / fname
            plt.savefig(out_path, dpi=150)
            plt.close()
            return str(out_path)

        outputs[f"attn_ref_layer{layer}"] = save_heatmap(
            ref_mat,
            title=f"Attention (ref) {variant_id} layer {layer}",
            fname=f"attn_ref_{variant_id}_layer{layer}.png",
            diverging=False,
        )
        outputs[f"attn_alt_layer{layer}"] = save_heatmap(
            alt_mat,
            title=f"Attention (alt) {variant_id} layer {layer}",
            fname=f"attn_alt_{variant_id}_layer{layer}.png",
            diverging=False,
        )
        outputs[f"attn_diff_layer{layer}"] = save_heatmap(
            diff_mat,
            title=f"Attention diff (alt-ref) {variant_id} layer {layer}",
            fname=f"attn_diff_{variant_id}_layer{layer}.png",
            diverging=True,
        )

    return outputs


def activation_patching(
    record: Dict[str, Any],
    *,
    out_dir: Path = Path("results"),
    token_window: int = 5,
    layers: Optional[List[int]] = None,
) -> Dict[str, str]:
    """Activation patching on the residual stream (layer outputs).

    Procedure:
    - Compute ref and alt baseline embeddings.
    - For each selected layer L and each token in a local window around the variant,
      run the model on ALT while patching the layer output at that token with the REF
      activation.
    - Measure how much this patch reduces the ref-vs-alt embedding shift.

    This is a causal test: if patching a component makes ALT behave more like REF,
    that component is (part of) the mechanism.
    """

    from infer import load_model

    out_dir.mkdir(parents=True, exist_ok=True)

    variant_id = str(record.get("id"))
    ref_seq = str(record["ref_seq"]).upper()
    alt_seq = str(record["alt_seq"]).upper()
    variant_index = int(record["variant_index"])

    tokenizer, model, device = load_model()

    ref_enc = {k: v.to(device) for k, v in tokenizer(ref_seq, return_tensors="pt").items()}
    alt_enc = {k: v.to(device) for k, v in tokenizer(alt_seq, return_tensors="pt").items()}

    ref_emb = _embed_encoded(model, ref_enc)
    alt_emb = _embed_encoded(model, alt_enc)
    base_impact = 1.0 - float(
        torch.dot(ref_emb.float(), alt_emb.float())
        / (ref_emb.norm(p=2) * alt_emb.norm(p=2)).clamp_min(1e-12)
    )

    encoder_layers = list(model.esm.encoder.layer)
    n_layers = len(encoder_layers)
    if layers is None:
        layers = [n_layers - 1]

    variant_tok_idx = _variant_token_index(
        tokenizer=tokenizer,
        encoded=ref_enc,
        nucleotide_index=variant_index,
        seq_len=len(ref_seq),
    )
    start = max(0, variant_tok_idx - token_window)
    end = min(int(ref_enc["input_ids"].shape[1]), variant_tok_idx + token_window + 1)
    token_indices = list(range(start, end))

    rows: List[Dict[str, Any]] = []

    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= n_layers:
            continue

        layer = encoder_layers[layer_idx]
        ref_cache: Dict[str, torch.Tensor] = {}

        def capture_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            ref_cache["hidden"] = hidden.detach()
            return output

        h1 = layer.register_forward_hook(capture_hook)
        try:
            _ = model(**ref_enc, return_dict=True)
        finally:
            h1.remove()

        if "hidden" not in ref_cache:
            raise RuntimeError("Failed to capture reference activation for activation patching.")
        ref_hidden = ref_cache["hidden"]  # (1, seq, hidden_dim)

        for tok in token_indices:
            def patch_hook(_module, _inputs, output):
                if isinstance(output, (tuple, list)):
                    hidden = output[0]
                    patched = hidden.clone()
                    patched[0, tok, :] = ref_hidden[0, tok, :]
                    return (patched,) + tuple(output[1:])
                patched = output.clone()
                patched[0, tok, :] = ref_hidden[0, tok, :]
                return patched

            h2 = layer.register_forward_hook(patch_hook)
            try:
                patched_emb = _embed_encoded(model, alt_enc)
            finally:
                h2.remove()

            patched_impact = 1.0 - float(
                torch.dot(ref_emb.float(), patched_emb.float())
                / (ref_emb.norm(p=2) * patched_emb.norm(p=2)).clamp_min(1e-12)
            )

            rows.append(
                {
                    "variant_id": variant_id,
                    "layer": int(layer_idx),
                    "token_index": int(tok),
                    "token_offset": int(tok - variant_tok_idx),
                    "base_impact": float(base_impact),
                    "patched_impact": float(patched_impact),
                    "effect": float(base_impact - patched_impact),
                }
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"activation_patching_{variant_id}.csv"
    df.to_csv(csv_path, index=False)

    # Simple heatmap: layers x token_offset
    if not df.empty:
        pivot = df.pivot_table(index="layer", columns="token_offset", values="effect", aggfunc="mean")
        plt.figure(figsize=(8, 3))
        plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
        plt.xticks(range(len(pivot.columns)), [str(i) for i in pivot.columns], rotation=0)
        plt.title(f"Activation patching effect (base - patched): {variant_id}")
        plt.xlabel("Token offset from variant")
        plt.ylabel("Layer")
        plt.tight_layout()
        png_path = out_dir / f"activation_patching_{variant_id}.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
    else:
        png_path = out_dir / f"activation_patching_{variant_id}.png"

    return {"activation_patching_csv": str(csv_path), "activation_patching_plot": str(png_path)}


def attention_head_ablation(
    record: Dict[str, Any],
    *,
    out_dir: Path = Path("results"),
    layer: Optional[int] = None,
) -> Dict[str, str]:
    """Ablate attention heads (zero their contribution) and measure impact.

    We ablate heads by reshaping the self-attention context output into
    (num_heads, head_dim) and zeroing a head slice.

    Metric: ref-vs-alt embedding shift (1 - cosine). Larger means alt is more different.
    """

    from infer import load_model

    out_dir.mkdir(parents=True, exist_ok=True)

    variant_id = str(record.get("id"))
    ref_seq = str(record["ref_seq"]).upper()
    alt_seq = str(record["alt_seq"]).upper()

    tokenizer, model, device = load_model()
    ref_enc = {k: v.to(device) for k, v in tokenizer(ref_seq, return_tensors="pt").items()}
    alt_enc = {k: v.to(device) for k, v in tokenizer(alt_seq, return_tensors="pt").items()}

    ref_emb = _embed_encoded(model, ref_enc)
    alt_emb = _embed_encoded(model, alt_enc)
    base_impact = 1.0 - float(
        torch.dot(ref_emb.float(), alt_emb.float())
        / (ref_emb.norm(p=2) * alt_emb.norm(p=2)).clamp_min(1e-12)
    )

    encoder_layers = list(model.esm.encoder.layer)
    n_layers = len(encoder_layers)
    if layer is None:
        layer = n_layers - 1
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"layer must be in [0, {n_layers-1}]")

    attn_self = encoder_layers[layer].attention.self
    num_heads = int(getattr(attn_self, "num_attention_heads"))
    head_dim = int(getattr(attn_self, "attention_head_size"))

    rows: List[Dict[str, Any]] = []

    for h in range(num_heads):
        def ablate_hook(_module, _inputs, output):
            # output is typically (context, attn_probs) or (context,)
            is_tuple = isinstance(output, (tuple, list))
            if is_tuple:
                context = output[0]
                rest = tuple(output[1:])
            else:
                context = output
                rest = ()

            # context: (batch, seq, hidden_dim)
            bsz, seqlen, hidden_dim = context.shape
            if hidden_dim != num_heads * head_dim:
                return output
            x = context.view(bsz, seqlen, num_heads, head_dim)
            x[:, :, h, :] = 0.0
            patched = x.view(bsz, seqlen, hidden_dim)

            # Preserve the original output type. Returning a bare tensor when the
            # module normally returns a tuple causes downstream code to index into
            # the tensor (dropping the batch dim) and crash.
            if is_tuple:
                return (patched,) + rest
            return patched

        hook = attn_self.register_forward_hook(ablate_hook)
        try:
            ablated_emb = _embed_encoded(model, alt_enc)
        finally:
            hook.remove()

        ablated_impact = 1.0 - float(
            torch.dot(ref_emb.float(), ablated_emb.float())
            / (ref_emb.norm(p=2) * ablated_emb.norm(p=2)).clamp_min(1e-12)
        )

        rows.append(
            {
                "variant_id": variant_id,
                "layer": int(layer),
                "head": int(h),
                "base_impact": float(base_impact),
                "ablated_impact": float(ablated_impact),
                "delta": float(ablated_impact - base_impact),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"head_ablation_{variant_id}_layer{layer}.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 3))
    plt.bar(df["head"].astype(int), df["delta"].astype(float))
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Attention head ablation delta impact (layer {layer}): {variant_id}")
    plt.xlabel("Head")
    plt.ylabel("Δ impact (ablated - base)")
    plt.tight_layout()
    png_path = out_dir / f"head_ablation_{variant_id}_layer{layer}.png"
    plt.savefig(png_path, dpi=150)
    plt.close()

    return {"head_ablation_csv": str(csv_path), "head_ablation_plot": str(png_path)}


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

    outputs: Dict[str, Any] = {
        "impact_scores_csv": str(out_dir / "impact_scores.csv"),
        "selected_variant_id": variant_id,
        "mutagenesis_scores_csv": str(out_dir / "mutagenesis_scores.csv"),
        "mutagenesis_plot": str(plot_path),

    }

    return outputs
