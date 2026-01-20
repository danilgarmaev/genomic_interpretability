"""Global attention head ablation search.

This script systematically finds the most important attention heads by ablating
(=zeroing) each head's output and measuring how much the model's logits change.

Impact Score (per head):
  mean_{sequences}( MSE(original_logits, ablated_logits) )

By default it evaluates BOTH ref and alt sequences from results/sequences.jsonl.

Example:
  .venv/bin/python scripts/global_head_ablation_search.py \
    --sequences results/sequences.jsonl \
    --out-dir results \
    --max-variants 30

Then visualize the top-1 head attention map:
  (saved automatically as results/attn_top_head_layer<L>_head<H>_<variant_id>.png)

Notes:
- Designed for InstaDeepAI/nucleotide-transformer-v2-50m-multi-species, which loads
  as an ESM-like model with layers at model.esm.encoder.layer.
- Runs on CPU or GPU depending on src/infer.py load_model().
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


@dataclass(frozen=True)
class EncodedExample:
    variant_id: str
    kind: str  # "ref" or "alt"
    enc: Dict[str, torch.Tensor]
    baseline_logits_cpu: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sequences",
        type=str,
        default="results/sequences.jsonl",
        help="Path to results/sequences.jsonl (from scripts/prepare_dataset.py).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Where to write CSV + plots.",
    )
    p.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="If set, only use the first N variants from sequences.jsonl.",
    )
    p.add_argument(
        "--use",
        type=str,
        default="both",
        choices=["ref", "alt", "both"],
        help="Which sequences to evaluate.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for baseline/ablated forward passes.",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to search (default: all layers).",
    )
    p.add_argument(
        "--heads",
        type=str,
        default=None,
        help="Comma-separated head indices to search (default: all heads).",
    )
    p.add_argument(
        "--viz-variant-id",
        type=str,
        default=None,
        help="Which variant id to use for top-head attention visualization (default: first).",
    )
    p.add_argument(
        "--viz-kind",
        type=str,
        default="alt",
        choices=["ref", "alt"],
        help="Whether to visualize attention on the ref or alt sequence.",
    )
    p.add_argument(
        "--token-window",
        type=int,
        default=40,
        help="Token window radius for the attention heatmap crop.",
    )
    return p.parse_args()


def _load_records(sequences_jsonl: Path, max_variants: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in sequences_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
        if max_variants is not None and len(records) >= max_variants:
            break
    if not records:
        raise FileNotFoundError(f"No records found in {sequences_jsonl}")
    return records


def _batched(iterable: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _encode_sequences(
    *,
    records: List[Dict[str, Any]],
    which: str,
    tokenizer: Any,
    device: torch.device,
) -> List[Tuple[str, str, Dict[str, torch.Tensor]]]:
    seqs: List[Tuple[str, str, str]] = []
    for r in records:
        vid = str(r.get("id"))
        if which in ("ref", "both"):
            seqs.append((vid, "ref", str(r["ref_seq"]).upper()))
        if which in ("alt", "both"):
            seqs.append((vid, "alt", str(r["alt_seq"]).upper()))

    # Tokenize in one call for consistent padding.
    texts = [s for _vid, _kind, s in seqs]
    enc_all = tokenizer(texts, return_tensors="pt", padding=True)
    enc_all = {k: v.to(device) for k, v in enc_all.items()}

    encoded: List[Tuple[str, str, Dict[str, torch.Tensor]]] = []
    for idx, (vid, kind, _s) in enumerate(seqs):
        one = {k: v[idx : idx + 1] for k, v in enc_all.items()}
        encoded.append((vid, kind, one))
    return encoded


def _forward_logits(model: torch.nn.Module, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        out = model(**enc, return_dict=True)
    if not hasattr(out, "logits") or out.logits is None:
        raise RuntimeError("Model did not return logits; expected a MaskedLM head.")
    return out.logits


def _compute_baselines(
    *,
    model: torch.nn.Module,
    encoded: List[Tuple[str, str, Dict[str, torch.Tensor]]],
    batch_size: int,
) -> List[EncodedExample]:
    examples: List[EncodedExample] = []
    for batch in _batched(encoded, batch_size):
        # Concatenate batch tensors
        keys = batch[0][2].keys()
        batch_enc = {k: torch.cat([ex[2][k] for ex in batch], dim=0) for k in keys}
        logits = _forward_logits(model, batch_enc)  # (B, S, V)
        logits_cpu = logits.detach().float().cpu()
        for i, (vid, kind, one_enc) in enumerate(batch):
            examples.append(
                EncodedExample(
                    variant_id=vid,
                    kind=kind,
                    enc=one_enc,
                    baseline_logits_cpu=logits_cpu[i : i + 1],
                )
            )
    return examples


def _get_layer_and_attn_self(model: torch.nn.Module, layer_idx: int):
    layer = model.esm.encoder.layer[layer_idx]
    attn_self = layer.attention.self
    return layer, attn_self


def _head_ablation_hook(attn_self: torch.nn.Module, head_idx: int):
    num_heads = int(getattr(attn_self, "num_attention_heads"))
    head_dim = int(getattr(attn_self, "attention_head_size"))

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, (tuple, list))
        if is_tuple:
            context = output[0]
            rest = tuple(output[1:])
        else:
            context = output
            rest = ()

        # context: (B, S, H)
        bsz, seqlen, hidden_dim = context.shape
        if hidden_dim != num_heads * head_dim:
            return output

        x = context.reshape(bsz, seqlen, num_heads, head_dim).clone()
        x[:, :, head_idx, :] = 0.0
        patched = x.reshape(bsz, seqlen, hidden_dim)

        if is_tuple:
            return (patched,) + rest
        return patched

    return hook


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b) ** 2).item())


def _parse_int_list(spec: Optional[str]) -> Optional[List[int]]:
    if not spec:
        return None
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _crop_square(mat: torch.Tensor, center: int, radius: int) -> torch.Tensor:
    n = int(mat.shape[0])
    start = max(0, center - radius)
    end = min(n, center + radius + 1)
    return mat[start:end, start:end]


def main() -> None:
    _add_src_to_path()

    from infer import load_model

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences_jsonl = Path(args.sequences)
    records = _load_records(sequences_jsonl, args.max_variants)

    tokenizer, model, device = load_model()
    model.eval()

    # Encode all sequences and compute baseline logits once.
    encoded = _encode_sequences(
        records=records,
        which=str(args.use),
        tokenizer=tokenizer,
        device=device,
    )
    print(f"Loaded {len(records)} variants -> {len(encoded)} sequences ({args.use}).")

    t0 = time.time()
    examples = _compute_baselines(model=model, encoded=encoded, batch_size=int(args.batch_size))
    print(f"Computed baselines for {len(examples)} sequences in {time.time()-t0:.1f}s")

    n_layers = int(model.config.num_hidden_layers)
    # Determine number of heads from the first layer.
    _, attn0 = _get_layer_and_attn_self(model, 0)
    n_heads = int(getattr(attn0, "num_attention_heads"))

    layer_list = _parse_int_list(args.layers) or list(range(n_layers))
    head_list = _parse_int_list(args.heads) or list(range(n_heads))

    rows: List[Dict[str, Any]] = []

    total = len(layer_list) * len(head_list)
    done = 0
    t_search = time.time()

    for layer_idx in layer_list:
        _layer, attn_self = _get_layer_and_attn_self(model, int(layer_idx))
        for head_idx in head_list:
            hook_fn = _head_ablation_hook(attn_self, int(head_idx))
            handle = attn_self.register_forward_hook(hook_fn)
            try:
                mse_sum = 0.0
                for ex in examples:
                    ablated_logits = _forward_logits(model, ex.enc).detach().float().cpu()
                    mse_sum += _mse(ex.baseline_logits_cpu, ablated_logits)
                impact = mse_sum / max(1, len(examples))
            finally:
                handle.remove()

            rows.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "impact_score": float(impact),
                    "n_sequences": int(len(examples)),
                }
            )

            done += 1
            if done % 10 == 0 or done == total:
                rate = done / max(1e-9, (time.time() - t_search))
                eta = (total - done) / max(rate, 1e-9)
                print(f"[{done:>4}/{total}] layer {layer_idx:>2} head {head_idx:>2} impact={impact:.6g} | ETA {eta/60:.1f} min")

    # Rank heads.
    rows_sorted = sorted(rows, key=lambda r: r["impact_score"], reverse=True)
    top = rows_sorted[0]

    # Write CSV
    csv_path = out_dir / "global_head_ablation_search.csv"
    header = ["layer", "head", "impact_score", "n_sequences"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows_sorted:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    print("\nTop head:")
    print(f"- layer: {top['layer']}")
    print(f"- head:  {top['head']}")
    print(f"- impact_score (MSE logits): {top['impact_score']:.6g}")
    print(f"Wrote ranking CSV: {csv_path}")

    # Visualize attention map for the winning head.
    viz_variant_id = str(args.viz_variant_id) if args.viz_variant_id else str(records[0].get("id"))
    rec_by_id = {str(r.get("id")): r for r in records}
    if viz_variant_id not in rec_by_id:
        raise ValueError(f"viz_variant_id={viz_variant_id} not found in loaded records")
    viz_rec = rec_by_id[viz_variant_id]

    seq = str(viz_rec[f"{args.viz_kind}_seq"]).upper()
    enc = tokenizer(seq, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, output_attentions=True, return_dict=True)

    attns = out.attentions
    if attns is None:
        raise RuntimeError("Model did not return attentions; ensure output_attentions=True works.")

    layer_idx = int(top["layer"])
    head_idx = int(top["head"])
    attn = attns[layer_idx][0, head_idx].detach().float().cpu()  # (S, S)

    # Center crop around the variant token if we can map it; otherwise center crop.
    # We reuse the nucleotide->token mapping heuristic (bases_per_token) based on special-token mask.
    input_ids = enc["input_ids"][0].detach().cpu()
    special_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    non_special = [i for i, m in enumerate(special_mask) if m == 0]
    if non_special:
        seq_len = len(seq)
        tok_count = len(non_special)
        bases_per_token = max(1, int(round(seq_len / tok_count)))
        nucleotide_index = int(viz_rec.get("variant_index", seq_len // 2))
        tok_pos = min(max(int(nucleotide_index // bases_per_token), 0), tok_count - 1)
        center = int(non_special[tok_pos])
    else:
        center = int(attn.shape[0] // 2)

    crop = _crop_square(attn, center=center, radius=int(args.token_window))

    plt.figure(figsize=(6, 5))
    plt.imshow(crop.numpy(), aspect="auto", cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(
        f"Top head attention (layer {layer_idx}, head {head_idx})\n"
        f"variant {viz_variant_id} ({args.viz_kind})"
    )
    plt.xlabel("Key token")
    plt.ylabel("Query token")
    plt.tight_layout()

    png_path = out_dir / f"attn_top_head_layer{layer_idx}_head{head_idx}_{viz_variant_id}.png"
    plt.savefig(png_path, dpi=160)
    plt.close()

    print(f"Wrote attention heatmap: {png_path}")


if __name__ == "__main__":
    main()
