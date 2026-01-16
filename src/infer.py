"""Minimal single-sequence inference.

PHASE 2: minimal model loading + inference only.
- Uses a transformer-based genomic model from Hugging Face.
- No datasets, no batching, no interpretability.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"

_MODEL: Optional[torch.nn.Module] = None
_TOKENIZER = None
_DEVICE: Optional[torch.device] = None


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    trust_remote_code: bool = True,
) -> Tuple[object, torch.nn.Module, torch.device]:
    """Load tokenizer + model once and reuse.

    Returns:
        (tokenizer, model, device)
    """

    global _MODEL, _TOKENIZER, _DEVICE

    if _MODEL is not None and _TOKENIZER is not None and _DEVICE is not None:
        return _TOKENIZER, _MODEL, _DEVICE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=False,
            dtype=torch.float32,
        )
    except ImportError as exc:
        raise ImportError(
            "Missing an optional dependency required by this Hugging Face model. "
            "Install the missing package (e.g., `pip install einops`) or choose a different model_name."
        ) from exc
    model.to(device)
    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model
    _DEVICE = device
    return tokenizer, model, device


def predict(sequence: str) -> torch.Tensor:
    """Run inference on a single genomic sequence.

    Args:
        sequence: DNA sequence string (e.g., "ACGT...").

    Returns:
        Logits tensor of shape (seq_len, vocab_size) for masked-LM style models.
    """

    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("sequence must be a non-empty string")

    sequence = "".join(sequence.split()).upper()

    tokenizer, model, device = load_model()
    encoded = tokenizer(sequence, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    return outputs.logits[0].detach().cpu()
