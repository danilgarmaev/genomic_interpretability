"""Mechanistic evaluation utilities (probe + causal interventions).

This module focuses on *evaluation* (not fine-tuning):
- Efficient embedding extraction from Nucleotide Transformer v2 (ESM-like).
- Multiple representation choices (raw center token, mean pool, delta embeddings, controls).
- Probe training with cross-validated C and robust metrics + bootstrap CIs.

Designed for local runs on macOS (MPS) with reasonable defaults.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_clnsig_to_y(clnsig: Any) -> Optional[int]:
    if clnsig is None or (isinstance(clnsig, float) and np.isnan(clnsig)):
        return None
    s = str(clnsig).strip().lower()
    if "pathogenic" in s and "benign" not in s:
        return 1
    if "benign" in s and "pathogenic" not in s:
        return 0
    return None


def load_balanced_clinvar_parquet(
    parquet_path: Path,
    *,
    n_total: int = 2000,
    seed: int = 13,
    require_cols: Sequence[str] = ("ref", "alt", "CLNSIG"),
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    for c in require_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'. Columns: {list(df.columns)}")

    tmp = df.copy()
    tmp["y"] = tmp["CLNSIG"].map(normalize_clnsig_to_y)
    tmp = tmp.dropna(subset=["y", "ref", "alt"]).reset_index(drop=True)

    n_per = n_total // 2
    pos = tmp[tmp["y"] == 1]
    neg = tmp[tmp["y"] == 0]

    if len(pos) < n_per or len(neg) < n_per:
        raise ValueError(f"Not enough samples for {n_total} balanced: pos={len(pos)}, neg={len(neg)}")

    pos_s = pos.sample(n=n_per, random_state=seed)
    neg_s = neg.sample(n=n_per, random_state=seed)
    out = pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Ensure strings
    out["ref"] = out["ref"].astype(str).str.upper()
    out["alt"] = out["alt"].astype(str).str.upper()
    out["y"] = out["y"].astype(int)
    if "variant_type" in out.columns:
        out["variant_type"] = out["variant_type"].astype(str)
    return out


def variant_group(variant_type: Any) -> str:
    """Coarse stratification group derived from `variant_type`.

    This is intentionally simple and robust to heterogeneous annotations.
    """

    s = str(variant_type or "").lower()
    if "splice" in s:
        return "splice"
    return "non_splice"


def make_train_test_indices(y: np.ndarray, *, seed: int = 13, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def mean_pool(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def non_special_token_indices(tokenizer: Any, input_ids_1d: torch.Tensor) -> List[int]:
    ids = input_ids_1d.detach().cpu().tolist()
    special = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
    return [i for i, m in enumerate(special) if m == 0]


def center_token_index_for_bases(
    *,
    tokenizer: Any,
    input_ids_1d: torch.Tensor,
    seq_len_bases: int,
    center_base_index: int,
) -> int:
    non_special = non_special_token_indices(tokenizer, input_ids_1d)
    if not non_special:
        return int(input_ids_1d.shape[0] // 2)
    token_count = len(non_special)
    bases_per_token = max(1, int(round(seq_len_bases / token_count)))
    tok_pos = min(max(int(center_base_index // bases_per_token), 0), token_count - 1)
    return int(non_special[tok_pos])


def random_token_index(
    *,
    tokenizer: Any,
    input_ids_1d: torch.Tensor,
    rng: np.random.Generator,
) -> int:
    non_special = non_special_token_indices(tokenizer, input_ids_1d)
    if not non_special:
        return int(input_ids_1d.shape[0] // 2)
    return int(rng.choice(non_special))


@dataclass(frozen=True)
class Representations:
    center_raw: np.ndarray
    center_delta: np.ndarray
    mean_raw: np.ndarray
    random_raw: np.ndarray


def extract_representations(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    ref_seqs: Sequence[str],
    alt_seqs: Sequence[str],
    seq_len_bases: int = 1024,
    batch_size: int = 4,
    seed: int = 13,
) -> Representations:
    """Extract representations for a list of (ref, alt) pairs.

    Representations:
    - center_raw: center token embedding from ALT
    - center_delta: center token embedding ALT - center token embedding REF
    - mean_raw: mean pooled embedding from ALT
    - random_raw: random token embedding from ALT (negative control)

    Notes:
    - Center is defined in *base space* as seq_len_bases//2 and mapped to token space.
    - Uses the base model (model.esm) to get last_hidden_state.
    """

    model.eval()
    rng = np.random.default_rng(seed)

    n = len(ref_seqs)
    if len(alt_seqs) != n:
        raise ValueError("ref_seqs and alt_seqs must have same length")

    center_base = int(seq_len_bases // 2)

    center_alt_list: List[np.ndarray] = []
    center_ref_list: List[np.ndarray] = []
    mean_alt_list: List[np.ndarray] = []
    random_alt_list: List[np.ndarray] = []

    with torch.inference_mode():
        for i in range(0, n, batch_size):
            ref_batch = list(ref_seqs[i : i + batch_size])
            alt_batch = list(alt_seqs[i : i + batch_size])

            # Tokenize separately because sequences differ.
            enc_ref = tokenizer(ref_batch, return_tensors="pt", padding=True, truncation=True)
            enc_alt = tokenizer(alt_batch, return_tensors="pt", padding=True, truncation=True)
            enc_ref = {k: v.to(device) for k, v in enc_ref.items()}
            enc_alt = {k: v.to(device) for k, v in enc_alt.items()}

            out_ref = model.esm(**enc_ref, return_dict=True)
            out_alt = model.esm(**enc_alt, return_dict=True)
            h_ref = out_ref.last_hidden_state  # (B, T, D)
            h_alt = out_alt.last_hidden_state

            # Mean pooled
            mean_alt = mean_pool(h_alt, enc_alt.get("attention_mask"))

            # Center + random token indices per example
            center_idxs: List[int] = []
            rand_idxs: List[int] = []
            for b in range(h_alt.shape[0]):
                center_idxs.append(
                    center_token_index_for_bases(
                        tokenizer=tokenizer,
                        input_ids_1d=enc_alt["input_ids"][b],
                        seq_len_bases=seq_len_bases,
                        center_base_index=center_base,
                    )
                )
                rand_idxs.append(
                    random_token_index(tokenizer=tokenizer, input_ids_1d=enc_alt["input_ids"][b], rng=rng)
                )

            b_idx = torch.arange(h_alt.shape[0], device=device)
            center_alt = h_alt[b_idx, torch.tensor(center_idxs, device=device), :]
            center_ref = h_ref[b_idx, torch.tensor(center_idxs, device=device), :]
            rand_alt = h_alt[b_idx, torch.tensor(rand_idxs, device=device), :]

            center_alt_list.append(center_alt.detach().float().cpu().numpy())
            center_ref_list.append(center_ref.detach().float().cpu().numpy())
            mean_alt_list.append(mean_alt.detach().float().cpu().numpy())
            random_alt_list.append(rand_alt.detach().float().cpu().numpy())

    center_alt_np = np.concatenate(center_alt_list, axis=0)
    center_ref_np = np.concatenate(center_ref_list, axis=0)
    mean_alt_np = np.concatenate(mean_alt_list, axis=0)
    random_alt_np = np.concatenate(random_alt_list, axis=0)

    return Representations(
        center_raw=center_alt_np,
        center_delta=center_alt_np - center_ref_np,
        mean_raw=mean_alt_np,
        random_raw=random_alt_np,
    )


@dataclass(frozen=True)
class ProbeMetrics:
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    pr_auc: float
    accuracy: float
    best_C: float


def _bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 13,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        # Need both classes present
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(float(roc_auc_score(yt, ys)))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def train_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 13,
    cv_folds: int = 5,
    C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> Tuple[Pipeline, ProbeMetrics, Dict[str, Any]]:
    """Train a logistic regression probe with CV over C.

    Returns:
        (fitted_pipeline, metrics_on_test, extra_info)

    Metrics are evaluated on a held-out test split.
    """

    train_idx, test_idx = make_train_test_indices(y, seed=seed, test_size=0.2)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        pipe,
        param_grid={"lr__C": list(C_grid)},
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_pipe: Pipeline = grid.best_estimator_
    best_C = float(grid.best_params_["lr__C"])

    y_score = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_score))
    pr_auc = float(average_precision_score(y_test, y_score))
    acc = float(accuracy_score(y_test, y_pred))
    lo, hi = _bootstrap_auc_ci(y_test.astype(int), y_score.astype(float), seed=seed)

    metrics = ProbeMetrics(
        auc=auc,
        auc_ci_low=lo,
        auc_ci_high=hi,
        pr_auc=pr_auc,
        accuracy=acc,
        best_C=best_C,
    )

    extra = {
        "best_params": grid.best_params_,
        "cv_results": {
            "mean_test_score": grid.cv_results_["mean_test_score"].tolist(),
            "params": [dict(p) for p in grid.cv_results_["params"]],
        },
        "test_size": 0.2,
        "seed": seed,
        "split": {
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
        },
    }

    return best_pipe, metrics, extra


def groupwise_auc(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: Sequence[str],
    min_n: int = 25,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    g = np.asarray(list(groups))
    for name in sorted(set(g.tolist())):
        mask = g == name
        yt = y_true[mask]
        ys = y_score[mask]
        if mask.sum() < min_n or len(np.unique(yt)) < 2:
            out[name] = {"n": int(mask.sum()), "auc": None}
            continue
        out[name] = {"n": int(mask.sum()), "auc": float(roc_auc_score(yt, ys))}
    return out


def save_probe_artifacts(
    *,
    out_dir: Path,
    reps: Representations,
    y: np.ndarray,
    meta: Dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "embeddings.npz"
    np.savez_compressed(
        npz_path,
        center_raw=reps.center_raw.astype(np.float32),
        center_delta=reps.center_delta.astype(np.float32),
        mean_raw=reps.mean_raw.astype(np.float32),
        random_raw=reps.random_raw.astype(np.float32),
        y=y.astype(np.int64),
        meta_json=json.dumps(meta),
    )
    return npz_path


def save_metrics_json(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
