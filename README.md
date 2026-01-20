# Genomic Interpretability

Reproducible analysis code for mechanistic interpretability experiments on Nucleotide Transformer v2 using ClinVar-derived variant sequences.

Author: Danil Garmaev

This repository contains the scripts used to train linear probes, run activation patching, and perform controlled attention-head ablations reported in the accompanying write-up.

## Environment

Tested with Python ≥ 3.10.

Main dependencies:
- PyTorch
- Hugging Face `transformers`
- `pandas`, `pyarrow`
- `scikit-learn`
- `matplotlib`

macOS MPS acceleration is used automatically when available.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Most scripts expect a processed parquet file at:

`data/processed/my_processed_clinvar.parquet`

Required columns:
- `ref`: reference sequence (1024 bp)
- `alt`: alternate sequence (1024 bp)
- `variant_type`
- `CLNSIG` (e.g. `Pathogenic`, `Benign`)

A small processed parquet is included for convenience.
Large intermediate outputs and figures are not tracked in git.

## Regenerating the parquet (optional)

If you prefer to build the dataset locally from ClinVar:

1) Extract variant-centered sequences:

```bash
python scripts/prepare_dataset.py --help
```

2) Convert to parquet:

```bash
python scripts/build_probe_parquet_from_sequences.py \
  --sequences results/sequences.jsonl \
  --out data/processed/my_processed_clinvar.parquet
```

Note: in this repo, an example sequences file may already exist at:
- `results/legacy/sequences.jsonl`

If so, you can run:

```bash
python scripts/build_probe_parquet_from_sequences.py \
  --sequences results/legacy/sequences.jsonl \
  --out data/processed/my_processed_clinvar.parquet
```

## Reproducing the Experiments

All commands assume the virtual environment is active.

### 1) Train and evaluate linear probes

```bash
python scripts/run_delta_embedding_probe.py \
  --parquet data/processed/my_processed_clinvar.parquet \
  --model InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
  --out results/delta_probe_500m \
  --n-total 2000
```

Outputs:
- `results/delta_probe_500m/metrics.json`
- trained probes: `results/delta_probe_500m/probe_*.joblib`
- saved embeddings: `results/delta_probe_500m/embeddings.npz`

### 2) Activation patching

```bash
python scripts/run_activation_patching_eval.py \
  --parquet data/processed/my_processed_clinvar.parquet \
  --probe results/delta_probe_500m/probe_center_raw.joblib \
  --out results/patching_500m \
  --fig reports/figures/patching_recovery.png
```

Produces layer-wise logit recovery curves with controls.

### 3) Attention head ablation

```bash
python scripts/run_head_ablation_eval.py \
  --parquet data/processed/my_processed_clinvar.parquet \
  --probe results/delta_probe_500m/probe_center_raw.joblib \
  --out results/head_ablation_500m \
  --fig reports/figures/head_ablation_vs_k_mean_std.png
```

Runs repeated random baselines and reports mean ± std AUC as a function of the number of ablated heads.

## Notes

`results/` and `reports/figures/` are intentionally ignored; all outputs are reproducible.
