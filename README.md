# Genomic Interpretability

Lightweight, local-first mechanistic interpretability tooling for Nucleotide Transformer v2 on ClinVar-style variants.

## Stack
- PyTorch (`torch`) with macOS MPS acceleration when available
- Hugging Face Transformers (`transformers`) for Nucleotide Transformer v2
- Data: `pandas`, `pyarrow`
- Probe/eval: `scikit-learn`
- Viz: `matplotlib`, `seaborn`

## Golden Path (Reproducible)

All commands assume the repo root and the `.venv` interpreter.

1) Install deps
- `python -m venv .venv && . .venv/bin/activate`
- `pip install -r requirements.txt`

2) Train/evaluate probes on multiple representations (incl. Δ-embeddings)
- `./.venv/bin/python scripts/run_delta_embedding_probe.py \
	--parquet data/processed/my_processed_clinvar.parquet \
	--model InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
	--out results/delta_probe_500m \
	--n-total 2000`

Outputs:
- `results/delta_probe_500m/metrics.json`
- `results/delta_probe_500m/embeddings.npz`
- `results/delta_probe_500m/probe_*.joblib`

3) Activation patching evaluation (logit recovery vs layer + controls)
- `./.venv/bin/python scripts/run_activation_patching_eval.py \
	--parquet data/processed/my_processed_clinvar.parquet \
	--probe results/delta_probe_500m/probe_center_raw.joblib \
	--out results/patching_500m \
	--fig reports/figures/patching_recovery.png`

Outputs:
- `results/patching_500m/patching_recovery.csv`
- `reports/figures/patching_recovery.png`

4) Attention head ablation with proper controls (AUC vs K)
- `./.venv/bin/python scripts/run_head_ablation_eval.py \
	--parquet data/processed/my_processed_clinvar.parquet \
	--probe results/delta_probe_500m/probe_center_raw.joblib \
	--out results/head_ablation_500m \
	--fig reports/figures/head_ablation_vs_k.png`

Outputs:
- `results/head_ablation_500m/head_ablation_vs_k.csv`
- `reports/figures/head_ablation_vs_k.png`

## Data & Artifacts

This repo is set up so that large or derived artifacts are **not** committed by default.

- `results/` and `reports/figures/` are ignored via `.gitignore` (they are reproducible outputs).
- `data/processed/` is also ignored; you can generate it locally.

### Probe parquet schema

Several probe/mechanistic scripts expect a parquet at `data/processed/my_processed_clinvar.parquet` with:

- `ref`: reference 1024bp window sequence
- `alt`: alternate 1024bp window sequence
- `variant_type`: e.g. `single_nucleotide_variant`, `Insertion`, `Deletion`, `Indel`
- `CLNSIG`: label string (usually `Pathogenic` / `Benign`)

### Build the parquet (recommended)

1) Generate `results/sequences.jsonl` using the ClinVar VCF + a local hg38/GRCh38 FASTA:
- `./.venv/bin/python scripts/prepare_dataset.py --help`

2) Convert `results/sequences.jsonl` → `data/processed/my_processed_clinvar.parquet`:
- `./.venv/bin/python scripts/build_probe_parquet_from_sequences.py \
	--sequences results/sequences.jsonl \
	--out data/processed/my_processed_clinvar.parquet`

If you choose to publish any processed parquet or figures, double-check the terms for any upstream sources (e.g., genome FASTA-derived windows).
