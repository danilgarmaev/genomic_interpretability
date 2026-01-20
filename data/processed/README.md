# Processed ClinVar parquet

This folder contains a small processed dataset used by the probe/mechanistic evaluation scripts.

## File: `my_processed_clinvar.parquet`

Schema:
- `ref`: reference 1024bp window sequence (uppercase)
- `alt`: alternate 1024bp window sequence (uppercase)
- `variant_type`: e.g. `single_nucleotide_variant`, `Insertion`, `Deletion`, `Indel`
- `CLNSIG`: clinical significance label (typically `Pathogenic` or `Benign`)

Notes:
- This parquet is **derived** from upstream public variant resources + a reference genome FASTA.
- If you prefer to regenerate locally, follow the dataset pipeline documented in the repository README (ClinVar VCF + hg38/GRCh38 FASTA → `results/sequences.jsonl` → parquet).
