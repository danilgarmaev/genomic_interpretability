import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import List

# --- CONFIG ---
PARQUET_PATH = "data/processed/my_processed_clinvar.parquet"
# Use 100m or 500m
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species" 
# Sequence length in BASES (your parquet ref/alt are 1024bp windows).
SEQ_LEN = 1024
BATCH_SIZE = 8
# Use 'alt' by default for pathogenicity classification (the alternate allele window).
# If your parquet has a 'Sequence' column, set SEQ_COL = 'Sequence'.
SEQ_COL = "alt"

def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def load_data(n_samples=2000):
    print("Loading Parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    if SEQ_COL not in df.columns:
        raise ValueError(
            f"Expected a sequence column '{SEQ_COL}' in parquet, but columns are: {list(df.columns)}. "
            "Set SEQ_COL at the top of this file (e.g., 'ref' or 'alt' or 'Sequence')."
        )
    
    # Balance the dataset
    pos = df[df['CLNSIG'] == 'Pathogenic']
    neg = df[df['CLNSIG'] == 'Benign']
    
    n = min(len(pos), len(neg), n_samples // 2)
    balanced_df = pd.concat([pos.sample(n), neg.sample(n)]).sample(frac=1).reset_index(drop=True)
    
    print(f"Data Loaded: {len(balanced_df)} samples ({n} Pathogenic, {n} Benign)")
    return balanced_df


def _center_token_indices(tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    """Return center token index (per example) in token space.

    - We treat the biological center as nucleotide position SEQ_LEN//2.
    - Map that nucleotide position to a token index by estimating bases-per-token
      from non-special tokens.

    input_ids: (B, T)
    returns: (B,) long tensor of token indices into the T dimension.
    """

    bsz, t = input_ids.shape
    centers: List[int] = []
    nucleotide_center = SEQ_LEN // 2

    for b in range(bsz):
        ids = input_ids[b].tolist()
        special_mask = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)

        # Exclude special tokens and padding for center calculation.
        non_special = [i for i, m in enumerate(special_mask) if m == 0]
        if not non_special:
            centers.append(t // 2)
            continue

        token_count = len(non_special)
        bases_per_token = max(1, int(round(SEQ_LEN / token_count)))
        tok_pos = min(max(int(nucleotide_center // bases_per_token), 0), token_count - 1)
        centers.append(non_special[tok_pos])

    return torch.tensor(centers, dtype=torch.long, device=input_ids.device)

def extract_center_embeddings(model, tokenizer, device, sequences):
    """
    Improved Pooling: Extracts ONLY the center token embedding.
    """
    model.eval()
    embeddings = []
    
    print("Extracting Center Embeddings...")
    for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="Embedding"):
        batch_seqs = sequences[i:i+BATCH_SIZE]
        
        # IMPORTANT: max_length is in TOKEN space, not base space.
        # Do NOT pad to 1024 tokens (that would mostly be padding and break center pooling).
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            # Use base model hidden states for embeddings.
            outputs = model.esm(**inputs, return_dict=True)
        
        # Center token pooling in token space
        center_idx = _center_token_indices(tokenizer, inputs["input_ids"])  # (B,)
        
        # Shape: [Batch, Seq_Len, Hidden_Dim] -> [Batch, Hidden_Dim]
        # We take the vector at the center index
        hidden = outputs.last_hidden_state  # (B, T, D)
        batch_emb = hidden[torch.arange(hidden.shape[0], device=hidden.device), center_idx, :]
        batch_emb = batch_emb.detach().float().cpu().numpy()
        embeddings.append(batch_emb)
        
    return np.vstack(embeddings)

def train_and_evaluate(embeddings, labels):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    # Train Logistic Regression (with scaling for stability)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    clf.fit(X_train, y_train)
    
    # Predict Probabilities (Critical for AUC)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)
    
    # Metrics
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n--- RESULTS (Center Pooling) ---")
    print(f"AUC:      {auc:.4f}  <-- This is your new headline number")
    print(f"Accuracy: {acc:.4f}")
    
    # Plot ROC Curve (Nice visual for report)
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Probe Performance (Center Token)")
    plt.legend()
    plt.savefig("results/probe_auc_curve.png")
    print("Saved ROC Curve to results/probe_auc_curve.png")
    
    return clf

# --- MAIN ---
if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)
    
    # Load Data & Labels
    df = load_data()
    seqs = df[SEQ_COL].astype(str).str.upper().tolist()
    # Map labels: Pathogenic -> 1, Benign -> 0
    labels = df['CLNSIG'].map({'Pathogenic': 1, 'Benign': 0}).values
    
    # Run
    embs = extract_center_embeddings(model, tokenizer, device, seqs)
    probe = train_and_evaluate(embs, labels)