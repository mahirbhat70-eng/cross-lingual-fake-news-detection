#!/usr/bin/env python
# SCRIPT FILE: scripts/evaluate_clean_and_run.py
# Purpose: Clean labels robustly and evaluate model on human-labeled Hindi dataset
import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xlm_roberta_fake_news_classifier")
INPUT_FILE = os.path.join(BASE_DIR, "data", "dataset", "hindi_test_human_labeled.csv")
CLEANED_FILE = os.path.join(BASE_DIR, "data", "dataset", "hindi_test_human_labeled.cleaned.csv")
OUTPUT_RESULTS = os.path.join(BASE_DIR, "results", "hindi_eval_results.csv")

MAX_LEN = 512
# ----------------------------

def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    # Try common encodings if needed
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding='utf-8', engine='python')

def clean_true_label_column(df):
    # If column not present, error
    if "true_label" not in df.columns:
        raise KeyError("Column 'true_label' not found in the CSV. Please ensure it's present.")

    col = df["true_label"].astype(str).str.strip()
    # Fix common header-copy issue: if a row contains the literal column name, drop it
    col = col.replace({"": np.nan, "nan": np.nan})

    # Map textual labels to numeric: common variations
    mapping = {
        "real": 0, "r": 0, "0": 0, "true": 0, "t": 0,
        "fake": 1, "f": 1, "1": 1, "false": 1
    }
    # Lowercase for mapping
    lowered = col.str.lower().replace(mapping)

    # If still not numeric, try to coerce
    coerced = pd.to_numeric(lowered, errors='coerce')

    # Identify bad rows
    bad_mask = coerced.isna()
    bad_count = bad_mask.sum()

    if bad_count > 0:
        print(f"⚠ Found {bad_count} row(s) in 'true_label' that could not be parsed. They will be dropped.")
        # Show examples (up to 5)
        print("Examples of bad label values:")
        print(df.loc[bad_mask, "true_label"].drop_duplicates().head(10).to_list())

    # Keep only valid rows
    df_clean = df.loc[~bad_mask].copy()
    df_clean["true_label"] = coerced.loc[~bad_mask].astype(int).values
    return df_clean, bad_count

def detect_text_column(df):
    for c in ("text", "Headline", "Content"):
        if c in df.columns:
            return c
    raise KeyError("No text column found. Expected one of: 'text', 'Headline', 'Content'.")

def run_inference_and_eval(df, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    text_col = detect_text_column(df)
    texts = df[text_col].astype(str).tolist()
    true_labels = df["true_label"].astype(int).tolist()

    preds = []
    prob_fake = []
    prob_real = []

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits.cpu()
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        pred = int(probs.argmax())
        preds.append(pred)
        # assume label order [REAL, FAKE]
        prob_real.append(float(probs[0]))
        prob_fake.append(float(probs[1]))

    df_out = df.copy()
    df_out["predicted"] = preds
    df_out["pred_prob_REAL"] = prob_real
    df_out["pred_prob_FAKE"] = prob_fake

    # Metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds)
    cr = classification_report(true_labels, preds)
    try:
        roc = roc_auc_score(true_labels, prob_fake)
    except Exception:
        roc = "N/A"

    # Save outputs
    os.makedirs(os.path.dirname(OUTPUT_RESULTS), exist_ok=True)
    df_out.to_csv(OUTPUT_RESULTS, index=False, encoding="utf-8")

    # Print metrics
    print("\n=========== EVALUATION METRICS ===========")
    print(f"Samples evaluated: {len(true_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(cr)
    print(f"\nSaved predictions+probs to: {OUTPUT_RESULTS}")
    return df_out

def main():
    print(f"Loading input CSV: {INPUT_FILE}")
    df = load_df(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")
    print("Cleaning 'true_label' column...")
    try:
        df_clean, bad_count = clean_true_label_column(df)
    except KeyError as e:
        print("ERROR:", e)
        sys.exit(1)

    print(f"Cleaned dataset has {len(df_clean)} valid rows (dropped {bad_count}).")
    # Save cleaned file
    os.makedirs(os.path.dirname(CLEANED_FILE), exist_ok=True)
    df_clean.to_csv(CLEANED_FILE, index=False, encoding="utf-8")
    print(f"Saved cleaned labeled file to: {CLEANED_FILE}")

    # Run inference + evaluation
    print("Running inference and evaluation...")
    run_inference_and_eval(df_clean, MODEL_PATH)
    print("All done.")

if __name__ == "__main__":
    main()
