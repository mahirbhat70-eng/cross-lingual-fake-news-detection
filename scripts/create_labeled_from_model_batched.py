#!/usr/bin/env python
# SCRIPT FILE: scripts/create_labeled_from_model_batched.py
# Purpose: Create a labeled CSV from hindi_test.csv by running batched inference
# Usage: python scripts/create_labeled_from_model_batched.py

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import math

# --------------------- CONFIG ---------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "xlm_roberta_fake_news_classifier")
INPUT_CSV = os.path.join(BASE_DIR, "data", "dataset", "hindi_test.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "dataset", "hindi_test_labeled.csv")

BATCH_SIZE = 32            # <--- tune this depending on your GPU/CPU memory
MAX_LENGTH = 256           # <--- limit token length to speed up inference
USE_GPU = True             # set to False to force CPU
SHOW_PROGRESS = True
# --------------------------------------------------

def get_device():
    if USE_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def detect_text_column(df):
    for col in ("text", "Headline", "Content"):
        if col in df.columns:
            return col
    raise ValueError("No text column found. Expected one of: text, Headline, Content")

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size], i, min(i+batch_size, len(lst))

def main():
    # 1) load CSV
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file missing: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    text_col = detect_text_column(df)
    texts = df[text_col].astype(str).tolist()

    # 2) load model + tokenizer
    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    device = get_device()
    model.to(device)
    print(f"Using device: {device}")

    # 3) prepare outputs
    preds = [None] * len(texts)
    prob_fake = [None] * len(texts)
    prob_real = [None] * len(texts)

    # 4) batched inference
    num_batches = math.ceil(len(texts) / BATCH_SIZE)
    iterator = enumerate(batchify(texts, BATCH_SIZE), start=1)
    if SHOW_PROGRESS:
        pbar = tqdm(total=len(texts), desc="Inferring", unit="rows")
    else:
        pbar = None

    try:
        for batch_idx, (batch_texts, start_i, end_i) in iterator:
            # batch_texts: list of strings
            enc = tokenizer(batch_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=MAX_LENGTH)
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
                logits = outputs.logits.cpu()
                probs = torch.softmax(logits, dim=-1).numpy()  # shape (B, num_labels)
                batch_preds = probs.argmax(axis=1).tolist()

            # store results
            for offset, pred in enumerate(batch_preds):
                idx = start_i + offset
                preds[idx] = int(pred)
                # assuming class order is [REAL, FAKE] (same as your predict.py)
                prob_real[idx] = float(probs[offset, 0])
                prob_fake[idx] = float(probs[offset, 1])

            if pbar:
                pbar.update(len(batch_texts))

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Saving partial results...")
    finally:
        if pbar:
            pbar.close()

    # 5) attach columns to DataFrame (use None -> -1 if you prefer)
    df["true_label"] = preds
    df["pred_prob_REAL"] = prob_real
    df["pred_prob_FAKE"] = prob_fake

    # ensure output folder exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved labeled file to: {OUTPUT_CSV}")
    print("Done.")

if __name__ == "__main__":
    main()
