# SCRIPT FILE: scripts/extract_errors.py

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(BASE_DIR, "results", "hindi_eval_results.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "results", "misclassified_samples.csv")

df = pd.read_csv(RESULTS_FILE)

errors = df[df["predicted"] != df["true_label"]].copy()
errors.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"Saved {len(errors)} misclassified samples to: {OUTPUT_FILE}")
