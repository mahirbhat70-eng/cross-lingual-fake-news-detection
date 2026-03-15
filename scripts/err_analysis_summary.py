# SCRIPT FILE: scripts/err_analysis_summary.py

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIS_FILE = os.path.join(BASE_DIR, "results", "misclassified_samples.csv")

df = pd.read_csv(MIS_FILE)

if "error_tag" not in df.columns:
    print("❗ Please add an 'error_tag' column in misclassified_samples.csv first.")
else:
    print("Error Tag Frequency:")
    print(df["error_tag"].value_counts())

    out = os.path.join(BASE_DIR, "results", "errors_by_tag_sample.csv")
    df.groupby("error_tag").head(5).to_csv(out, index=False, encoding="utf-8")
    print("Saved sample-per-tag file to:", out)
