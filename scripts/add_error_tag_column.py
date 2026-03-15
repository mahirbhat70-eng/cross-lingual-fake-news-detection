# SCRIPT FILE: scripts/add_error_tag_column.py

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIS_FILE = os.path.join(BASE_DIR, "results", "misclassified_samples.csv")

df = pd.read_csv(MIS_FILE)

if "error_tag" not in df.columns:
    df["error_tag"] = ""   # create empty column
    df.to_csv(MIS_FILE, index=False, encoding="utf-8")
    print("✔ Added empty 'error_tag' column.")
else:
    print("✔ 'error_tag' column already exists.")
