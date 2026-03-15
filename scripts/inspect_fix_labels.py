# SCRIPT FILE: scripts/inspect_fix_labels.py
import pandas as pd, os
base = os.path.dirname(os.path.dirname(__file__))
inp = os.path.join(base, "data", "dataset", "hindi_test_human_labeled.csv")
df = pd.read_csv(inp, dtype=str)
print("Columns:", df.columns.tolist())
print("First 10 rows of true_label column:", df["true_label"].head(10).tolist())
# Show rows where true_label is non-numeric
bad = df[~df["true_label"].astype(str).str.strip().str.match(r'^[01]$')]
print("Bad rows (display up to 20):")
print(bad.head(20))
# Optionally save a fixed copy where we drop bad rows
df_fixed = df[df["true_label"].astype(str).str.strip().str.match(r'^[01]$')].copy()
df_fixed["true_label"] = df_fixed["true_label"].astype(int)
out = os.path.join(base, "data", "dataset", "hindi_test_human_labeled.cleaned2.csv")
df_fixed.to_csv(out, index=False, encoding='utf-8')
print("Saved cleaned copy:", out)
