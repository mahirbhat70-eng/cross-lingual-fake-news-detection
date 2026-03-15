# SCRIPT FILE: scripts/auto_error_tag.py
# Automatically generates a basic error_tag for each misclassified row using simple rules.

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIS_FILE = os.path.join(BASE, "results", "misclassified_samples.csv")

df = pd.read_csv(MIS_FILE)

def auto_tag(row):
    text = str(row["Headline"]) + " " + str(row["Content"])

    text_lower = text.lower()

    if any(w in text_lower for w in ["covid", "corona", "virus", "health", "cancer", "treatment"]):
        return "health_claim"

    if any(w in text_lower for w in ["pm", "modi", "bjp", "congress", "election", "govt"]):
        return "political_claim"

    if any(w in text_lower for w in ["miracle", "magic", "unbelievable", "अद्भुत", "चमत्कार"]):
        return "sensational_fake"

    if len(text_lower.split()) < 10:
        return "short_text"

    if len(text_lower.split()) > 60:
        return "long_text"

    if any(w in text_lower for w in ["video", "clip", "viral"]):
        return "viral_claim"

    return "other"

df["error_tag"] = df.apply(auto_tag, axis=1)

df.to_csv(MIS_FILE, index=False, encoding="utf-8")
print("✔ Auto-tagging complete. Saved updated file:", MIS_FILE)
