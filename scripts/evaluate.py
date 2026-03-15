#!/usr/bin/env python
# SCRIPT FILE: scripts/evaluate.py
# PURPOSE: Evaluate model on REAL human-labeled Hindi dataset

import os
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ================================
# CONFIG
# ================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MODEL PATH
model_path = os.path.join(base_dir, "models", "xlm_roberta_fake_news_classifier")

# HINDI HUMAN-LABELED DATASET
file_in = os.path.join(base_dir, "data", "dataset", "hindi_test_human_labeled.csv")

# OUTPUT CSV
output_path = os.path.join(base_dir, "results", "hindi_eval_results.csv")


# ================================
# LOAD MODEL
# ================================
print(f"🔍 Loading model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Using device: {device}")


# ================================
# LOAD DATASET
# ================================
if not os.path.exists(file_in):
    raise FileNotFoundError(f"❌ Labeled dataset not found: {file_in}")

df = pd.read_csv(file_in)

# Detect text column
if "text" in df.columns:
    texts = df["text"].tolist()
elif "Headline" in df.columns:
    texts = df["Headline"].tolist()
elif "Content" in df.columns:
    texts = df["Content"].tolist()
else:
    raise ValueError("❌ No text column found! (Expected: text, Headline, Content)")

true_labels = df["true_label"].astype(int).tolist()

preds = []
prob_fake = []
prob_real = []


# ================================
# INFERENCE
# ================================
print("\n⚙ Running inference on labeled dataset...\n")

for text in texts:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        out = model(**encoded)

    logits = out.logits.cpu().numpy()[0]
    pred = logits.argmax()

    preds.append(int(pred))

    # class order = [REAL, FAKE]
    prob_real.append(float(torch.softmax(torch.tensor(logits), dim=0)[0]))
    prob_fake.append(float(torch.softmax(torch.tensor(logits), dim=0)[1]))


df["predicted"] = preds
df["pred_prob_REAL"] = prob_real
df["pred_prob_FAKE"] = prob_fake


# ================================
# METRICS
# ================================
accuracy = accuracy_score(true_labels, preds)
precision = precision_score(true_labels, preds)
recall = recall_score(true_labels, preds)
f1 = f1_score(true_labels, preds)
cm = confusion_matrix(true_labels, preds)
cls_report = classification_report(true_labels, preds)

try:
    roc = roc_auc_score(true_labels, prob_fake)
except:
    roc = "N/A"


# ================================
# PRINT RESULTS
# ================================
print("=========== FINAL MODEL EVALUATION ===========")
print(f"Model used: {model_path}")
print(f"Dataset:    {file_in}")
print("-----------------------------------------------")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1 Score:   {f1:.4f}")
print(f"ROC AUC:    {roc}")
print("-----------------------------------------------")
print("Confusion Matrix:")
print(cm)
print("-----------------------------------------------")
print("Classification Report:")
print(cls_report)
print("================================================\n")


# ================================
# SAVE RESULTS
# ================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"📁 Saved detailed results to:\n{output_path}")
print("✅ Evaluation completed successfully!")
