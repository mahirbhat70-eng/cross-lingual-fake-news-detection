import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==================================================
# 1. Load Final Model
# ==================================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "xlm_roberta_fake_news_classifier")

print(f"🔍 Loading final trained model: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==================================================
# 2. Load Hindi Test File
# ==================================================
hindi_test_path = os.path.join(base_dir, "data", "dataset", "hindi_test_fixed.csv")
df = pd.read_csv(hindi_test_path)

# Detect text column
if "text" in df.columns:
    texts = df["text"].tolist()
elif "Headline" in df.columns:
    texts = df["Headline"].tolist()
elif "Content" in df.columns:
    texts = df["Content"].tolist()
else:
    raise ValueError("❌ No text column found in CSV!")

true_labels = df["true_label"].tolist()  # REQUIRED

preds = []

# ==================================================
# 3. AGENT Loop
# ==================================================
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    preds.append(pred)

df["predicted"] = preds

# ==================================================
# 4: Ensure results directory exists
# ==================================================
output_path = os.path.join(base_dir, "results", "hindi_test_agent_results.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save predictions
df.to_csv(output_path, index=False)

# ==================================================
# 5: Evaluation Metrics
# ==================================================
accuracy = accuracy_score(true_labels, preds)
precision = precision_score(true_labels, preds)
recall = recall_score(true_labels, preds)
f1 = f1_score(true_labels, preds)

print("\n=========== FINAL AGENT EVALUATION REPORT ===========")
print(f"Model used: {model_path}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("======================================================")
print(f"\n📁 Predictions saved to: {output_path}")
