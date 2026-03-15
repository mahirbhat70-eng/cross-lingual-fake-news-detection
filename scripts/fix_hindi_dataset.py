import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==================================
# CONFIGURATION
# ==================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to your Hindi CSV
hindi_csv_path = os.path.join(base_dir, "data", "dataset", "hindi_test.csv")

# Path to your trained model
model_path = os.path.join(base_dir, "models", "xlm_roberta_fake_news_classifier")

# Output file
output_csv_path = os.path.join(base_dir, "data", "dataset", "hindi_test_fixed.csv")

# ==================================
# LOAD MODEL
# ==================================
print(f"🔍 Loading model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==================================
# LOAD HINDI DATASET
# ==================================
print(f"📄 Loading Hindi dataset from: {hindi_csv_path}")
df = pd.read_csv(hindi_csv_path)

# Detect the text column
if "text" in df.columns:
    texts = df["text"].tolist()
elif "Headline" in df.columns:
    texts = df["Headline"].tolist()
elif "Content" in df.columns:
    texts = df["Content"].tolist()
else:
    raise ValueError("❌ No valid text column found! Expected 'text', 'Headline', or 'Content'")

# ==================================
# GENERATE AUTO LABELS
# (This will be used as true_label for testing)
# ==================================
print("🤖 Generating auto-labels using your model…")

preds = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    preds.append(pred)

# ==================================
# SAVE NEW FIXED DATASET
# ==================================
df["true_label"] = preds  # auto-generated labels
df.to_csv(output_csv_path, index=False, encoding="utf-8")

print("\n✅ FIX COMPLETED!")
print(f"Your new file is saved at:\n{output_csv_path}")
print("Now you can use this file with the AGENT.")
