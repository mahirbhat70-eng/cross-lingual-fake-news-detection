import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "xlm_roberta_fake_news_classifier")

input_file = os.path.join(base_dir, "data", "dataset", "hindi_test.csv")
output_file = os.path.join(base_dir, "data", "dataset", "hindi_test_labeled.csv")

df = pd.read_csv(input_file)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Detect text column
if "text" in df.columns:
    texts = df["text"].tolist()
elif "Headline" in df.columns:
    texts = df["Headline"].tolist()
else:
    texts = df["Content"].tolist()

labels = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    labels.append(pred)

df["true_label"] = labels

df.to_csv(output_file, index=False)
print(f"✔ Created labeled file: {output_file}")
