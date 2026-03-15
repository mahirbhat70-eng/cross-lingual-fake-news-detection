import os
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_label(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    pred = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
    return pred.item()

def evaluate_f1(model_path, texts, true_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    predictions = []

    print("\nRunning Evaluation...\n")

    for text in texts:
        pred = predict_label(model, tokenizer, text, device)
        predictions.append(pred)

    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return precision, recall, f1


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'xlm_roberta_fake_news_classifier')

    # ---- Example test dataset ----
    test_texts = [
        "Government launches new affordable housing scheme.",          # REAL
        "NASA finds aliens living on Jupiter’s clouds.",               # FAKE
        "RBI reduces interest rates by 0.5%",                          # REAL
        "Drinking soda cures cancer instantly, doctors confirm.",      # FAKE
    ]

    # TRUE labels in 0/1 form (REAL=0, FAKE=1)
    true_labels = [0, 1, 0, 1]

    precision, recall, f1 = evaluate_f1(model_path, test_texts, true_labels)

    print("------ Evaluation Results ------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("--------------------------------")
