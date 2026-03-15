import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from tqdm import tqdm

def evaluate_on_new_language(model_path, test_data_path):
    # This line checks that your trained model exists.
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path does not exist: {model_path}")
        return

    # This line automatically detects and prepares to use your CUDA GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")
    print(f"Loading model from: {model_path}")

    # These lines load your fine-tuned XLM-RoBERTa model and its tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # This line checks that your non-English test file exists.
    print(f"📂 Loading test data from: {test_data_path}")
    if not os.path.exists(test_data_path):
        print(f"❌ Error: Test data file not found at {test_data_path}")
        return
        
    # This line reads your CSV file into memory.
    df = pd.read_csv(test_data_path)

    # --- Column Standardization Section ---
    # This block renames the columns from your CSV to the names the script expects ('text' and 'label').
    if 'headline' in df.columns:
        df.rename(columns={'headline': 'text'}, inplace=True)
    if 'Category' in df.columns:
        df.rename(columns={'Category': 'label'}, inplace=True)
    
    # This block handles text-based labels (e.g., 'True'/'False') and converts them to numbers (0/1).
    if df['label'].dtype == 'object':
        # *** IMPORTANT: Use the output from Step 2 to set the correct labels here. ***
        label_map = {'TRUE': 0, 'FALSE': 1} 
        df['label'] = df['label'].str.upper().map(label_map)
    
    # These lines clean the data to make sure it's ready for the model.
    df.dropna(subset=['label', 'text'], inplace=True)
    df['label'] = df['label'].astype(int)

    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    predictions = []

    # This loop goes through every news article in your test file, one by one.
    print(f"🚀 Running predictions on {len(texts)} test samples...")
    for text in tqdm(texts, desc="Evaluating"):
        # The text is tokenized and sent to the GPU.
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        # The model makes a prediction without any retraining (this is the "zero-shot" part).
        with torch.no_grad():
            logits = model(**inputs).logits
        # The prediction is saved to a list.
        predictions.append(torch.argmax(logits, dim=-1).item())

    # This final block compares the model's predictions to the true labels and prints the results.
    print("\n" + "="*50 + "\n📊 Zero-Shot Performance Evaluation Results\n" + "="*50)
    print(classification_report(true_labels, predictions, target_names=["REAL (0)", "FAKE (1)"]))
    print("="*50)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    english_model_path = os.path.join(base_dir, 'models', 'xlm_roberta_fake_news_classifier')
    multilingual_test_path = os.path.join(base_dir, 'data', 'dataset', 'hindi_test.csv')
    evaluate_on_new_language(english_model_path, multilingual_test_path)