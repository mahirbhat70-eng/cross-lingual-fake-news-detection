import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def load_data(file_path):
    """Loads a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_metrics(p):
    """Computes accuracy and macro-F1 from predictions."""
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="macro"),
    }

def main():
    # --- Paths and Configuration ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # --- Adapted to use the new BALANCED dataset ---
    processed_data_path = os.path.join(base_dir, "data", "dataset", "combined_balanced.json")
    
    # --- Using XLM-RoBERTa as requested ---
    model_name = "xlm-roberta-base"
    model_output_dir = os.path.join(base_dir, "models", "xlm_roberta_fake_news_classifier")

    # --- Load and Split Dataset ---
    print(f"🚀 Loading final BALANCED data from {processed_data_path}...")
    full_dataset = Dataset.from_list(load_data(processed_data_path))
    dataset_dict = full_dataset.train_test_split(test_size=0.2, seed=42)
    print(f"✅ Data loaded: {len(dataset_dict['train'])} train, {len(dataset_dict['test'])} test examples.")

    # --- Tokenization ---
    print("⚙️ Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    # --- Model Loading ---
    print(f"🧠 Loading pre-trained model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # --- This enables fast training on CUDA GPUs ---
        fp16=True,
        report_to="none",
    )

    # --- Initialize Standard Trainer (no weighting needed for balanced data) ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    # --- Train and Save ---
    print("🚀 Starting model training...")
    trainer.train()
    print(f"💾 Saving the best model to {model_output_dir}")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print("🎉 Training complete!")

if __name__ == "__main__":
    main()