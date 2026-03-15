import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_news(text, model_path):
    """
    Loads a fine-tuned model from a given path and predicts the label for a text.
    
    Args:
        text (str): The news text to classify.
        model_path (str): The path to the saved model directory.

    Returns:
        tuple: A tuple containing the predicted label and the confidence score.
    """
    # Check if a CUDA-enabled GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and the fine-tuned model
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device) # Move the model to the selected device (GPU or CPU)

    # Prepare the text for the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move tensors to the same device as the model

    # Make a prediction
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert the output logits to probabilities using the softmax function
    probabilities = torch.softmax(logits, dim=-1)
    
    # Get the confidence score and the predicted class index
    confidence, predicted_class_idx = torch.max(probabilities, dim=-1)
    
    # Map the index to the corresponding label
    labels = ["REAL", "FAKE"] #REAL,FAKE
    predicted_label = labels[predicted_class_idx.item()]
    
    return predicted_label, confidence.item()

if __name__ == "__main__":
    # --- Configuration ---
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # <<< THIS IS THE ONLY LINE THAT CHANGED >>>
    # This path now correctly points to your new XLM-RoBERTa model
    model_path = os.path.join(base_dir, 'models', 'xlm_roberta_fake_news_classifier')
    
    # --- Text to Classify ---
    # You can change this text to experiment with your new model
    news_text = "A viral screenshot claims ministers will tax social media posts starting next month."
    # --- Run Prediction ---
    prediction, confidence = predict_news(news_text, model_path)

    # --- Print Results ---
    print("-" * 30)
    print(f"Text: '{news_text}'")
    print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
    print("-" * 20)