import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ----------------------------
# SETUP MODEL PATH
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "xlm_roberta_fake_news_classifier")


# ----------------------------
# LOAD MODEL + TOKENIZER
# ----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_news(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)
    confidence, predicted_class_idx = torch.max(probabilities, dim=-1)

    # 🔥 EXACT SAME LABEL MAPPING AS predict.py
    labels = ["REAL", "FAKE"]
    predicted_label = labels[predicted_class_idx.item()]

    return predicted_label, confidence.item()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="AI Fake News Detector", layout="wide")

st.sidebar.title("Configuration")
st.sidebar.success("Model Loaded: XLM-RoBERTa (Advanced)")

st.markdown(
    "<h2 style='text-align:center;'>📰 AI Fake News Detector</h2>"
    "<h4 style='text-align:center;'>Cross-Lingual Fake News Detection using XLM-RoBERTa</h4>",
    unsafe_allow_html=True
)

st.markdown("### Paste News Article Here:")
user_input = st.text_area("", height=220)

if st.button("Check Authenticity", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter a news article.")
    else:
        label, confidence = predict_news(user_input)

        # 🔥 EXACT OUTPUT FORMAT FROM predict.py
        if label == "FAKE":
            st.error(f"✘ FAKE NEWS — Confidence: {confidence*100:.2f}%")
        else:
            st.success(f"✔ REAL NEWS — Confidence: {confidence*100:.2f}%")
