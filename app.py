import streamlit as st
import joblib
import os
import re
import string

# --- Load Model and Vectorizer ---
MODEL_PATH = "model/naive_bayes_spam.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("🔴 Model or vectorizer file missing! Ensure they are in the 'model' folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# --- Preprocessing Function ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  
    text = re.sub(r"\d+", "", text)  
    return text

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: #ccc9dc;
        }
        .stApp {
            background-color: #0c1821;
        }
        .stTextInput, .stTextArea, .stButton>button {
            background-color: #1b2a41 !important;
            color: #ccc9dc !important;
            border-radius: 10px;
            padding: 10px;
            border: none;
            font-size: 16px;
        }
        .stTextArea textarea {
            background-color: #324a5f !important;
            color: #ccc9dc !important;
        }
        .stMarkdown, .stTitle {
            color: #ccc9dc !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- UI Layout ---
st.title("📧 Spam Email Classifier")
st.markdown("### 🔹 Enter an email below to classify:")

user_input = st.text_area("✉️ Type or paste your email here:", height=150)

if st.button("Check Email"):
    if user_input.strip():
        cleaned_text = preprocess_text(user_input)
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vectorized)[0]
        result = "🚨 **Spam**" if prediction == 1 else "✅ **Not Spam**"
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("⚠️ Please enter some text to classify.")
