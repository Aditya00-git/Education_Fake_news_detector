import streamlit as st
import joblib
import re
import nltk

from summarize_text import summarize_text

nltk.download("stopwords")
from nltk.corpus import stopwords

# ----------------------------
# Load model and vectorizer
# ----------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

# ----------------------------
# Text cleaning function
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fake News Detector for Students", layout="centered")

st.title("📰 Fake News Detector for Students")
st.write(
    "This AI-based system analyzes news articles, assesses credibility, "
    "and provides a concise summary to help students avoid misinformation."
)

news_text = st.text_area(
    "Paste a news article here:",
    height=220,
    placeholder="Enter a full news article (minimum 15 words recommended)..."
)

if st.button("Check News"):
    if not news_text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(news_text)
        word_count = len(cleaned.split())

        if word_count < 15:
            st.warning("Text is too short for reliable prediction. Please enter a longer article.")
        else:
            vector = vectorizer.transform([cleaned])
            prob = model.predict_proba(vector)[0]

            fake_prob = prob[0]
            real_prob = prob[1]

            st.subheader("📊 Prediction Result")

            # Decision logic
            if real_prob >= 0.50:
                st.success("REAL NEWS 🟢")
            else:
                st.error("FAKE NEWS 🔴")

            st.write(f"**Confidence Scores:**")
            st.write(f"- Fake: `{fake_prob:.2f}`")
            st.write(f"- Real: `{real_prob:.2f}`")

            st.subheader("📝 Concise Summary")
            summary = summarize_text(news_text, max_sentences=2)
            st.info(summary)
