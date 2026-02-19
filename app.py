import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from summarize_text import summarize_text
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}

.stTextArea textarea {
    background-color: #1c1f26;
    color: white;
    border-radius: 10px;
}

.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #2E8B57);
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1c1f26;
    box-shadow: 0px 0px 15px rgba(0,255,0,0.2);
    margin-top: 20px;
}
.real-badge {
    color: #00ff88;
    font-size: 24px;
    font-weight: bold;
}

.fake-badge {
    color: #ff4b4b;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer
model, vectorizer = load_model()
stop_words = set(stopwords.words("english"))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)
st.sidebar.title(" About This Project")
st.sidebar.write("""
This AI system detects fake news using:
- TF-IDF Vectorization
- Logistic Regression
- Extractive Summarization
Designed for student safety and digital literacy.
""")
st.sidebar.markdown("---")
st.sidebar.write("Developed by **Aditya Seswani**")
st.markdown("<h1 style='text-align:center;'> Fake News Detector for Students</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze news credibility and get a concise summary instantly.</p>", unsafe_allow_html=True)
st.markdown("### Paste News Article Below")
news_text = st.text_area(
    "Paste your news article here:",
    height=250,
    label_visibility="collapsed"
)
if st.button(" Analyze News"):
    if not news_text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(news_text)
        word_count = len(cleaned.split())
        if word_count < 15:
            st.warning("Text too short for reliable prediction.")
        else:
            vector = vectorizer.transform([cleaned])
            prob = model.predict_proba(vector)[0]
            fake_prob = prob[0]
            real_prob = prob[1]
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("📊 Prediction Result")
            if abs(real_prob - fake_prob) < 0.10:
                st.info("⚠ Model is uncertain about this article.")
            elif real_prob > fake_prob:
                st.markdown("<p class='real-badge'>🟢 REAL NEWS</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='fake-badge'>🔴 FAKE NEWS</p>", unsafe_allow_html=True)
            st.markdown("#### Confidence Scores")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Real Probability")
                st.progress(float(real_prob))
                st.write(f"{real_prob:.2f}")
            with col2:
                st.write("Fake Probability")
                st.progress(float(fake_prob))
                st.write(f"{fake_prob:.2f}")
            st.markdown("#### 📝 Summary")
            summary = summarize_text(news_text, max_sentences=2)
            st.write(summary)
            st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align:center;'>⚖ AI-generated analysis. Always verify from trusted sources.</p>", unsafe_allow_html=True)
