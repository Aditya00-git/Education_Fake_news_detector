import joblib
import re
import nltk

from summarize_text import summarize_text

nltk.download("stopwords")
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

print("🎓 Student Fake News Detector")
print("Enter news text (type EXIT to quit)")

while True:
    news = input("\nNews: ")

    if news.lower() == "exit":
        print("Exiting detector.")
        break

    cleaned = clean_text(news)
    word_count = len(cleaned.split())

    if word_count < 15:
        print("⚠️ Text too short for reliable prediction.")
        continue

    vector = vectorizer.transform([cleaned])
    prob = model.predict_proba(vector)[0]

    fake_prob = prob[0]
    real_prob = prob[1]

    print(f"\nWord Count: {word_count}")
    print(f"Confidence → Fake: {fake_prob:.2f}, Real: {real_prob:.2f}")

    if real_prob >= 0.50:
        print("Prediction: REAL NEWS 🟢")
    else:
        print("Prediction: FAKE NEWS 🔴")

    # ✅ SUMMARY SHOULD COME HERE (AFTER PREDICTION)
    print("\n📝 Generated Summary:")
    summary = summarize_text(news, max_sentences=2)
    print(summary)
