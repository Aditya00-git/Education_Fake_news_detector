import os
import re
import joblib
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download("stopwords")
from nltk.corpus import stopwords

# ----------------------------
# EXACT dataset paths (verified)
# ----------------------------
TRAIN_PATH = "dataset/archive (2)/training/training/fakeNewsDataset"
TEST_PATH = "dataset/archive (2)/Testing_dataset/testingSet"

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def load_data(base_path):
    texts, labels = [], []

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if not os.path.isdir(folder_path):
            continue

        label = 0 if folder == "fake" else 1  # legit / real → 1

        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                with open(
                    os.path.join(folder_path, file),
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:
                    texts.append(clean_text(f.read()))
                    labels.append(label)

    return texts, labels

# Load data
X_train, y_train = load_data(TRAIN_PATH)
X_test, y_test = load_data(TEST_PATH)

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test_vec))
print("Training completed")
print("Test Accuracy:", accuracy)

# Save model
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully")
