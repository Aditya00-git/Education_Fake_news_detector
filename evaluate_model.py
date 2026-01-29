import os
import re
import joblib
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

nltk.download("stopwords")
from nltk.corpus import stopwords

# ----------------------------
# EXACT testing dataset path (verified)
# ----------------------------
TEST_PATH = "dataset/archive (2)/Testing_dataset/testingSet"

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def load_test_data(base_path):
    texts, labels = [], []

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if not os.path.isdir(folder_path):
            continue

        label = 0 if folder == "fake" else 1  # real → 1

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

# Load test data
X_test, y_test = load_test_data(TEST_PATH)

# Load model & vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Predict
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

# Report
print("\nClassification Report (Test Dataset)\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Fake News Detection")
plt.tight_layout()
plt.show()
