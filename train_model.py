import os
import re
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
nltk.download("stopwords")
from nltk.corpus import stopwords
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
        label = 0 if folder == "fake" else 1
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
    return texts, label
X_train, y_train = load_data(TRAIN_PATH)
X_test, y_test = load_data(TEST_PATH)
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=1,
    sublinear_tf=True,
    stop_words="english"
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(
    C=4.0,
    max_iter=3000,
    class_weight="balanced",
    solver="liblinear"
)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, model.predict(X_train_vec))
print("Train Accuracy:", train_acc)
print("Test Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully")
