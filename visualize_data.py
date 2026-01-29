import os
import re
import nltk
import matplotlib.pyplot as plt

nltk.download("stopwords")
from nltk.corpus import stopwords

# ----------------------------
# EXACT training dataset path (verified)
# ----------------------------
DATA_PATH = "dataset/archive (2)/training/training/fakeNewsDataset"

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

        label = "Fake" if folder == "fake" else "Real"  # legit → Real

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

texts, labels = load_data(DATA_PATH)

# ----------------------------
# Class distribution
# ----------------------------
fake_count = labels.count("Fake")
real_count = labels.count("Real")

plt.figure(figsize=(5, 4))
plt.bar(["Fake", "Real"], [fake_count, real_count])
plt.title("Fake vs Real News Distribution (Training Data)")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.show()

# ----------------------------
# Text length distribution
# ----------------------------
fake_lengths = [len(t.split()) for t, l in zip(texts, labels) if l == "Fake"]
real_lengths = [len(t.split()) for t, l in zip(texts, labels) if l == "Real"]

plt.figure(figsize=(6, 4))
plt.hist(fake_lengths, bins=50, alpha=0.6, label="Fake")
plt.hist(real_lengths, bins=50, alpha=0.6, label="Real")
plt.legend()
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.title("Text Length Distribution")
plt.tight_layout()
plt.show()
