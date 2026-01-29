import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# REQUIRED downloads (important)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def summarize_text(text, max_sentences=2):
    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Sentence tokenization
    sentences = sent_tokenize(text)

    if len(sentences) <= max_sentences:
        return text

    # Word tokenization
    words = word_tokenize(text.lower())

    # Word frequency
    word_freq = {}
    for word in words:
        if word.isalpha() and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    if not word_freq:
        return sentences[0]

    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] /= max_freq

    # Sentence scoring
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

    # Select top sentences
    summary_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )[:max_sentences]

    return " ".join(summary_sentences)
