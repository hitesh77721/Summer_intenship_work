import re
import string
import contractions
import emoji

import nltk

# ==========================================================
# Download Required NLTK Resources
# ==========================================================

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# Initialize
# ==========================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ==========================================================
# Text Preprocessing
# ==========================================================

def normalize_text(text):
    """
    Clean and preprocess text.
    """

    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Expand contractions
    text = contractions.fix(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        token for token in tokens
        if token not in stop_words
    ]

    # Remove single-character words
    tokens = [
        token for token in tokens
        if len(token) > 1
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(token, pos="v")
        for token in tokens
    ]

    return " ".join(tokens)

# ==========================================================
# Compare Two Documents
# ==========================================================

def compare_documents(original_text, suspected_text):
    """
    Compare two documents using TF-IDF + Cosine Similarity.

    Returns:
        float: Similarity percentage
    """

    original = normalize_text(original_text)
    suspected = normalize_text(suspected_text)

    # If both become empty after preprocessing
    if not original and not suspected:
        return 100.0

    # If one becomes empty
    if not original or not suspected:
        return 0.0

    documents = [
        original,
        suspected
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)