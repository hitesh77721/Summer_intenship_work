import os
import re
import string
import contractions
import emoji
import pandas as pd

import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize once
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def read_documents(folder_path):
    """
    Read all .txt files from a folder.
    Returns:
        dict -> {filename: text}
    """
    documents = {}

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                documents[file] = f.read()

    return documents


def normalize_text(text):
    """
    Clean and preprocess text.
    """

    text = text.lower()

    text = contractions.fix(text)

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'\S+@\S+', '', text)

    text = emoji.replace_emoji(text, replace='')

    text = re.sub(r'\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [word for word in tokens if len(word) > 1]

    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]

    return " ".join(tokens)


def preprocess_documents(documents):
    """
    Normalize all documents.
    """

    normalized = {}

    for filename, text in documents.items():
        normalized[filename] = normalize_text(text)

    return normalized


def calculate_similarity(normalized_documents):
    """
    Calculate TF-IDF and cosine similarity.
    """

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(normalized_documents.values())

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix


def get_plagiarism_cases(similarity_matrix, filenames, threshold=0.70):
    """
    Return all document pairs above threshold.
    """

    results = []

    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):

            similarity = similarity_matrix[i][j]

            if similarity >= threshold:

                results.append({
                    "file1": filenames[i],
                    "file2": filenames[j],
                    "similarity": round(similarity * 100, 2)
                })

    return results

def compare_uploaded_document(uploaded_text, folder_path="data"):
    """
    Compare an uploaded document with all documents in the dataset.
    """

    # Read existing documents
    documents = read_documents(folder_path)

    # Normalize them
    normalized_documents = preprocess_documents(documents)

    # Normalize uploaded document
    normalized_uploaded = normalize_text(uploaded_text)

    # Add uploaded document
    normalized_documents["Uploaded File"] = normalized_uploaded

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(normalized_documents.values())

    # Calculate similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    filenames = list(normalized_documents.keys())

    uploaded_index = filenames.index("Uploaded File")

    results = []

    for i, filename in enumerate(filenames):

        if filename != "Uploaded File":

            similarity = similarity_matrix[uploaded_index][i]

            results.append({
                "Document": filename,
                "Similarity (%)": round(similarity * 100, 2)
            })

    # Highest similarity first
    results.sort(
        key=lambda x: x["Similarity (%)"],
        reverse=True
    )

    return results