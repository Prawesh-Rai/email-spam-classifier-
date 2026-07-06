"""
feature_extraction.py

This script performs feature extraction on the SMS Spam Collection dataset.

Features implemented:
1. Bag of Words (CountVectorizer)
2. TF-IDF Vectorization
3. Display extracted vocabulary
4. Train-Test split on feature matrix
5. Check missing values
"""

import os
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download Required Resources

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# Load Dataset

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "SMSSpamCollection"
)

df = pd.read_csv(
    DATASET_PATH,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Text Preprocessing

df["message"] = df["message"].str.lower()

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    """
    Remove punctuation and stopwords.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    text = "".join(
        char
        for char in text
        if char not in string.punctuation
    )

    words = word_tokenize(text)

    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)


df["message"] = df["message"].apply(clean_text)

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Task 1 : Bag of Words

bow = CountVectorizer(stop_words="english")

X_bow = bow.fit_transform(df["message"])

print("Bag of Words Shape:")
print(X_bow.shape)

# Task 2 & 3 : TF-IDF

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=1000
)

X = tfidf.fit_transform(df["message"])
y = df["label"]

print("\nTF-IDF Shape:")
print(X.shape)

print("\nFirst 20 Features:")
print(tfidf.get_feature_names_out()[:20])

# Task 4 : Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# Task 5 : Missing Values

print("\nMissing Values:")
print(df.isnull().sum())