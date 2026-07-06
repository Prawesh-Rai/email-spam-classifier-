"""
preprocess.py

This script loads the SMS Spam Collection dataset and performs
basic preprocessing including:
- Lowercasing text
- Removing punctuation
- Removing stopwords
- Tokenization
- Label encoding
- Train-test splitting
"""

import os
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Download required NLTK resources
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
    Remove punctuation and stopwords from a message.

    Parameters
    ----------
    text : str
        Input SMS message.

    Returns
    -------
    str
        Cleaned text.
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

# Tokenization

df["tokens"] = df["message"].apply(word_tokenize)

# Label Encoding

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Dataset Information

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))

print("\nDataset Preview")
print(df.head())

print("\nClass Distribution")
print(df["label"].value_counts())


spam_percentage = (df["label"].sum() / len(df)) * 100
ham_percentage = 100 - spam_percentage

print(f"\nSpam Emails : {spam_percentage:.2f}%")
print(f"Ham Emails  : {ham_percentage:.2f}%")