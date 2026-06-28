import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# Load dataset
import os

dataset_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "SMSSpamCollection"
)

df = pd.read_csv(
    dataset_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert text to lowercase
df["message"] = df["message"].str.lower()

# Remove punctuation and stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["message"] = df["message"].apply(clean_text)

# Tokenization
df["tokens"] = df["message"].apply(word_tokenize)

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(df.head())

print("\nClass Distribution:")
print(df["label"].value_counts())

spam_percentage = (df["label"].sum() / len(df)) * 100
ham_percentage = 100 - spam_percentage

print(f"\nSpam Emails: {spam_percentage:.2f}%")
print(f"Ham Emails: {ham_percentage:.2f}%")