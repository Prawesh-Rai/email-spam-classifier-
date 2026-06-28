import pandas as pd
import os
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

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

df["message"] = df["message"].str.lower()

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = "".join(
        ch for ch in text
        if ch not in string.punctuation
    )

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["message"] = df["message"].apply(clean_text)

# Convert labels to numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Task 1: Bag of Words

bow = CountVectorizer(stop_words="english")
X_bow = bow.fit_transform(df["message"])

print("Bag of Words Shape:")
print(X_bow.shape)

# Task 2 & 3: TF-IDF + Top 1000 Features

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

# Task 4: Train-Test Feature Matrix

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

# Task 5: Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())