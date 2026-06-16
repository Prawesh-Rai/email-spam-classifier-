import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

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
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# Task 5: Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())