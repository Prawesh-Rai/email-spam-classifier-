import pandas as pd
import os
import string
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download("stopwords")

# Dataset path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "data", "SMSSpamCollection")

# Load dataset
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

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["message"] = df["message"].apply(clean_text)

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Features
vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model with Hyperparameter Tuning

alphas = [0.1, 0.5, 1.0, 2.0]

best_accuracy = 0
best_alpha = 1.0

for alpha in alphas:
    model = MultinomialNB(alpha=alpha)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Alpha={alpha} -> Accuracy={accuracy*100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_alpha = alpha

print("\nBest Alpha:", best_alpha)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")

model = MultinomialNB(alpha=best_alpha)

model.fit(X_train, y_train)

# Logistic Regression Comparison

print("\n----- Logistic Regression Comparison -----")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:",
      round(lr_accuracy * 100, 2), "%")

if lr_accuracy > best_accuracy:
    print("Logistic Regression performs better.")
else:
    print("Naive Bayes performs better.")

# Save model
os.makedirs(
    os.path.join(base_dir, "models"),
    exist_ok=True
)

# Save model
with open(os.path.join(base_dir, "models", "spam_model.pkl"), "wb") as f:
    pickle.dump(model, f)

# Save TF-IDF vectorizer
with open(os.path.join(base_dir, "models", "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")