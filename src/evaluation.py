import os
import pickle
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Dataset path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(
    base_dir,
    "data",
    "SMSSpamCollection"
)

# Load dataset
df = pd.read_csv(
    dataset_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# TF-IDF Features
vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df["message"])

y = df["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load saved model
model_path = os.path.join(
    base_dir,
    "models",
    "spam_model.pkl"
)

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPrecision:")
print(round(precision_score(y_test, y_pred), 4))

print("\nRecall:")
print(round(recall_score(y_test, y_pred), 4))

print("\nF1 Score:")
print(round(f1_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nClass Distribution:")
print(df["label"].value_counts())

spam_percent = (df["label"].sum() / len(df)) * 100

print("\nSpam Percentage:", round(spam_percent, 2), "%")

print("\nOverfitting / Underfitting Analysis")

if f1_score(y_test, y_pred) > 0.90:
    print("Model is performing well.")
else:
    print("Model may need improvement.")

    print("\n----- Cross Validation -----")

cv_model = MultinomialNB(alpha=0.1)

cv_scores = cross_val_score(
    cv_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("Cross Validation Scores:", cv_scores)
print("Average CV Accuracy:", round(cv_scores.mean() * 100, 2), "%")