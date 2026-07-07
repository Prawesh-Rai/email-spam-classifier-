"""
Model Evaluation Script

This script:
1. Loads the trained spam classification model.
2. Preprocesses the SMS Spam Collection dataset.
3. Extracts TF-IDF features.
4. Evaluates the model using multiple performance metrics.
5. Performs 5-fold cross-validation.
6. Reports confusion matrix, precision, recall, F1-score, and classification report.
"""

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

# Locate dataset

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(
    base_dir,
    "data",
    "SMSSpamCollection"
)

# Load SMS Spam Collection dataset

df = pd.read_csv(
    dataset_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Encode labels

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Encode labels

labels = df["label"]

# Split dataset into training and testing sets

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["message"],
    labels,
    test_size=0.2,
    random_state=42
)

# Load trained model

model_path = os.path.join(
    base_dir,
    "models",
    "spam_model.pkl"
)

vectorizer_path = os.path.join(
    base_dir,
    "models",
    "vectorizer.pkl"
)

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

    X_test = vectorizer.transform(X_test_text)
    X_train = vectorizer.transform(X_train_text)

# Generate predictions

y_pred = model.predict(X_test)

# Evaluate model performance

print("\nConfusion Matrix")
print("-" * 20)
print(confusion_matrix(y_test, y_pred))

print("\nPrecision Score:")
print(round(precision_score(y_test, y_pred), 4))

print("\nRecall Score:")
print(round(recall_score(y_test, y_pred), 4))

f1 = f1_score(y_test, y_pred)

print("\nF1 Score:")
print(round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nClass Distribution:")
print(df["label"].value_counts())

spam_percent = (df["label"].sum() / len(df)) * 100

print("\nSpam Percentage:", round(spam_percent, 2), "%")

print("\nOverfitting / Underfitting Analysis")

if f1 > 0.90:
    print("Model is performing well.")
else:
    print("Model may need improvement.")

    print("\n----- Cross Validation -----")

cv_model = MultinomialNB(alpha=0.1)

cv_scores = cross_val_score(
    cv_model,
    vectorizer.transform(X_train_text),
    y_train,
    cv=5,
    scoring="accuracy"
)

print("Cross Validation Scores:", cv_scores)
print("Average CV Accuracy:", round(cv_scores.mean() * 100, 2), "%")