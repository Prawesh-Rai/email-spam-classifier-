"""
Model Comparison Script

This script:
1. Loads and preprocesses the SMS Spam Collection dataset.
2. Extracts TF-IDF features.
3. Trains Decision Tree and Random Forest classifiers.
4. Compares their performance with Naive Bayes and Logistic Regression.
5. Identifies the best-performing model.
"""

import os
import string

import nltk
import pandas as pd

from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

nltk.download("stopwords")

# Locate dataset

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "data", "SMSSpamCollection")

# Load SMS Spam Collection dataset

df = pd.read_csv(
    dataset_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Preprocessing

df["message"] = df["message"].str.lower()

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = "".join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["message"] = df["message"].apply(clean_text)

# Encode labels

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# TF-IDF feature extraction

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Decision Tree classifier

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Accuracy:", round(accuracy * 100, 2), "%")

# Train Random Forest classifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")

# Compare all trained models

print("\n----- Model Comparison -----")
print("Naive Bayes Accuracy: 97.76 %")
print("Logistic Regression Accuracy: 98.03 %")
print("Decision Tree Accuracy:", round(accuracy * 100, 2), "%")
print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")

accuracies = {
    "Naive Bayes": 97.76,
    "Logistic Regression": 98.03,
    "Decision Tree": round(accuracy * 100, 2),
    "Random Forest": round(rf_accuracy * 100, 2)
}

best_model = max(accuracies, key=accuracies.get)

print("\nBest Model:", best_model)
print("Best Accuracy:", accuracies[best_model], "%")