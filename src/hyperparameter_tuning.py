import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df["message"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Hyperparameters to test
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train Grid Search
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(round(grid_search.best_score_ * 100, 2), "%")

# Default Logistic Regression Model
default_model = LogisticRegression(max_iter=1000)

default_model.fit(X_train, y_train)

default_accuracy = default_model.score(X_test, y_test)

print("\nDefault Model Accuracy:")
print(round(default_accuracy * 100, 2), "%")

# Tuned Logistic Regression Model
best_model = grid_search.best_estimator_

tuned_accuracy = best_model.score(X_test, y_test)

print("\nTuned Model Accuracy:")
print(round(tuned_accuracy * 100, 2), "%")

# Compare Results
print("\nAccuracy Improvement:")
print(round((tuned_accuracy - default_accuracy) * 100, 2), "%")

# Predictions using tuned model
y_pred = best_model.predict(X_test)

# Evaluation Metrics
print("\nEvaluation Metrics")
print("-------------------")

print("Accuracy :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred) * 100, 2), "%")
print("Recall   :", round(recall_score(y_test, y_pred) * 100, 2), "%")
print("F1 Score :", round(f1_score(y_test, y_pred) * 100, 2), "%")

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# -----------------------------
# Error Analysis
# -----------------------------

print("\n" + "=" * 50)
print("ERROR ANALYSIS")
print("=" * 50)

# Convert sparse matrix to DataFrame index mapping
test_messages = df.iloc[y_test.index]["message"]

# False Positives
false_positive = (y_test == 0) & (y_pred == 1)

# False Negatives
false_negative = (y_test == 1) & (y_pred == 0)

print("\nFalse Positives (Ham predicted as Spam):")
print(false_positive.sum())

for msg in test_messages[false_positive].head(5):
    print("-" * 50)
    print(msg)

print("\nFalse Negatives (Spam predicted as Ham):")
print(false_negative.sum())

for msg in test_messages[false_negative].head(5):
    print("-" * 50)
    print(msg)