"""
Hyperparameter Tuning Script

This script:
1. Loads and preprocesses the SMS Spam Collection dataset.
2. Extracts TF-IDF features.
3. Performs hyperparameter tuning using GridSearchCV.
4. Compares the tuned Logistic Regression model with the default model.
5. Evaluates the tuned model using multiple performance metrics.
6. Displays false positive and false negative predictions.
"""

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

# Encode class labels

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Extract TF-IDF features

vectorizer = TfidfVectorizer(max_features=1000)

features = vectorizer.fit_transform(df["message"])
labels = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42
)

# Initialize Logistic Regression model

logistic_model = LogisticRegression(max_iter=1000)

# Define hyperparameter search space

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}

# Configure GridSearchCV

grid_search = GridSearchCV(
    estimator=logistic_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Perform hyperparameter tuning

grid_search.fit(X_train, y_train)

# Display best hyperparameters

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(round(grid_search.best_score_ * 100, 2), "%")

# Train default Logistic Regression model

default_model = LogisticRegression(max_iter=1000)

default_model.fit(X_train, y_train)

default_accuracy = default_model.score(X_test, y_test)

print("\nDefault Model Accuracy:")
print(round(default_accuracy * 100, 2), "%")

# Evaluate tuned Logistic Regression model

tuned_model = grid_search.best_estimator_

model_path = os.path.join(base_dir, "models", "spam_model_tuned.pkl")

with open(model_path, "wb") as f:
    pickle.dump(tuned_model, f)

print("\nTuned model saved successfully.")
print("Saved at:", model_path)

tuned_accuracy = tuned_model.score(X_test, y_test)

print("\nTuned Model Accuracy:")
print(round(tuned_accuracy * 100, 2), "%")

# Compare model performance

print("\nAccuracy Improvement:")
print(round((tuned_accuracy - default_accuracy) * 100, 2), "%")

# Generate predictions using tuned model

y_pred = tuned_model.predict(X_test)

# Display evaluation metrics

print("\nEvaluation Metrics")
print("-------------------")

print("Accuracy :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred) * 100, 2), "%")
print("Recall   :", round(recall_score(y_test, y_pred) * 100, 2), "%")
print("F1 Score :", round(f1_score(y_test, y_pred) * 100, 2), "%")

# Display confusion matrix

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Display classification report

print("\nClassification Report")
print(classification_report(y_test, y_pred))


# Error Analysis


print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

# Retrieve original test messages

test_messages = df.iloc[y_test.index]["message"]

# Identify false positives

false_positive = (y_test == 0) & (y_pred == 1)

# Identify false negatives

false_negative = (y_test == 1) & (y_pred == 0)

print("\nFalse Positives (Ham predicted as Spam):")
print(f"Total False Positives: {false_positive.sum()}")

for msg in test_messages[false_positive].head(5):
    print("-" * 60)
    print(msg)

print("\nFalse Negatives (Spam predicted as Ham):")
print(f"Total False Negatives: {false_negative.sum()}")

for msg in test_messages[false_negative].head(5):
    print("-" * 60  )
    print(msg)