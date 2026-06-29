import os
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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

# Preprocessing
df["message"] = df["message"].str.lower()

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = "".join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["message"] = df["message"].apply(clean_text)

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# TF-IDF Features
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Accuracy:", round(accuracy * 100, 2), "%")

# Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")

# Model Comparison
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