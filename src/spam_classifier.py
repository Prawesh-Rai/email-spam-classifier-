import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset Loaded Successfully")
print(df.head())

# Convert labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Features and target
X = df["message"]
y = df["label"]

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)