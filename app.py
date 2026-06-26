import pickle
import os

# Load model
model_path = os.path.join("models", "spam_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load vectorizer
vectorizer_path = os.path.join("models", "vectorizer.pkl")
with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

while True:
    print("\n==============================")
    email = input("Enter email text (or type 'exit'): ")

    if email.lower() == "exit":
        break

    # Convert text into TF-IDF features
    email_vector = vectorizer.transform([email])

    # Predict
    prediction = model.predict(email_vector)

    if prediction[0] == 1:
        print("\nPrediction: SPAM")
    else:
        print("\nPrediction: NOT SPAM")