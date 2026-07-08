from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model
model_path = os.path.join("models", "spam_model_tuned.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load vectorizer
vectorizer_path = os.path.join("models", "vectorizer.pkl")
with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        email = request.form["email"]

        email_vector = vectorizer.transform([email])

        result = model.predict(email_vector)

        if result[0] == 1:
            prediction = "SPAM"
        else:
            prediction = "NOT SPAM"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)