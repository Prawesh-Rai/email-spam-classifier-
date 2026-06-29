# Email Spam Classifier using Machine Learning

## Project Overview

This project is a Machine Learning-based Email Spam Classifier developed using Python and Scikit-learn. It classifies emails as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and a Logistic Regression model with TF-IDF feature extraction. The project also includes a Flask web application for real-time email spam prediction.

---

## Features

* Email text preprocessing
* TF-IDF feature extraction
* Multiple model comparison
* Model evaluation using Accuracy, Precision, Recall and F1-Score
* Cross-validation
* Flask-based web interface
* Real-time spam prediction

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Flask
* Pickle
* HTML & CSS

---

## Dataset

Dataset Used:
SMS Spam Collection Dataset

* Total Messages: 5,572
* Ham Messages: 4,825
* Spam Messages: 747

---

## Machine Learning Models Compared

* Naive Bayes
* Logistic Regression
* Decision Tree
* Random Forest

### Model Comparison

| Model               | Accuracy |
| ------------------- | -------- |
| Naive Bayes         | 97.76%   |
| Logistic Regression | 98.03%   |
| Decision Tree       | 96.14%   |
| Random Forest       | 98.03%   |

**Best Selected Model:** Logistic Regression

---

## Project Structure

```
EmailSpamClassifier/
│
├── data/
├── models/
├── notebooks/
├── src/
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── model_comparison.py
│   ├── evaluation.py
│   └── spam_classifier.py
│
├── templates/
│   └── index.html
│
├── app.py
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Prawesh-Rai/email-spam-classifier-.git
```

Go to the project directory

```bash
cd email-spam-classifier-
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Flask application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## Model Performance

* Best Accuracy: **98.03%**
* Cross Validation Accuracy: **98.03%**
* Spam Detection through Flask Web Application

---

## Screenshots

Add screenshots of:

* Home Page
* Spam Prediction
* Non-Spam Prediction

---

## Future Improvements

* Deploy on Render or Railway
* Improve UI design
* Add user authentication
* Support multiple languages
* Email attachment scanning

---

## Author

**Prawesh Kumar Rai**

B.Tech CSE (AI & Data Science)

Apeejay Stya University
