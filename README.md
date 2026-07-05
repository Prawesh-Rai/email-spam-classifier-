# Email Spam Classifier using Machine Learning

## Project Overview

This project is a Machine Learning-based Email Spam Classifier developed using Python and Scikit-learn. It classifies SMS and email messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and a Logistic Regression model with TF-IDF feature extraction. The project also includes hyperparameter tuning, comprehensive model evaluation, error analysis, and a Flask web application for real-time spam prediction.

---

## Features

* Email/SMS text preprocessing
* TF-IDF feature extraction
* Multiple Machine Learning model comparison
* Hyperparameter tuning using GridSearchCV
* Model evaluation using Accuracy, Precision, Recall and F1-Score
* Confusion Matrix and Classification Report
* Error analysis (False Positives & False Negatives)
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

**Dataset Used:** SMS Spam Collection Dataset

* Total Messages: **5,572**
* Ham Messages: **4,825**
* Spam Messages: **747**

---

## Machine Learning Models Compared

* Naive Bayes
* Logistic Regression
* Decision Tree
* Random Forest

### Model Comparison

| Model | Accuracy |
|-------|----------|
| Naive Bayes | 97.76% |
| Logistic Regression | 98.03% |
| Decision Tree | 96.14% |
| Random Forest | 98.03% |

**Best Selected Model:** Logistic Regression

---

## Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** to optimize the Logistic Regression model.

### Best Parameters

```text
C = 10
Solver = liblinear
```

### Cross Validation Accuracy

**98.16%**

---

## Project Structure

```text
EmailSpamClassifier/
│
├── data/
├── models/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│
├── src/
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── model_comparison.py
│   ├── hyperparameter_tuning.py
│   ├── evaluation.py
│   ├── spam_classifier.py
│   └── error_analysis.txt
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

```text
http://127.0.0.1:5000
```

---

## Model Performance

### Before Hyperparameter Tuning

* Accuracy: **97.85%**

### After Hyperparameter Tuning

* Accuracy: **98.92%**
* Precision: **98.58%**
* Recall: **93.29%**
* F1-Score: **95.86%**

### Accuracy Improvement

* **97.85% → 98.92%**
* **Improvement: +1.08%**

---

## Error Analysis

### False Positives

* **2** Ham messages classified as Spam.

### False Negatives

* **10** Spam messages classified as Ham.

### Observations

* Some conversational spam messages resemble normal messages.
* Reminder-style spam messages appear similar to legitimate notifications.
* TF-IDF relies on word frequency and cannot fully understand message context.

### Possible Improvements

* Increase training dataset
* Better text preprocessing
* Stemming and Lemmatization
* Support Vector Machine (SVM)
* Random Forest optimization
* XGBoost
* BERT Transformer Model

---

## Screenshots

Add screenshots of:

* Home Page
* Spam Prediction
* Non-Spam Prediction
* Model Evaluation Output

---

## Live Demo

Deployment Link:

```text
Will be added after deployment (Week 5 Friday)
```

---

## Demo Video

Video Link:

```text
Will be added after project completion
```

---

## Future Improvements

* Deploy on Render or Railway
* Improve UI design
* Add user authentication
* Support multiple languages
* Email attachment scanning
* REST API support
* Deep Learning-based spam detection

---

## Author

**Prawesh Kumar Rai**

B.Tech CSE (AI & Data Science)

Apeejay Stya University

GitHub: https://github.com/Prawesh-Rai