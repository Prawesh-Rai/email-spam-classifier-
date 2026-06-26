# Email Spam Classifier using Machine Learning

## Project Overview
This project demonstrates an Email Spam Classification System using Machine Learning. The model classifies incoming email messages as either **Spam** or **Not Spam** by applying Natural Language Processing (NLP) techniques and supervised learning algorithms.

## Dataset
The project uses the SMS Spam Collection dataset containing **5,572** messages with the following information:

- Message Text
- Label (Spam / Ham)

### Dataset Statistics
- Total Messages: 5,572
- Ham Messages: 4,825
- Spam Messages: 747
- Spam Percentage: 13.41%

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Pickle
- Git & GitHub

## Machine Learning Models
- Multinomial Naive Bayes
- Logistic Regression

## Project Workflow
1. Data Loading
2. Data Cleaning
3. Text Preprocessing
4. Stopword Removal
5. TF-IDF Feature Extraction
6. Train-Test Split
7. Hyperparameter Tuning
8. Model Training
9. Model Evaluation
10. Cross Validation
11. Model Comparison
12. Model Deployment
13. Model Serialization

## Results

### Multinomial Naive Bayes
- Accuracy: **98.03%**
- Best Alpha: **0.1**
- Cross Validation Accuracy: **98.03%**

### Logistic Regression
- Accuracy: **97.40%**

### Best Model
**Multinomial Naive Bayes** achieved the highest performance and was selected as the final model for deployment.

## Features
- Text preprocessing using NLTK
- Stopword removal
- TF-IDF Vectorization
- Hyperparameter tuning
- Model comparison
- Cross-validation
- Model serialization using Pickle
- Interactive command-line prediction system

## Project Structure

```
EmailSpamClassifier/
│
├── data/
│   └── SMSSpamCollection
│
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
│   ├── evaluation.py
│   ├── model_comparison.py
│   ├── spam_classifier.py
│   └── eda.py
│
├── app.py
├── test.py
└── README.md
```

## Future Improvements
- Develop a Flask Web Application
- Build a Streamlit Dashboard
- Deploy the model on Render
- Train using larger email datasets
- Integrate Deep Learning models such as LSTM and BERT
- Add REST API support

## Author

**Prawesh Kumar Rai**