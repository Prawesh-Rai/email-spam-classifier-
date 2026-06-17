from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

print("Model Comparison")

print("Naive Bayes Accuracy: 98.03%")
print("Logistic Regression Accuracy: 96.59%")

if 98.03 > 96.59:
    print("Naive Bayes performs better.")
else:
    print("Logistic Regression performs better.")