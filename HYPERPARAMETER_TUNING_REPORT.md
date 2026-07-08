# Hyperparameter Tuning Report

## Project
Email Spam Classifier using Machine Learning

---

## Objective

The objective of hyperparameter tuning was to improve the performance of the Email Spam Classifier by finding the best parameters for the Logistic Regression model.

---

## Hyperparameter Tuning Technique

The model was optimized using **GridSearchCV** with **5-fold Cross Validation**.

### Parameter Grid

```python
{
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}
```

---

## Best Parameters

| Parameter | Value |
|-----------|-------|
| C | 10 |
| Solver | lbfgs |

---

## Cross Validation Result

**Best Cross Validation Accuracy:** **97.69%**

---

## Model Comparison

| Model | Accuracy |
|--------|---------:|
| Naive Bayes | 97.76% |
| Decision Tree | 96.14% |
| Random Forest | 98.03% |
| Logistic Regression (Default) | 96.59% |
| Logistic Regression (Tuned) | **98.74%** |

---

## Performance Improvement

| Metric | Value |
|--------|------:|
| Default Accuracy | 96.59% |
| Tuned Accuracy | 98.74% |
| Improvement | +2.15% |

---

## Final Evaluation Metrics

| Metric | Score |
|---------|------:|
| Accuracy | 98.74% |
| Precision | 98.56% |
| Recall | 91.95% |
| F1 Score | 95.14% |

---

## Confusion Matrix

```
[[964   2]
 [ 12 137]]
```

---

## Conclusion

GridSearchCV successfully identified the optimal hyperparameters for the Logistic Regression model. The tuned model achieved an accuracy of **98.74%**, outperforming the default model and providing excellent precision and recall for spam detection.

The tuned Logistic Regression model was selected as the final model and saved as:

- models/spam_model_tuned.pkl

This optimized model is used in the Flask web application for spam prediction.