import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------
# Accuracy Comparison
# -------------------------

models = [
    "Naive Bayes",
    "Decision Tree",
    "Random Forest",
    "Logistic Regression"
]

accuracies = [
    97.76,
    96.14,
    98.03,
    98.74
]

plt.figure(figsize=(8,5))
bars = plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(90,100)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height()+0.1,
        f"{bar.get_height():.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

# -------------------------
# Confusion Matrix
# -------------------------

cm = np.array([
    [964,2],
    [12,137]
])

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham","Spam"],
    yticklabels=["Ham","Spam"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -------------------------
# Precision Recall F1
# -------------------------

metrics = [
    "Precision",
    "Recall",
    "F1 Score"
]

values = [
    98.56,
    91.95,
    95.14
]

plt.figure(figsize=(7,5))

bars = plt.bar(metrics, values)

plt.ylim(80,100)
plt.title("Evaluation Metrics")

for bar in bars:
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.3,
        f"{bar.get_height():.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("evaluation_metrics.png")
plt.show()

print("Graphs saved successfully!")