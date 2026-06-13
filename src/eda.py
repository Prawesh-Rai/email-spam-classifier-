import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Message length
df["message_length"] = df["message"].apply(len)

# Class distribution
print(df["label"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="label", data=df)
plt.title("Spam vs Ham Distribution")
plt.show()

# Message length distribution
plt.figure(figsize=(8,5))
sns.histplot(
    data=df,
    x="message_length",
    hue="label",
    bins=50
)
plt.title("Message Length Distribution")
plt.show()
