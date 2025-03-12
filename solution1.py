import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# 2. Split the dataset into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train a Naïve Bayes classifier

model = GaussianNB()
model.fit(X_train, y_train)

# 4. Predict on the test set
y_pred = model.predict(X_test)

# 5. Print out the accuracy
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.2f}")

confusion_matrix = metrics.confusion_matrix(y_pred, y_test)
print(confusion_matrix)
# (Optional) Display the confusion matrix for deeper insight