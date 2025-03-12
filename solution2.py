import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# (Optional) Convert to a Pandas DataFrame for easier viewing
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y
#print(df.head())  # Uncomment to inspect

# 2. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train a Naïve Bayes classifier (from Exercise 1)
model_bayes = GaussianNB()
model_bayes.fit(X_train, y_train)

# 4. Train a Logistic Regression model
model_regr = LogisticRegression()
model_regr.fit(X_train, y_train)

# 5. Compare metrics: accuracy, precision, and recall for each model
# Note: Because we have three classes in the Wine dataset, we set average='macro' (or 'weighted') for multi-class
y_pred_regr = model_regr.predict(X_test)
y_pred_bayes = model_bayes.predict(X_test)


# 6. Print results
accuracy_regr = (y_pred_regr == y_test).mean()
accuracy_bayes = (y_pred_bayes == y_test).mean()
print(f"Test Accuracy regr: {accuracy_regr:.2f}")
print(f"Test Accuracy bayes: {accuracy_bayes:.2f}")

# Optional: If you’d like to see a confusion matrix for each model
# from sklearn.metrics import confusion_matrix