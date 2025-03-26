## Exercise 1 (10 minutes): Baseline Linear Regression
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a small synthetic dataset
#    For simplicity, let's assume we have only numeric features.
np.random.seed(42)  # For reproducibility
num_samples = 20
X = np.random.rand(num_samples, 2) * 100  # e.g., two numeric features
# True relationship (just as an example):
# price = 3.0*(feature1) + 2.0*(feature2) + some_noise
true_coeffs = np.array([3.0, 2.0])
y = X.dot(true_coeffs) + np.random.normal(0, 10, size=num_samples)  # Add noise

# Convert to a pandas DataFrame for familiarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Price"] = y

# 2. Separate features (X) and target (y)
X = df[['Feature1', 'Feature2']]
y = df['Price']

# 3. Split the dataset into training and test sets
X_encoded = pd.get_dummies(X, columns=['Feature1'], drop_first=True)


# 4. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_encoded, y)

# 5. Use the model to predict on the test set
new_feature = pd.DataFrame({
    'Feature1': [16],
    'Feature2': [3]
})

new_feature_encoded = pd.get_dummies(new_feature, columns=['Feature1'], drop_first=True)
predicted_feature1 = model.predict(new_feature_encoded)

print("Predicted price for the new house:", predicted_feature1[0])
