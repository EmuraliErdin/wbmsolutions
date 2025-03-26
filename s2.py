## Exercise 2 (10 minutes): Polynomial Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic non-linear dataset
np.random.seed(42)
num_samples = 30

# Single feature for clarity (e.g., 'sqft' or just X)
X = np.linspace(0, 10, num_samples).reshape(-1, 1)

# True relationship: y = 2 * X^2 - 3 * X + noise
y_true = 2 * (X**2) - 3 * X
noise = np.random.normal(0, 3, size=num_samples)
y = (y_true + noise).flatten()

# Convert to DataFrame
df = pd.DataFrame({"Feature": X.flatten(), "Target": y})

# 2. Separate features and target
X = df[["Feature"]].values
y = df["Target"].values

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Transform features to polynomial (degree=2 or 3 for illustration)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 5. Create and train a Linear Regression model on the polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 7. Optional: Plot to visualize the fit
y_pred = model.predict(X_test_poly)
#    Generate a smooth curve for plotting