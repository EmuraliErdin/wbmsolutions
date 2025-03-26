## Exercise 3 (10 minutes): Regression Trees
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic dataset with multiple features
np.random.seed(42)
num_samples = 30
X = np.random.rand(num_samples, 3) * 10  # e.g., three numeric features

# Let's define a "true" relationship for the target:
# Target = 2*Feature1 + 0.5*Feature2^2 - 3*Feature3 + noise
true_y = 2 * X[:, 0] + 0.5 * (X[:, 1]**2) - 3 * X[:, 2]
noise = np.random.normal(0, 5, size=num_samples)  # Add some noise
y = true_y + noise

# Convert to a pandas DataFrame for familiarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
df["Target"] = y

# 2. Separate features and target
X = df[["Feature1", "Feature2", "Feature3"]].values
y = df["Target"].values

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Create and train the Decision Tree Regressor
#    You can tune hyperparameters like max_depth to control overfitting
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y)

# 5. Evaluate on the test set
new_feature = pd.DataFrame({
    'Feature1': [4],
    'Feature2':  [2000],
    'Feature3': [1]
})

predicted_feature = tree_reg.predict(new_feature)
print("Predicted price for new feature: ", predicted_feature)

# Optional: Inspect feature importances

# Optional: You could visualize the tree with:
# from sklearn.tree import export_graphviz