import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


X, y = make_regression(n_samples=80, n_features=1, noise=12, random_state=42)
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

print("Slope:", round(model.coef_[0], 3))
print("Intercept:", round(model.intercept_, 3))
print("R2 score:", round(r2_score(y, predictions), 3))

preview = pd.DataFrame({"x": X.ravel(), "y": y, "pred": predictions}).head()
print()
print(preview)

