from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


X, y = make_regression(n_samples=100, n_features=3, noise=20, random_state=42)
model = LinearRegression().fit(X, y)
pred = model.predict(X)

print("MAE:", round(mean_absolute_error(y, pred), 3))
print("MSE:", round(mean_squared_error(y, pred), 3))
print("RMSE:", round(mean_squared_error(y, pred, squared=False), 3))
print("R2:", round(r2_score(y, pred), 3))

