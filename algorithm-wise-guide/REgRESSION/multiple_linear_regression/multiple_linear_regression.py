from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


X, y = make_regression(
    n_samples=120,
    n_features=3,
    n_informative=3,
    noise=15,
    random_state=42,
)

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

print("Coefficients:", model.coef_.round(3))
print("Intercept:", round(model.intercept_, 3))
print("R2 score:", round(r2_score(y, predictions), 3))

