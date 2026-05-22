from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


X, y = make_regression(n_samples=120, n_features=1, noise=18, random_state=42)

model = Pipeline(
    steps=[
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("reg", LinearRegression()),
    ]
)

model.fit(X, y)
predictions = model.predict(X)

print("R2 score:", round(r2_score(y, predictions), 3))

