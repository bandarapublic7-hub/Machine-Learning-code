from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X, y = make_regression(
    n_samples=180,
    n_features=8,
    n_informative=4,
    noise=22,
    random_state=42,
)

model = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ]
)

model.fit(X, y)
predictions = model.predict(X)
print("R2 score:", round(r2_score(y, predictions), 3))

