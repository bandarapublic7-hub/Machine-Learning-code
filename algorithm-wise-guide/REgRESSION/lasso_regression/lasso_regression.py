from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
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
        ("lasso", Lasso(alpha=0.1)),
    ]
)

model.fit(X, y)
coef = model.named_steps["lasso"].coef_

print("Coefficients:", coef.round(3))
print("How many became zero:", int((coef == 0).sum()))

