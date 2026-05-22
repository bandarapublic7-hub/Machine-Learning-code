from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
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
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
    ]
)

model.fit(X, y)
print(model.named_steps["elastic"].coef_.round(3))

