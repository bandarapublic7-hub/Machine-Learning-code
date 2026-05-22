import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def compare_models():
    X, y = make_regression(
        n_samples=300,
        n_features=12,
        n_informative=5,
        noise=25,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    models = {
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.1),
        "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5),
    }

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        coefficients = pipeline.named_steps["model"].coef_

        print(name)
        print("  r2:", round(r2_score(y_test, predictions), 3))
        print("  coefficient_norm:", round(np.linalg.norm(coefficients), 3))
        print()


if __name__ == "__main__":
    compare_models()

