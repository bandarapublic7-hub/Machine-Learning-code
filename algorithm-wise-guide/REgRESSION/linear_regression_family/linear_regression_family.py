import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def build_regression_data():
    X, y = make_regression(
        n_samples=200,
        n_features=3,
        noise=18,
        random_state=42,
    )
    frame = pd.DataFrame(X, columns=["experience", "hours_studied", "projects"])
    target = pd.Series(y, name="salary")
    return frame, target


def fit_models(X, y):
    simple_model = LinearRegression().fit(X[["experience"]], y)
    multiple_model = LinearRegression().fit(X, y)

    polynomial_model = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )
    polynomial_model.fit(X[["experience"]], y)

    results = {
        "simple_r2": r2_score(y, simple_model.predict(X[["experience"]])),
        "multiple_r2": r2_score(y, multiple_model.predict(X)),
        "polynomial_r2": r2_score(y, polynomial_model.predict(X[["experience"]])),
        "simple_coef": simple_model.coef_[0],
        "multiple_coef": multiple_model.coef_,
    }
    return results


if __name__ == "__main__":
    X, y = build_regression_data()
    results = fit_models(X, y)

    print("Model comparison:")
    for key, value in results.items():
        print(f"{key}: {value}")

