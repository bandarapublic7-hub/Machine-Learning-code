from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def regression_metrics_demo():
    X, y = make_regression(n_samples=200, n_features=4, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Regression metrics")
    print("  mae:", round(mean_absolute_error(y_test, predictions), 3))
    print("  mse:", round(mean_squared_error(y_test, predictions), 3))
    print("  rmse:", round(mean_squared_error(y_test, predictions, squared=False), 3))
    print("  r2:", round(r2_score(y_test, predictions), 3))
    print()


def classification_metrics_demo():
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Classification metrics")
    print("  accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("  precision:", round(precision_score(y_test, predictions), 3))
    print("  recall:", round(recall_score(y_test, predictions), 3))
    print("  f1:", round(f1_score(y_test, predictions), 3))
    print("  roc_auc:", round(roc_auc_score(y_test, probabilities), 3))


if __name__ == "__main__":
    regression_metrics_demo()
    classification_metrics_demo()

