from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_base_models():
    logistic_model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    knn_model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]
    )

    return logistic_model, knn_model


def run_examples():
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    logistic_model, knn_model = build_base_models()

    logistic_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    blended_probabilities = (
        logistic_model.predict_proba(X_test)[:, 1]
        + knn_model.predict_proba(X_test)[:, 1]
    ) / 2
    blended_predictions = (blended_probabilities >= 0.5).astype(int)

    stack_model = StackingClassifier(
        estimators=[
            ("logistic", build_base_models()[0]),
            ("knn", build_base_models()[1]),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
    )
    stack_model.fit(X_train, y_train)
    stack_predictions = stack_model.predict(X_test)

    print("Blending accuracy:", round(accuracy_score(y_test, blended_predictions), 3))
    print("Stacking accuracy:", round(accuracy_score(y_test, stack_predictions), 3))


if __name__ == "__main__":
    run_examples()

