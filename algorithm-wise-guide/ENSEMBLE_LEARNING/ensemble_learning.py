from sklearn.datasets import load_wine
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def compare_ensembles():
    X, y = load_wine(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=150,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(name, round(accuracy_score(y_test, predictions), 3))


if __name__ == "__main__":
    compare_ensembles()

