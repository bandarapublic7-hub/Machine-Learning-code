import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_titanic_like_data():
    return pd.DataFrame(
        {
            "age": [22, 38, None, 35, 28, None, 19, 42],
            "fare": [7.25, 71.28, 8.05, 53.10, 13.00, 26.55, 7.89, 52.00],
            "family_size": [1, 2, 1, 2, 0, 1, 0, 3],
            "sex": ["male", "female", "female", "female", "male", "male", "female", "male"],
            "embarked": ["S", "C", "S", "S", "Q", "S", None, "C"],
            "survived": [0, 1, 1, 1, 0, 0, 1, 1],
        }
    )


def build_pipeline():
    numeric_features = ["age", "fare", "family_size"]
    categorical_features = ["sex", "embarked"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    ).set_output(transform="pandas")

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    return model


if __name__ == "__main__":
    df = build_titanic_like_data()
    X = df.drop(columns="survived")
    y = df["survived"]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    transformed = pipeline.named_steps["preprocess"].transform(X)

    print("Prepared feature matrix:")
    print(transformed.head())
    print()

    print("Predicted survival probabilities:")
    print(pipeline.predict_proba(X)[:3])

