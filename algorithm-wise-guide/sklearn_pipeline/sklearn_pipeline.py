import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


df = pd.DataFrame(
    {
        "age": [22, 38, None, 28, 19, 42],
        "fare": [7.25, 71.28, 8.05, 13.00, 7.89, 52.00],
        "sex": ["male", "female", "female", "male", "female", "male"],
        "embarked": ["S", "C", "S", "Q", None, "C"],
        "survived": [0, 1, 1, 0, 1, 1],
    }
)

X = df.drop(columns="survived")
y = df["survived"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            ["age", "fare"],
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            ["sex", "embarked"],
        ),
    ]
)

pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

pipe.fit(X, y)
print(pipe.predict_proba(X)[:3])

