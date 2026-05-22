from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
import pandas as pd


def build_missing_frame():
    return pd.DataFrame(
        {
            "age": [22, None, 29, 31, None, 40],
            "income": [25000, 42000, None, 62000, 58000, None],
            "score": [620, 650, 640, None, 700, 680],
            "city": ["Delhi", "Pune", None, "Jaipur", "Delhi", None],
        }
    )


def complete_case(df):
    return df.dropna()


def simple_imputation(df):
    numeric_cols = ["age", "income", "score"]
    simple_numeric = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index,
    )

    simple_city = pd.DataFrame(
        SimpleImputer(strategy="constant", fill_value="Missing").fit_transform(df[["city"]]),
        columns=["city"],
        index=df.index,
    )

    result = pd.concat([simple_numeric, simple_city], axis=1)
    return result


def indicator_features(df):
    indicator = MissingIndicator(features="all")
    values = indicator.fit_transform(df)
    columns = [f"{column}_was_missing" for column in df.columns]
    return pd.DataFrame(values, columns=columns, index=df.index)


def advanced_imputation(df):
    numeric_cols = ["age", "income", "score"]

    knn_df = pd.DataFrame(
        KNNImputer(n_neighbors=2).fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index,
    )

    iterative_df = pd.DataFrame(
        IterativeImputer(random_state=42, max_iter=10).fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index,
    )

    return knn_df, iterative_df


if __name__ == "__main__":
    missing_df = build_missing_frame()
    print("Original data:")
    print(missing_df)
    print()

    print("Complete case analysis:")
    print(complete_case(missing_df))
    print()

    print("Simple imputation:")
    print(simple_imputation(missing_df))
    print()

    print("Missing indicators:")
    print(indicator_features(missing_df))
    print()

    knn_df, iterative_df = advanced_imputation(missing_df)
    print("KNN imputation:")
    print(knn_df)
    print()

    print("Iterative imputation:")
    print(iterative_df)

