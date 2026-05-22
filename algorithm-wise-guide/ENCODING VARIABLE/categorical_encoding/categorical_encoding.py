import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_categorical_frame():
    return pd.DataFrame(
        {
            "priority": ["low", "medium", "high", "medium"],
            "city": ["Delhi", "Pune", "Delhi", "Jaipur"],
            "channel": ["app", "web", "store", "web"],
        }
    )


def encode_categories(df):
    ordinal = OrdinalEncoder(categories=[["low", "medium", "high"]])
    ordinal_values = ordinal.fit_transform(df[["priority"]])
    ordinal_df = pd.DataFrame(ordinal_values, columns=["priority_encoded"], index=df.index)

    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    one_hot_values = one_hot.fit_transform(df[["city", "channel"]])
    one_hot_df = pd.DataFrame(
        one_hot_values,
        columns=one_hot.get_feature_names_out(["city", "channel"]),
        index=df.index,
    )

    return ordinal_df, one_hot_df


if __name__ == "__main__":
    frame = build_categorical_frame()
    ordinal_df, one_hot_df = encode_categories(frame)

    print("Original data:")
    print(frame)
    print()

    print("Ordinal encoding:")
    print(ordinal_df)
    print()

    print("One-hot encoding:")
    print(one_hot_df)

