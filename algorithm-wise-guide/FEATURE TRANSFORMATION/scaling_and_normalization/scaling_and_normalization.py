import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler


def build_numeric_frame():
    return pd.DataFrame(
        {
            "income": [25000, 42000, 51000, 86000],
            "distance_km": [2, 8, 12, 30],
            "monthly_orders": [4, 11, 16, 28],
        }
    )


def run_scalers(df):
    standard = StandardScaler().set_output(transform="pandas")
    minmax = MinMaxScaler().set_output(transform="pandas")
    normalizer = Normalizer()

    standard_df = standard.fit_transform(df)
    minmax_df = minmax.fit_transform(df)

    normalized_values = normalizer.fit_transform(df)
    normalized_df = pd.DataFrame(
        normalized_values,
        columns=df.columns,
        index=df.index,
    )

    return standard_df, minmax_df, normalized_df


if __name__ == "__main__":
    numeric_df = build_numeric_frame()
    standard_df, minmax_df, normalized_df = run_scalers(numeric_df)

    print("Original data:")
    print(numeric_df)
    print()

    print("Standardized data:")
    print(standard_df)
    print()

    print("Min-max scaled data:")
    print(minmax_df)
    print()

    print("Normalized rows:")
    print(normalized_df)

