import numpy as np
import pandas as pd


def build_salary_frame():
    return pd.DataFrame(
        {
            "employee": ["A", "B", "C", "D", "E", "F", "G"],
            "salary": [45000, 47000, 49000, 52000, 51000, 53000, 175000],
        }
    )


def zscore_filter(df, column, threshold=3.0):
    z_scores = ((df[column] - df[column].mean()) / df[column].std(ddof=0)).abs()
    return df[z_scores <= threshold]


def iqr_filter(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[df[column].between(lower, upper)]


def percentile_cap(df, column, lower_pct=0.05, upper_pct=0.95):
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    capped = df.copy()
    capped[column] = capped[column].clip(lower=lower, upper=upper)
    return capped


if __name__ == "__main__":
    salary_df = build_salary_frame()
    print("Original data:")
    print(salary_df)
    print()

    print("Z-score filtered:")
    print(zscore_filter(salary_df, "salary"))
    print()

    print("IQR filtered:")
    print(iqr_filter(salary_df, "salary"))
    print()

    print("Percentile capped:")
    print(percentile_cap(salary_df, "salary"))

