import numpy as np
import pandas as pd
from sklearn.preprocessing import Binarizer, KBinsDiscretizer, PowerTransformer


def build_feature_frame():
    return pd.DataFrame(
        {
            "monthly_income": [12000, 18000, 26000, 95000, 42000],
            "site_visits": [2, 4, 7, 14, 9],
            "signup_date": pd.to_datetime(
                ["2026-01-02", "2026-01-05", "2026-01-09", "2026-01-10", "2026-01-14"]
            ),
            "city_code": ["DEL-01", "PUN-02", "DEL-03", "JAI-01", "PUN-04"],
        }
    )


def transform_features(df):
    result = df.copy()

    result["log_income"] = np.log1p(result["monthly_income"])

    power_values = PowerTransformer().fit_transform(df[["monthly_income", "site_visits"]])
    result["income_power"] = power_values[:, 0]
    result["visits_power"] = power_values[:, 1]

    binner = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
    result["visit_bucket"] = binner.fit_transform(df[["site_visits"]]).ravel().astype(int)

    binary = Binarizer(threshold=7)
    result["is_high_visit_user"] = binary.fit_transform(df[["site_visits"]]).ravel().astype(int)

    result["signup_month"] = result["signup_date"].dt.month
    result["signup_weekday"] = result["signup_date"].dt.day_name()
    result["is_weekend_signup"] = (result["signup_date"].dt.dayofweek >= 5).astype(int)

    result["city_prefix"] = result["city_code"].str.split("-").str[0]
    result["city_number"] = result["city_code"].str.split("-").str[1].astype(int)
    return result


if __name__ == "__main__":
    frame = build_feature_frame()
    transformed = transform_features(frame)

    print("Original data:")
    print(frame)
    print()

    print("Transformed features:")
    print(transformed)

