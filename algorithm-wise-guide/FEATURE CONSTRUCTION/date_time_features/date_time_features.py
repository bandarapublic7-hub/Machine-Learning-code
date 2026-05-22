import pandas as pd


df = pd.DataFrame(
    {
        "order_time": pd.to_datetime(
            ["2026-01-03 10:15", "2026-01-04 21:40", "2026-01-06 08:05"]
        )
    }
)

df["month"] = df["order_time"].dt.month
df["weekday"] = df["order_time"].dt.day_name()
df["hour"] = df["order_time"].dt.hour
df["is_weekend"] = (df["order_time"].dt.dayofweek >= 5).astype(int)

print(df)
