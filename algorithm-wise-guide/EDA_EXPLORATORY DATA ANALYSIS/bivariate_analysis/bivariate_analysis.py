import pandas as pd


df = pd.DataFrame(
    {
        "city": ["Delhi", "Pune", "Delhi", "Jaipur", "Pune", "Delhi"],
        "monthly_spend": [1200, 1600, 2100, 1800, 1900, 2400],
        "subscribed": [0, 1, 1, 0, 1, 1],
    }
)

print("Average spend by city:")
print(df.groupby("city")["monthly_spend"].mean())
print()
print("City vs subscribed:")
print(pd.crosstab(df["city"], df["subscribed"]))
