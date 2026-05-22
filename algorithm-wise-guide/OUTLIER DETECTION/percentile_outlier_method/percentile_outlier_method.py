import pandas as pd


df = pd.DataFrame({"salary": [42000, 45000, 47000, 51000, 53000, 180000]})

lower = df["salary"].quantile(0.05)
upper = df["salary"].quantile(0.95)

capped = df.copy()
capped["salary"] = capped["salary"].clip(lower=lower, upper=upper)

print(capped)
