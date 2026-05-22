import pandas as pd


df = pd.DataFrame({"salary": [42000, 45000, 47000, 51000, 53000, 180000]})

q1 = df["salary"].quantile(0.25)
q3 = df["salary"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

result = df[df["salary"].between(lower, upper)]
print(result)
