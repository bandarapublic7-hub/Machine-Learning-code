import pandas as pd


df = pd.DataFrame({"salary": [42000, 45000, 47000, 51000, 53000, 180000]})
z = ((df["salary"] - df["salary"].mean()) / df["salary"].std(ddof=0)).abs()
result = df[z <= 3]

print(result)
