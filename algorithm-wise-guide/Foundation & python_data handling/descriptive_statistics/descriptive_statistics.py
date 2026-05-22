import pandas as pd


df = pd.DataFrame(
    {
        "age": [21, 25, 31, 28, 35],
        "income": [25000, 32000, 54000, 42000, 61000],
        "score": [62, 75, 88, 72, 91],
    }
)

print(df.describe().round(2))
