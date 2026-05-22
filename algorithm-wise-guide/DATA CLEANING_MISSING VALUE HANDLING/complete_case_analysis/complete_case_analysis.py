import pandas as pd


df = pd.DataFrame(
    {
        "age": [22, None, 31, 28],
        "income": [25000, 30000, None, 42000],
    }
)

clean_df = df.dropna()
print(clean_df)
