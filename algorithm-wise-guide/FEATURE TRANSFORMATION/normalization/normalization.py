import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.DataFrame(
    {
        "marks": [35, 48, 62, 78, 91],
        "hours_studied": [1, 2, 4, 5, 7],
    }
)

scaler = MinMaxScaler().set_output(transform="pandas")
normalized_df = scaler.fit_transform(df)

print("Original data:")
print(df)
print()
print("Normalized data:")
print(normalized_df.round(3))

