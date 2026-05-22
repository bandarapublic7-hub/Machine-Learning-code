import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.DataFrame(
    {
        "age": [21, 25, 32, 40, 46],
        "salary": [25000, 32000, 54000, 72000, 88000],
    }
)

scaler = StandardScaler().set_output(transform="pandas")
scaled_df = scaler.fit_transform(df)

print("Original data:")
print(df)
print()
print("Standardized data:")
print(scaled_df.round(3))

