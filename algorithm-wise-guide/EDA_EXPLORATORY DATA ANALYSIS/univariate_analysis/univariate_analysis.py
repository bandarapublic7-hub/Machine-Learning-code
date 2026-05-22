import pandas as pd


df = pd.DataFrame(
    {
        "city": ["Delhi", "Pune", "Delhi", "Jaipur", "Pune", "Delhi"],
        "age": [21, 25, 31, 28, 35, 29],
    }
)

print("City counts:")
print(df["city"].value_counts())
print()
print("Age summary:")
print(df["age"].describe())
