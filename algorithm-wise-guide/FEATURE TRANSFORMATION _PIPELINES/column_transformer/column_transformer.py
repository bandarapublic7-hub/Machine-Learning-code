import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


df = pd.DataFrame(
    {
        "age": [21, 25, None, 31],
        "fever": [101, 102, 100, None],
        "city": ["Delhi", "Pune", "Delhi", "Jaipur"],
    }
)

transformer = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age"]),
        ("impute", SimpleImputer(strategy="most_frequent"), ["fever"]),
        ("cat", OneHotEncoder(sparse_output=False), ["city"]),
    ]
).set_output(transform="pandas")

result = transformer.fit_transform(df)
print(result)

