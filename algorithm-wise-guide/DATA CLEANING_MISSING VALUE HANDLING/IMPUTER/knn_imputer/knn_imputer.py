import pandas as pd
from sklearn.impute import KNNImputer


df = pd.DataFrame(
    {
        "maths": [78, 82, 91, None, 75],
        "science": [80, None, 89, 84, 73],
        "english": [74, 79, 92, 81, 70],
    }
)

imputer = KNNImputer(n_neighbors=2)
filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("Original data:")
print(df)
print()
print("After KNN imputation:")
print(filled.round(2))

