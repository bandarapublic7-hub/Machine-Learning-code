import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


df = pd.DataFrame(
    {
        "size": ["small", "medium", "large", "medium", "small"]
    }
)

encoder = OrdinalEncoder(categories=[["small", "medium", "large"]])
encoded = encoder.fit_transform(df[["size"]])

result = df.copy()
result["size_encoded"] = encoded.astype(int)

print(result)

