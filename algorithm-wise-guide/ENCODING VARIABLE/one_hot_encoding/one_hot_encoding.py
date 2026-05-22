import pandas as pd
from sklearn.preprocessing import OneHotEncoder


df = pd.DataFrame(
    {
        "city": ["Delhi", "Pune", "Jaipur", "Delhi"],
        "channel": ["app", "web", "store", "web"],
    }
)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = encoder.fit_transform(df[["city", "channel"]])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["city", "channel"]),
)

print(encoded_df)

