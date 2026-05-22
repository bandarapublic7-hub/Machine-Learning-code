import pandas as pd
from sklearn.impute import MissingIndicator


df = pd.DataFrame(
    {
        "age": [22, None, 31, 28],
        "income": [25000, 30000, None, 42000],
    }
)

indicator = MissingIndicator(features="all")
flags = indicator.fit_transform(df)
flag_df = pd.DataFrame(flags, columns=["age_missing", "income_missing"])

print(flag_df)
