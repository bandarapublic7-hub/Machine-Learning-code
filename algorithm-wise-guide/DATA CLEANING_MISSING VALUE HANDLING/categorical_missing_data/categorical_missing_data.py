import pandas as pd
from sklearn.impute import SimpleImputer


df = pd.DataFrame({"city": ["Delhi", None, "Pune", "Delhi", None]})

freq_result = SimpleImputer(strategy="most_frequent").fit_transform(df)
missing_result = SimpleImputer(strategy="constant", fill_value="Missing").fit_transform(df)

print("Most frequent:", freq_result.ravel())
print("Missing label:", missing_result.ravel())
