import pandas as pd
from sklearn.impute import SimpleImputer


df = pd.DataFrame({"age": [22, None, 31, 28, None]})

mean_result = SimpleImputer(strategy="mean").fit_transform(df)
median_result = SimpleImputer(strategy="median").fit_transform(df)
constant_result = SimpleImputer(strategy="constant", fill_value=0).fit_transform(df)

print("Mean:", mean_result.ravel())
print("Median:", median_result.ravel())
print("Constant:", constant_result.ravel())
