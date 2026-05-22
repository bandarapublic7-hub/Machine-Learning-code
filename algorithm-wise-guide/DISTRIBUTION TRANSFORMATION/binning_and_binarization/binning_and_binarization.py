import pandas as pd
from sklearn.preprocessing import Binarizer, KBinsDiscretizer


df = pd.DataFrame({"score": [32, 48, 59, 71, 89]})

binner = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
df["score_bucket"] = binner.fit_transform(df[["score"]]).ravel().astype(int)

binary = Binarizer(threshold=50)
df["passed"] = binary.fit_transform(df[["score"]]).ravel().astype(int)

print(df)
