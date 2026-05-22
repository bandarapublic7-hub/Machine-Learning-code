import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


df = pd.DataFrame({"income": [12000, 18000, 25000, 40000]})

log_transformer = FunctionTransformer(np.log1p).set_output(transform="pandas")
result = log_transformer.fit_transform(df)

print(result.round(3))
