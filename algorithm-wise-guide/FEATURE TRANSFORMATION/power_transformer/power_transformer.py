import pandas as pd
from sklearn.preprocessing import PowerTransformer


df = pd.DataFrame({"income": [12000, 15000, 18000, 40000, 85000]})

transformer = PowerTransformer().set_output(transform="pandas")
result = transformer.fit_transform(df)

print(result.round(3))
