from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import pandas as pd


df = pd.DataFrame(
    {
        "age": [22, 25, None, 31, 40],
        "income": [25000, 32000, 41000, None, 62000],
        "score": [68, 72, 79, 82, None],
    }
)

imputer = IterativeImputer(random_state=42, max_iter=10)
filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print(filled.round(2))

