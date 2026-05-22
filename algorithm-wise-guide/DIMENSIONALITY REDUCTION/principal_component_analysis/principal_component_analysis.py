import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_pca():
    X, y = load_wine(as_frame=True, return_X_y=True)

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=2)),
        ]
    )

    reduced = pipeline.fit_transform(X)
    reduced_df = pd.DataFrame(reduced, columns=["pc1", "pc2"])
    reduced_df["target"] = y.to_numpy()

    explained = pipeline.named_steps["pca"].explained_variance_ratio_
    return reduced_df, explained


if __name__ == "__main__":
    reduced_df, explained = run_pca()
    print("Reduced data:")
    print(reduced_df.head())
    print()

    print("Explained variance ratio:")
    print(explained)
    print(f"Total kept variance: {explained.sum():.3f}")

