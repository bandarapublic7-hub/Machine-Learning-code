import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def build_cluster_data():
    X, _ = make_blobs(
        n_samples=120,
        centers=3,
        cluster_std=1.1,
        random_state=42,
    )
    return pd.DataFrame(X, columns=["hours_studied", "practice_score"])


def assign_points_to_centroids(points, centroids):
    distances = np.sqrt(((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    return distances.argmin(axis=1)


def run_kmeans(df):
    model = KMeans(n_clusters=3, n_init="auto", random_state=42)
    labels = model.fit_predict(df)
    clustered = df.copy()
    clustered["cluster"] = labels
    return model, clustered


if __name__ == "__main__":
    cluster_df = build_cluster_data()
    model, clustered = run_kmeans(cluster_df)

    print("Cluster centers:")
    print(model.cluster_centers_)
    print()

    print("Cluster sizes:")
    print(clustered["cluster"].value_counts().sort_index())
    print()

    labels_again = assign_points_to_centroids(
        cluster_df.to_numpy(),
        model.cluster_centers_,
    )
    print("Nearest-centroid assignment matches sklearn style:")
    print(labels_again[:10])

