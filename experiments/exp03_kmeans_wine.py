"""Evaluate K-Means on the Wine dataset with label-invariant metrics."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from prlib.kmeans import KMeans
from prlib.metrics import clustering_accuracy


RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def main() -> None:
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    start = time.perf_counter()
    ours = KMeans(n_clusters=3, random_state=42).fit(X)
    ours_time = time.perf_counter() - start

    start = time.perf_counter()
    reference = SklearnKMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
    reference_time = time.perf_counter() - start

    print(f"ours_ari={adjusted_rand_score(y, ours.labels_):.4f}")
    print(f"ours_nmi={normalized_mutual_info_score(y, ours.labels_):.4f}")
    print(f"ours_matched_accuracy={clustering_accuracy(y, ours.labels_):.4f}")
    print(f"sklearn_ari={adjusted_rand_score(y, reference.labels_):.4f}")
    print(f"ours_time_seconds={ours_time:.6f}")
    print(f"sklearn_time_seconds={reference_time:.6f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    projected = PCA(n_components=2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=ours.labels_, s=32)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means Clusters on Wine (PCA projection)")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "kmeans_wine_clusters.png", dpi=160)
    plt.close(fig)

    ks = range(1, 9)
    inertias = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(ks), inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Curve — Wine")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "kmeans_wine_elbow_curve.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
