"""Evaluate Fuzzy C-Means on the Wine dataset."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from prlib.fuzzy_cmeans import FuzzyCMeans
from prlib.metrics import clustering_accuracy


RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def main() -> None:
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    start = time.perf_counter()
    model = FuzzyCMeans(n_clusters=3, m=2.0, random_state=42).fit(X)
    elapsed = time.perf_counter() - start

    print(f"ari={adjusted_rand_score(y, model.labels_):.4f}")
    print(f"nmi={normalized_mutual_info_score(y, model.labels_):.4f}")
    print(f"matched_accuracy={clustering_accuracy(y, model.labels_):.4f}")
    print(f"time_seconds={elapsed:.6f}")
    print(f"iterations={model.n_iter_}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    projected = PCA(n_components=2, random_state=42).fit_transform(X)
    confidence = model.membership_.max(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        projected[:, 0], projected[:, 1], c=model.labels_, s=25 + 55 * confidence
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Fuzzy C-Means on Wine — size reflects membership confidence")
    fig.colorbar(scatter, ax=ax, label="Hard label from max membership")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fuzzy_cmeans_wine_clusters.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(model.objective_history_) + 1), model.objective_history_)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("FCM objective")
    ax.set_title("Fuzzy C-Means Objective Convergence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fuzzy_cmeans_objective.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
