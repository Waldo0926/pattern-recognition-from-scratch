"""Benchmark Fisher LDA on the Wine dataset."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prlib.lda import FisherLDA


RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def main() -> None:
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    start = time.perf_counter()
    ours = FisherLDA(n_components=2).fit(X_train, y_train)
    ours_accuracy = ours.score(X_test, y_test)
    ours_time = time.perf_counter() - start

    start = time.perf_counter()
    reference = LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train)
    reference_accuracy = reference.score(X_test, y_test)
    reference_time = time.perf_counter() - start

    print(f"ours_accuracy={ours_accuracy:.4f}")
    print(f"sklearn_accuracy={reference_accuracy:.4f}")
    print(f"ours_time_seconds={ours_time:.6f}")
    print(f"sklearn_time_seconds={reference_time:.6f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    full_scaled = scaler.transform(X)
    projection = ours.transform(full_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=y, s=34)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_title("Fisher LDA Projection — Wine")
    fig.colorbar(scatter, ax=ax, label="Wine class")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fisher_lda_wine_projection.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
