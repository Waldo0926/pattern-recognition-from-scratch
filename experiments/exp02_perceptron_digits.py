"""Benchmark the multiclass perceptron on the digits dataset."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prlib.perceptron import MulticlassPerceptron


RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def main() -> None:
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    start = time.perf_counter()
    ours = MulticlassPerceptron(
        learning_rate=0.1, max_epochs=60, random_state=42
    ).fit(X_train, y_train)
    ours_accuracy = ours.score(X_test, y_test)
    ours_time = time.perf_counter() - start

    start = time.perf_counter()
    reference = Perceptron(max_iter=60, tol=None, random_state=42).fit(X_train, y_train)
    reference_accuracy = reference.score(X_test, y_test)
    reference_time = time.perf_counter() - start

    print(f"ours_accuracy={ours_accuracy:.4f}")
    print(f"sklearn_accuracy={reference_accuracy:.4f}")
    print(f"ours_time_seconds={ours_time:.6f}")
    print(f"sklearn_time_seconds={reference_time:.6f}")
    print(f"epochs={ours.n_iter_}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, ours.n_iter_ + 1), ours.mistakes_per_epoch_, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training mistakes")
    ax.set_title("Multiclass Perceptron Training Curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "perceptron_digits_training_curve.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
