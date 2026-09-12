"""Benchmark the NumPy Gaussian Naive Bayes implementation on digits."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from prlib.naive_bayes import GaussianNaiveBayes


RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def main() -> None:
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    start = time.perf_counter()
    ours = GaussianNaiveBayes().fit(X_train, y_train)
    ours_accuracy = ours.score(X_test, y_test)
    ours_time = time.perf_counter() - start

    start = time.perf_counter()
    reference = GaussianNB().fit(X_train, y_train)
    reference_accuracy = reference.score(X_test, y_test)
    reference_time = time.perf_counter() - start

    print(f"ours_accuracy={ours_accuracy:.4f}")
    print(f"sklearn_accuracy={reference_accuracy:.4f}")
    print(f"ours_time_seconds={ours_time:.6f}")
    print(f"sklearn_time_seconds={reference_time:.6f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = ours.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, colorbar=False)
    ax.set_title("Gaussian Naive Bayes — Digits Confusion Matrix")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "naive_bayes_digits_confusion_matrix.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
