"""Fuzzy C-Means clustering implemented with NumPy."""

from __future__ import annotations

import numpy as np


class FuzzyCMeans:
    """Fuzzy C-Means clustering.

    Unlike K-Means, every sample keeps a membership degree for every cluster.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-5,
        random_state: int | None = 42,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if m <= 1.0:
            raise ValueError("m must be greater than 1")
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2D array")
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed the number of samples")

        rng = np.random.default_rng(self.random_state)
        membership = rng.random((len(X), self.n_clusters))
        membership /= membership.sum(axis=1, keepdims=True)
        self.objective_history_: list[float] = []

        for iteration in range(1, self.max_iter + 1):
            powered = membership**self.m
            denominator = powered.sum(axis=0)[:, None]
            centers = (powered.T @ X) / denominator

            distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            distances = np.maximum(distances, 1e-12)
            objective = float(np.sum(powered * (distances**2)))
            self.objective_history_.append(objective)

            exponent = 2.0 / (self.m - 1.0)
            ratios = distances[:, :, None] / distances[:, None, :]
            new_membership = 1.0 / np.sum(ratios**exponent, axis=2)

            max_change = float(np.max(np.abs(new_membership - membership)))
            membership = new_membership
            if max_change <= self.tol:
                break

        self.cluster_centers_ = centers
        self.membership_ = membership
        self.labels_ = np.argmax(membership, axis=1)
        self.n_iter_ = iteration
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_.copy()

    def predict_membership(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        distances = np.linalg.norm(
            X[:, None, :] - self.cluster_centers_[None, :, :], axis=2
        )
        distances = np.maximum(distances, 1e-12)
        exponent = 2.0 / (self.m - 1.0)
        ratios = distances[:, :, None] / distances[:, None, :]
        return 1.0 / np.sum(ratios**exponent, axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_membership(X), axis=1)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Call fit before predict")
