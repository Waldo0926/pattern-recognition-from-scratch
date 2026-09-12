"""K-Means clustering with k-means++ initialization."""

from __future__ import annotations

import numpy as np


class KMeans:
    """K-Means clustering implemented with NumPy."""

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = 42,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2D array")
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed the number of samples")

        rng = np.random.default_rng(self.random_state)
        centers = self._init_kmeans_plus_plus(X, rng)

        for iteration in range(1, self.max_iter + 1):
            distances_sq = self._squared_distances(X, centers)
            labels = np.argmin(distances_sq, axis=1)
            new_centers = centers.copy()
            for cluster in range(self.n_clusters):
                members = X[labels == cluster]
                if len(members) == 0:
                    farthest = int(np.argmax(np.min(distances_sq, axis=1)))
                    new_centers[cluster] = X[farthest]
                else:
                    new_centers[cluster] = members.mean(axis=0)

            center_shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if center_shift <= self.tol:
                break

        distances_sq = self._squared_distances(X, centers)
        labels = np.argmin(distances_sq, axis=1)
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(np.sum(distances_sq[np.arange(len(X)), labels]))
        self.n_iter_ = iteration
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible shape")
        return np.argmin(self._squared_distances(X, self.cluster_centers_), axis=1)

    def _init_kmeans_plus_plus(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        first = int(rng.integers(len(X)))
        centers[0] = X[first]
        closest_sq = self._squared_distances(X, centers[:1]).ravel()

        for idx in range(1, self.n_clusters):
            total = float(closest_sq.sum())
            if total <= 0:
                chosen = int(rng.integers(len(X)))
            else:
                probabilities = closest_sq / total
                chosen = int(rng.choice(len(X), p=probabilities))
            centers[idx] = X[chosen]
            candidate_sq = self._squared_distances(X, centers[idx : idx + 1]).ravel()
            closest_sq = np.minimum(closest_sq, candidate_sq)
        return centers

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Call fit before predict")
