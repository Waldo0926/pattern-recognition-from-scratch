"""Fisher Linear Discriminant Analysis implemented with NumPy."""

from __future__ import annotations

import numpy as np

from .base import BaseClassifier


class FisherLDA(BaseClassifier):
    """Fisher LDA projection with nearest-centroid classification.

    The projection maximizes between-class scatter relative to within-class
    scatter. Classification is performed by nearest class centroid in the
    discriminant subspace.
    """

    def __init__(self, n_components: int | None = None, reg: float = 1e-6) -> None:
        if n_components is not None and n_components <= 0:
            raise ValueError("n_components must be positive or None")
        if reg < 0:
            raise ValueError("reg must be non-negative")
        self.n_components = n_components
        self.reg = float(reg)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FisherLDA":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2D array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1D array aligned with X")

        self.classes_ = np.unique(y)
        max_components = min(len(self.classes_) - 1, X.shape[1])
        if max_components < 1:
            raise ValueError("Fisher LDA requires at least two classes")
        n_components = self.n_components or max_components
        if n_components > max_components:
            raise ValueError(f"n_components cannot exceed {max_components}")

        overall_mean = X.mean(axis=0)
        sw = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        sb = np.zeros_like(sw)

        for cls in self.classes_:
            Xc = X[y == cls]
            mean_c = Xc.mean(axis=0)
            centered = Xc - mean_c
            sw += centered.T @ centered
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            sb += len(Xc) * (mean_diff @ mean_diff.T)

        scale = np.trace(sw) / X.shape[1] if np.trace(sw) > 0 else 1.0
        sw_regularized = sw + self.reg * scale * np.eye(X.shape[1])
        matrix = np.linalg.pinv(sw_regularized) @ sb
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        order = np.argsort(eigenvalues.real)[::-1]
        components = eigenvectors[:, order[:n_components]].real
        components /= np.maximum(np.linalg.norm(components, axis=0, keepdims=True), 1e-12)

        self.eigenvalues_ = eigenvalues.real[order[:n_components]]
        self.scalings_ = components
        self.n_features_in_ = X.shape[1]
        projected = X @ self.scalings_
        self.class_centroids_ = np.vstack(
            [projected[y == cls].mean(axis=0) for cls in self.classes_]
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible shape")
        return X @ self.scalings_

    def predict(self, X: np.ndarray) -> np.ndarray:
        projected = self.transform(X)
        diff = projected[:, None, :] - self.class_centroids_[None, :, :]
        distances_sq = np.sum(diff * diff, axis=2)
        return self.classes_[np.argmin(distances_sq, axis=1)]

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "scalings_"):
            raise RuntimeError("Call fit before transform or predict")
