"""Gaussian Naive Bayes implemented with NumPy."""

from __future__ import annotations

import numpy as np

from .base import BaseClassifier


class GaussianNaiveBayes(BaseClassifier):
    """Gaussian Naive Bayes classifier.

    Parameters
    ----------
    var_smoothing:
        Fraction of the largest global feature variance added to every class
        variance. This prevents division by zero for constant features.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        if var_smoothing <= 0:
            raise ValueError("var_smoothing must be positive")
        self.var_smoothing = float(var_smoothing)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1D array aligned with X")
        if len(X) == 0:
            raise ValueError("X must contain at least one sample")

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / counts.sum()
        self.theta_ = np.vstack([X[y == cls].mean(axis=0) for cls in self.classes_])
        raw_var = np.vstack([X[y == cls].var(axis=0) for cls in self.classes_])
        global_variance = np.var(X, axis=0)
        epsilon = self.var_smoothing * max(float(np.max(global_variance)), 1.0)
        self.var_ = raw_var + epsilon
        self.n_features_in_ = X.shape[1]
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible shape")

        log_prior = np.log(self.class_prior_)
        log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_), axis=1)
        diff = X[:, None, :] - self.theta_[None, :, :]
        quadratic = -0.5 * np.sum((diff * diff) / self.var_[None, :, :], axis=2)
        return log_prior[None, :] + log_norm[None, :] + quadratic

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        joint = self._joint_log_likelihood(X)
        max_joint = np.max(joint, axis=1, keepdims=True)
        log_evidence = max_joint + np.log(np.exp(joint - max_joint).sum(axis=1, keepdims=True))
        return joint - log_evidence

    def predict(self, X: np.ndarray) -> np.ndarray:
        joint = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint, axis=1)]

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "classes_"):
            raise RuntimeError("Call fit before predict")
