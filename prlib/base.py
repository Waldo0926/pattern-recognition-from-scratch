"""Shared estimator helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    """Minimal sklearn-like classifier interface."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """Fit the estimator."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean classification accuracy."""
        predictions = self.predict(X)
        return float(np.mean(predictions == np.asarray(y)))
