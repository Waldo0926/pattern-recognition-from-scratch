"""Multiclass perceptron implemented with NumPy."""

from __future__ import annotations

import numpy as np

from .base import BaseClassifier


class MulticlassPerceptron(BaseClassifier):
    """Online multiclass perceptron using the argmax update rule."""

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_epochs: int = 50,
        shuffle: bool = True,
        fit_intercept: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.shuffle = bool(shuffle)
        self.fit_intercept = bool(fit_intercept)
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassPerceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1D array aligned with X")
        if len(X) == 0:
            raise ValueError("X must contain at least one sample")

        self.classes_, encoded = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.coef_ = np.zeros((n_classes, n_features), dtype=float)
        self.intercept_ = np.zeros(n_classes, dtype=float)
        self.mistakes_per_epoch_: list[int] = []
        rng = np.random.default_rng(self.random_state)

        indices = np.arange(len(X))
        for _ in range(self.max_epochs):
            if self.shuffle:
                rng.shuffle(indices)
            mistakes = 0
            for idx in indices:
                scores = self.coef_ @ X[idx]
                if self.fit_intercept:
                    scores = scores + self.intercept_
                predicted = int(np.argmax(scores))
                target = int(encoded[idx])
                if predicted != target:
                    update = self.learning_rate * X[idx]
                    self.coef_[target] += update
                    self.coef_[predicted] -= update
                    if self.fit_intercept:
                        self.intercept_[target] += self.learning_rate
                        self.intercept_[predicted] -= self.learning_rate
                    mistakes += 1
            self.mistakes_per_epoch_.append(mistakes)
            if mistakes == 0:
                break

        self.n_iter_ = len(self.mistakes_per_epoch_)
        self.n_features_in_ = n_features
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible shape")
        scores = X @ self.coef_.T
        if self.fit_intercept:
            scores = scores + self.intercept_[None, :]
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before predict")
