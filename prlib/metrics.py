"""Small evaluation helpers used by the experiments."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("inputs must not be empty")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None
) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    lookup = {label: idx for idx, label in enumerate(labels.tolist())}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(y_true, y_pred, strict=True):
        if truth in lookup and pred in lookup:
            matrix[lookup[truth], lookup[pred]] += 1
    return matrix


def clustering_accuracy(y_true: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Return clustering accuracy after optimal label assignment."""
    y_true = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)
    if y_true.shape != cluster_labels.shape:
        raise ValueError("inputs must have the same shape")
    true_classes, true_encoded = np.unique(y_true, return_inverse=True)
    clusters, cluster_encoded = np.unique(cluster_labels, return_inverse=True)
    contingency = np.zeros((len(clusters), len(true_classes)), dtype=int)
    np.add.at(contingency, (cluster_encoded, true_encoded), 1)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = contingency[row_ind, col_ind].sum()
    return float(matched / len(y_true))


def adjusted_rand_index(y_true: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Adjusted Rand Index implemented from the contingency table formula."""
    y_true = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)
    if y_true.shape != cluster_labels.shape:
        raise ValueError("inputs must have the same shape")
    if y_true.size < 2:
        return 1.0

    _, true_encoded = np.unique(y_true, return_inverse=True)
    _, cluster_encoded = np.unique(cluster_labels, return_inverse=True)
    contingency = np.zeros(
        (cluster_encoded.max() + 1, true_encoded.max() + 1), dtype=int
    )
    np.add.at(contingency, (cluster_encoded, true_encoded), 1)

    sum_comb = sum(math.comb(int(n), 2) for n in contingency.ravel() if n >= 2)
    row_comb = sum(math.comb(int(n), 2) for n in contingency.sum(axis=1) if n >= 2)
    col_comb = sum(math.comb(int(n), 2) for n in contingency.sum(axis=0) if n >= 2)
    total_comb = math.comb(int(y_true.size), 2)
    expected = row_comb * col_comb / total_comb
    maximum = 0.5 * (row_comb + col_comb)
    denominator = maximum - expected
    if denominator == 0:
        return 1.0
    return float((sum_comb - expected) / denominator)
