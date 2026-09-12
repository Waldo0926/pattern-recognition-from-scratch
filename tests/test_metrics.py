import numpy as np
from sklearn.metrics import adjusted_rand_score

from prlib.metrics import adjusted_rand_index, clustering_accuracy, confusion_matrix


def test_clustering_accuracy_is_permutation_invariant():
    truth = np.array([0, 0, 1, 1, 2, 2])
    clusters = np.array([2, 2, 0, 0, 1, 1])
    assert clustering_accuracy(truth, clusters) == 1.0


def test_adjusted_rand_matches_sklearn():
    truth = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    clusters = np.array([1, 1, 0, 0, 0, 2, 2, 2])
    assert np.isclose(adjusted_rand_index(truth, clusters), adjusted_rand_score(truth, clusters))


def test_confusion_matrix_counts_pairs():
    truth = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    np.testing.assert_array_equal(confusion_matrix(truth, pred), np.array([[1, 1], [0, 2]]))
