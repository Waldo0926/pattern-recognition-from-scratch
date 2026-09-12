import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from prlib.fuzzy_cmeans import FuzzyCMeans


def test_membership_rows_sum_to_one():
    X, _ = make_blobs(n_samples=90, centers=3, cluster_std=0.4, random_state=5)
    model = FuzzyCMeans(n_clusters=3, random_state=5).fit(X)
    np.testing.assert_allclose(model.membership_.sum(axis=1), 1.0, atol=1e-8)


def test_well_separated_blobs_have_high_ari():
    X, y = make_blobs(n_samples=180, centers=3, cluster_std=0.35, random_state=42)
    labels = FuzzyCMeans(n_clusters=3, random_state=42).fit_predict(X)
    assert adjusted_rand_score(y, labels) > 0.95


def test_objective_is_nonincreasing_with_tolerance():
    X, _ = make_blobs(n_samples=120, centers=3, random_state=7)
    model = FuzzyCMeans(n_clusters=3, random_state=7).fit(X)
    differences = np.diff(model.objective_history_)
    assert np.all(differences <= 1e-7)
