import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from prlib.kmeans import KMeans


def test_three_well_separated_blobs_have_high_ari():
    X, y = make_blobs(n_samples=240, centers=3, cluster_std=0.35, random_state=42)
    labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    assert adjusted_rand_score(y, labels) > 0.95


def test_fit_is_reproducible_with_fixed_seed():
    X, _ = make_blobs(n_samples=120, centers=3, random_state=2)
    first = KMeans(n_clusters=3, random_state=7).fit(X)
    second = KMeans(n_clusters=3, random_state=7).fit(X)
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)
    np.testing.assert_array_equal(first.labels_, second.labels_)


def test_duplicate_points_and_empty_cluster_recovery_do_not_crash():
    X = np.array([[0.0], [0.0], [0.0], [10.0], [10.0], [20.0]])
    model = KMeans(n_clusters=3, random_state=3).fit(X)
    assert np.isfinite(model.cluster_centers_).all()
    assert len(model.labels_) == len(X)
