import numpy as np
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prlib.lda import FisherLDA


def test_simple_classes_are_separated():
    X = np.array([[-2, 0], [-1.5, 0.2], [2, 0], [1.5, -0.2]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = FisherLDA(n_components=1).fit(X, y)
    assert model.score(X, y) == 1.0


def test_wine_accuracy_is_close_to_sklearn_lda():
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    ours = FisherLDA(n_components=2).fit(X_train, y_train)
    reference = LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train)
    assert ours.score(X_test, y_test) > 0.90
    assert abs(ours.score(X_test, y_test) - reference.score(X_test, y_test)) < 0.08


def test_projection_dimension_is_limited_by_number_of_classes():
    X, y = load_wine(return_X_y=True)
    model = FisherLDA(n_components=2).fit(X, y)
    assert model.transform(X[:4]).shape == (4, 2)
