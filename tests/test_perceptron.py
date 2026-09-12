import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prlib.perceptron import MulticlassPerceptron


def test_linearly_separable_multiclass_data():
    X = np.array([[2, 0], [3, 0], [0, 2], [0, 3], [-2, -2], [-3, -2]], dtype=float)
    y = np.array([0, 0, 1, 1, 2, 2])
    model = MulticlassPerceptron(max_epochs=100, random_state=1).fit(X, y)
    assert model.score(X, y) == 1.0


def test_digits_accuracy_is_reasonably_close_to_sklearn():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    ours = MulticlassPerceptron(max_epochs=60, learning_rate=0.1, random_state=42).fit(
        X_train, y_train
    )
    reference = Perceptron(max_iter=60, random_state=42, tol=None).fit(X_train, y_train)
    assert ours.score(X_test, y_test) > 0.90
    assert abs(ours.score(X_test, y_test) - reference.score(X_test, y_test)) < 0.08


def test_single_class_is_supported():
    X = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([7, 7, 7])
    model = MulticlassPerceptron(max_epochs=5).fit(X, y)
    assert np.all(model.predict(X) == 7)
