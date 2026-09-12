import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from prlib.naive_bayes import GaussianNaiveBayes


def test_separable_data_is_classified_correctly():
    X = np.array([[0.0, 0.0], [0.1, 0.2], [3.0, 3.0], [3.1, 2.9]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNaiveBayes().fit(X, y)
    assert model.score(X, y) == 1.0


def test_digits_accuracy_is_close_to_sklearn():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    ours = GaussianNaiveBayes(var_smoothing=1e-9).fit(X_train, y_train)
    reference = GaussianNB(var_smoothing=1e-9).fit(X_train, y_train)
    assert abs(ours.score(X_test, y_test) - reference.score(X_test, y_test)) < 0.03


def test_constant_feature_does_not_break_probability_calculation():
    X = np.array([[0.0, 1.0], [0.0, 1.2], [0.0, 4.0], [0.0, 4.2]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNaiveBayes().fit(X, y)
    log_prob = model.predict_log_proba(X)
    assert np.isfinite(log_prob).all()
