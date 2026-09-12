import numpy as np

from prlib.hmm import DiscreteHiddenMarkovModel


def make_weather_model():
    return DiscreteHiddenMarkovModel(
        n_states=2,
        n_observations=3,
        startprob=np.array([0.6, 0.4]),
        transmat=np.array([[0.7, 0.3], [0.4, 0.6]]),
        emissionprob=np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]),
    )


def test_forward_probability_matches_known_example():
    model = make_weather_model()
    observations = np.array([0, 1, 2])
    likelihood = np.exp(model.score(observations))
    assert np.isclose(likelihood, 0.033612, atol=1e-6)


def test_viterbi_returns_valid_state_path():
    model = make_weather_model()
    observations = np.array([0, 1, 2, 2, 1])
    log_prob, states = model.decode(observations)
    assert len(states) == len(observations)
    assert set(states).issubset({0, 1})
    assert np.isfinite(log_prob)


def test_baum_welch_improves_training_log_likelihood():
    source = make_weather_model()
    sequences = [source.sample(60, random_state=i)[0] for i in range(8)]
    learner = DiscreteHiddenMarkovModel(2, 3, random_state=11)
    before = sum(learner.score(seq) for seq in sequences)
    learner.fit(sequences, n_iter=20, tol=1e-6)
    after = sum(learner.score(seq) for seq in sequences)
    assert after >= before - 1e-8
    assert np.allclose(learner.startprob_.sum(), 1.0)
    assert np.allclose(learner.transmat_.sum(axis=1), 1.0)
    assert np.allclose(learner.emissionprob_.sum(axis=1), 1.0)
