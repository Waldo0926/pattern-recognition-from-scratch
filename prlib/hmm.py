"""Discrete Hidden Markov Model algorithms implemented with NumPy."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class DiscreteHiddenMarkovModel:
    """Finite-state HMM with categorical emissions.

    Supports scaled forward/backward evaluation, Viterbi decoding, and
    Baum-Welch parameter estimation for one or more observation sequences.
    """

    def __init__(
        self,
        n_states: int,
        n_observations: int,
        startprob: np.ndarray | None = None,
        transmat: np.ndarray | None = None,
        emissionprob: np.ndarray | None = None,
        random_state: int | None = 42,
    ) -> None:
        if n_states <= 0 or n_observations <= 0:
            raise ValueError("n_states and n_observations must be positive")
        self.n_states = int(n_states)
        self.n_observations = int(n_observations)
        self.random_state = random_state
        rng = np.random.default_rng(random_state)

        self.startprob_ = self._normalize_vector(
            startprob if startprob is not None else rng.random(self.n_states)
        )
        self.transmat_ = self._normalize_rows(
            transmat if transmat is not None else rng.random((self.n_states, self.n_states))
        )
        self.emissionprob_ = self._normalize_rows(
            emissionprob
            if emissionprob is not None
            else rng.random((self.n_states, self.n_observations))
        )
        self.log_likelihood_history_: list[float] = []

    def forward(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        obs = self._validate_sequence(observations)
        alpha = np.zeros((len(obs), self.n_states), dtype=float)
        scales = np.zeros(len(obs), dtype=float)

        alpha[0] = self.startprob_ * self.emissionprob_[:, obs[0]]
        scales[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= scales[0]

        for t in range(1, len(obs)):
            alpha[t] = (alpha[t - 1] @ self.transmat_) * self.emissionprob_[:, obs[t]]
            scales[t] = max(alpha[t].sum(), 1e-300)
            alpha[t] /= scales[t]

        log_likelihood = float(np.log(scales).sum())
        return alpha, scales, log_likelihood

    def backward(self, observations: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
        obs = self._validate_sequence(observations)
        if scales is None:
            _, scales, _ = self.forward(obs)
        beta = np.zeros((len(obs), self.n_states), dtype=float)
        beta[-1] = 1.0
        for t in range(len(obs) - 2, -1, -1):
            beta[t] = self.transmat_ @ (
                self.emissionprob_[:, obs[t + 1]] * beta[t + 1]
            )
            beta[t] /= max(scales[t + 1], 1e-300)
        return beta

    def score(self, observations: np.ndarray) -> float:
        """Return log P(observations | model)."""
        return self.forward(observations)[2]

    def decode(self, observations: np.ndarray) -> tuple[float, np.ndarray]:
        """Viterbi decode the most likely hidden-state sequence."""
        obs = self._validate_sequence(observations)
        tiny = 1e-300
        log_start = np.log(np.maximum(self.startprob_, tiny))
        log_trans = np.log(np.maximum(self.transmat_, tiny))
        log_emit = np.log(np.maximum(self.emissionprob_, tiny))

        delta = np.zeros((len(obs), self.n_states), dtype=float)
        psi = np.zeros((len(obs), self.n_states), dtype=int)
        delta[0] = log_start + log_emit[:, obs[0]]

        for t in range(1, len(obs)):
            scores = delta[t - 1][:, None] + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_emit[:, obs[t]]

        states = np.zeros(len(obs), dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(len(obs) - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return float(np.max(delta[-1])), states

    def fit(
        self,
        sequences: Iterable[np.ndarray] | np.ndarray,
        n_iter: int = 50,
        tol: float = 1e-4,
        pseudocount: float = 1e-6,
    ) -> "DiscreteHiddenMarkovModel":
        """Estimate model parameters with Baum-Welch EM."""
        if isinstance(sequences, np.ndarray) and sequences.ndim == 1:
            sequence_list = [self._validate_sequence(sequences)]
        else:
            sequence_list = [self._validate_sequence(seq) for seq in sequences]
        if not sequence_list:
            raise ValueError("At least one observation sequence is required")

        self.log_likelihood_history_ = []
        for _ in range(n_iter):
            start_counts = np.full(self.n_states, pseudocount, dtype=float)
            trans_counts = np.full((self.n_states, self.n_states), pseudocount, dtype=float)
            emission_counts = np.full(
                (self.n_states, self.n_observations), pseudocount, dtype=float
            )
            total_log_likelihood = 0.0

            for obs in sequence_list:
                alpha, scales, log_likelihood = self.forward(obs)
                beta = self.backward(obs, scales)
                total_log_likelihood += log_likelihood

                gamma = alpha * beta
                gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
                start_counts += gamma[0]

                for t in range(len(obs) - 1):
                    xi = (
                        alpha[t][:, None]
                        * self.transmat_
                        * self.emissionprob_[:, obs[t + 1]][None, :]
                        * beta[t + 1][None, :]
                    )
                    xi /= max(float(xi.sum()), 1e-300)
                    trans_counts += xi

                for symbol in range(self.n_observations):
                    mask = obs == symbol
                    if np.any(mask):
                        emission_counts[:, symbol] += gamma[mask].sum(axis=0)

            self.startprob_ = self._normalize_vector(start_counts)
            self.transmat_ = self._normalize_rows(trans_counts)
            self.emissionprob_ = self._normalize_rows(emission_counts)
            self.log_likelihood_history_.append(total_log_likelihood)

            if len(self.log_likelihood_history_) >= 2:
                improvement = (
                    self.log_likelihood_history_[-1] - self.log_likelihood_history_[-2]
                )
                if abs(improvement) <= tol:
                    break
        return self

    def sample(self, n_steps: int, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        states = np.zeros(n_steps, dtype=int)
        observations = np.zeros(n_steps, dtype=int)
        states[0] = int(rng.choice(self.n_states, p=self.startprob_))
        observations[0] = int(rng.choice(self.n_observations, p=self.emissionprob_[states[0]]))
        for t in range(1, n_steps):
            states[t] = int(rng.choice(self.n_states, p=self.transmat_[states[t - 1]]))
            observations[t] = int(
                rng.choice(self.n_observations, p=self.emissionprob_[states[t]])
            )
        return observations, states

    def _validate_sequence(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=int)
        if obs.ndim != 1 or len(obs) == 0:
            raise ValueError("observations must be a non-empty 1D array")
        if obs.min() < 0 or obs.max() >= self.n_observations:
            raise ValueError("observation symbol outside configured range")
        return obs

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        total = float(vector.sum())
        if vector.ndim != 1 or total <= 0:
            raise ValueError("probability vector must be one-dimensional and positive")
        return vector / total

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or np.any(matrix < 0):
            raise ValueError("probability matrix must be two-dimensional and non-negative")
        sums = matrix.sum(axis=1, keepdims=True)
        if np.any(sums <= 0):
            raise ValueError("every probability row must have positive mass")
        return matrix / sums
