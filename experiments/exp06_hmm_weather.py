"""Demonstrate HMM evaluation, decoding, sampling, and Baum-Welch learning."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prlib.hmm import DiscreteHiddenMarkovModel
from prlib.metrics import clustering_accuracy


RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def make_source_model() -> DiscreteHiddenMarkovModel:
    return DiscreteHiddenMarkovModel(
        n_states=2,
        n_observations=3,
        startprob=np.array([0.6, 0.4]),
        transmat=np.array([[0.75, 0.25], [0.35, 0.65]]),
        emissionprob=np.array([[0.1, 0.35, 0.55], [0.65, 0.25, 0.10]]),
        random_state=42,
    )


def main() -> None:
    source = make_source_model()
    observations, true_states = source.sample(120, random_state=8)

    start = time.perf_counter()
    viterbi_log_probability, decoded_states = source.decode(observations)
    decode_time = time.perf_counter() - start
    matched_state_accuracy = clustering_accuracy(true_states, decoded_states)

    training_sequences = [source.sample(100, random_state=i)[0] for i in range(12)]
    learner = DiscreteHiddenMarkovModel(2, 3, random_state=17)
    initial_log_likelihood = sum(learner.score(seq) for seq in training_sequences)
    start = time.perf_counter()
    learner.fit(training_sequences, n_iter=40, tol=1e-5)
    training_time = time.perf_counter() - start
    final_log_likelihood = sum(learner.score(seq) for seq in training_sequences)

    print(f"sequence_log_likelihood={source.score(observations):.4f}")
    print(f"viterbi_log_probability={viterbi_log_probability:.4f}")
    print(f"viterbi_matched_state_accuracy={matched_state_accuracy:.4f}")
    print(f"decode_time_seconds={decode_time:.6f}")
    print(f"baum_welch_initial_log_likelihood={initial_log_likelihood:.4f}")
    print(f"baum_welch_final_log_likelihood={final_log_likelihood:.4f}")
    print(f"baum_welch_time_seconds={training_time:.6f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    horizon = 80
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(range(horizon), true_states[:horizon], where="mid", label="True state")
    ax.step(
        range(horizon),
        decoded_states[:horizon] + 0.04,
        where="mid",
        label="Viterbi state",
        alpha=0.8,
    )
    ax.set_yticks([0, 1])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Hidden state")
    ax.set_title("HMM Viterbi Decoding on a Synthetic Weather Sequence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "hmm_viterbi_state_path.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(learner.log_likelihood_history_) + 1), learner.log_likelihood_history_)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Training log-likelihood")
    ax.set_title("Baum-Welch Learning Curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "hmm_baum_welch_learning_curve.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
