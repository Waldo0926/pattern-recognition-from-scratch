# Benchmark snapshot

The values below were produced by the repository's experiment scripts with `random_state=42` where applicable. Runtime is only an illustrative local measurement and should not be interpreted as a rigorous performance benchmark.

| Algorithm | Dataset / task | From-scratch result | Reference / comparison |
|---|---|---:|---:|
| Gaussian Naive Bayes | Digits classification | Accuracy **0.8289** | sklearn GaussianNB **0.8289** |
| Multiclass Perceptron | Digits classification | Accuracy **0.9467** | sklearn Perceptron **0.9311** |
| K-Means | Wine clustering | ARI **0.8975**, matched acc. **0.9663** | sklearn KMeans ARI **0.8975** |
| Fuzzy C-Means | Wine clustering | ARI **0.8975**, matched acc. **0.9663** | K-Means shown separately above |
| Fisher LDA + nearest centroid | Wine classification | Accuracy **1.0000** | sklearn LDA **0.9556** |
| HMM | Synthetic sequence decoding | Matched hidden-state acc. **0.8333** | known generating HMM |

## HMM learning check

For 12 synthetic sequences of length 100, Baum-Welch improved total training log-likelihood from **-1483.26** to **-1278.95** in the captured run.

## Example local runtimes

| Experiment | From-scratch runtime | Reference runtime |
|---|---:|---:|
| Gaussian Naive Bayes | ~0.005 s | ~0.004 s |
| Multiclass Perceptron | ~0.188 s | ~0.073 s |
| K-Means | ~0.001 s | ~0.026 s |
| Fisher LDA | ~0.001 s | ~0.003 s |
| HMM Baum-Welch | ~0.94 s | — |

These timings are deliberately secondary: the project is about transparent implementations and reproducible algorithmic behavior, not outperforming optimized library code.
