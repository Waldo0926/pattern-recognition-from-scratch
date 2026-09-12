# Algorithm notes

These notes summarize the mathematical ideas behind the implementations. They are independently written for this repository rather than copied from course slides.

## 1. Gaussian Naive Bayes

For a class \(C_k\) and feature vector \(x=(x_1,\dots,x_d)\), Naive Bayes assumes conditional independence:

\[
P(x\mid C_k)=\prod_{j=1}^{d}P(x_j\mid C_k).
\]

With Gaussian feature likelihoods,

\[
P(x_j\mid C_k)=\frac{1}{\sqrt{2\pi\sigma_{kj}^2}}
\exp\left(-\frac{(x_j-\mu_{kj})^2}{2\sigma_{kj}^2}\right).
\]

The implementation works in log space:

\[
\log P(C_k\mid x) \propto \log P(C_k) + \sum_j \log P(x_j\mid C_k),
\]

which avoids numerical underflow caused by multiplying many small densities. A small variance-smoothing term protects constant features.

## 2. Multiclass Perceptron

Each class has a weight vector \(w_k\). Prediction chooses the largest linear score:

\[
\hat y=\arg\max_k (w_k^T x+b_k).
\]

For a misclassified example with true class \(y\) and predicted class \(\hat y\), the update is:

\[
w_y \leftarrow w_y+\eta x,\qquad
w_{\hat y} \leftarrow w_{\hat y}-\eta x.
\]

The same update is applied to the corresponding intercepts. The training curve stores the number of mistakes per epoch so convergence behavior is visible.

## 3. K-Means

K-Means minimizes within-cluster squared distance:

\[
J=\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|^2.
\]

It alternates between assigning each point to its nearest centroid and recomputing each centroid as the mean of its assigned points. The implementation uses k-means++ initialization to reduce sensitivity to unlucky random starting centers.

Cluster identifiers are arbitrary. Therefore `cluster 0 == class 0` is not a valid assumption. The experiments report Adjusted Rand Index and Normalized Mutual Information and, where an accuracy-like number is useful, perform an optimal one-to-one label assignment first.

## 4. Fuzzy C-Means

FCM generalizes K-Means by giving every sample a membership degree \(u_{ij}\) for every cluster rather than a single hard assignment. It minimizes

\[
J_m=\sum_i\sum_j u_{ij}^m\left\|x_i-c_j\right\|^2,
\]

where \(m>1\) controls fuzziness. Cluster centers are weighted by \(u_{ij}^m\), and memberships are updated from relative distances to all centers. The final hard label used for ARI/NMI is simply the cluster with maximum membership, but the membership matrix retains uncertainty information.

## 5. Fisher Linear Discriminant Analysis

Fisher LDA seeks a projection that keeps samples from the same class compact while separating different class means. Define within-class scatter \(S_W\) and between-class scatter \(S_B\). The discriminant directions maximize a generalized Rayleigh quotient:

\[
J(W)=\frac{|W^T S_B W|}{|W^T S_W W|}.
\]

In the multiclass implementation, the leading eigenvectors of

\[
S_W^{-1}S_B
\]

form the projection. Because \(S_W\) can be singular, the code adds a small regularization term and uses a pseudoinverse. Classification is then performed by the nearest class centroid in the discriminant subspace.

Unlike PCA, which ignores labels and maximizes total variance, Fisher LDA explicitly uses labels to maximize class separability.

## 6. Hidden Markov Model

A discrete HMM is parameterized by

- initial-state probabilities \(\pi\),
- transition matrix \(A\),
- emission matrix \(B\).

It addresses three classic problems.

### Evaluation — Forward algorithm

Compute \(P(O\mid\lambda)\) efficiently through dynamic programming instead of enumerating every hidden-state sequence. The implementation scales the forward variables at each time step to prevent underflow and accumulates the corresponding log-likelihood.

### Decoding — Viterbi algorithm

Find the most likely hidden-state sequence:

\[
Q^*=\arg\max_Q P(Q,O\mid\lambda).
\]

Viterbi keeps only the best predecessor for each state at each time step and then backtracks through those pointers.

### Learning — Baum-Welch

When hidden states are unknown, Baum-Welch uses expectation-maximization. The E-step estimates posterior state and transition probabilities from forward/backward quantities; the M-step normalizes those expected counts into updated \(\pi\), \(A\), and \(B\). The experiment plots the training log-likelihood across EM iterations.
