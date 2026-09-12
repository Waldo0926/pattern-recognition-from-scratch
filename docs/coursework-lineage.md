# Coursework lineage and repository scope

This repository grew out of undergraduate study in **Pattern Recognition**. It is intentionally not a dump of the original course folder: lecture slides, answer sheets, temporary IDE files, virtual environments, personal identifiers, and third-party coursework are excluded.

## What existed in the original coursework

The original hands-on experiments included:

- Gaussian Naive Bayes for handwritten digit recognition.
- A multiclass perceptron for handwritten digit recognition.
- K-Means clustering on the Wine dataset.
- Fuzzy C-Means clustering as part of the unsupervised-pattern-recognition work.

The course also covered **Fisher Linear Discriminant Analysis** and **Hidden Markov Models** as formal topics. Those two algorithms were revisited and implemented for this repository so that the codebase reflects the broader material studied in the course.

## What was changed for this repository

The original experiment code was useful for learning, but it contained typical coursework shortcuts: fixed train/test slices, local absolute paths, non-descriptive variable names, direct multiplication of many Gaussian densities, and cluster-label evaluation that assumed cluster IDs were class IDs.

The repository therefore uses clean-room rewrites with:

- reproducible random seeds and stratified splits;
- log-domain Gaussian Naive Bayes calculations;
- vectorized NumPy implementations;
- k-means++ initialization;
- label-invariant clustering metrics (ARI/NMI and optimal label matching);
- tests against controlled data and scikit-learn references where appropriate;
- a consistent package structure and CI.

The purpose is to show the progression from learning the algorithms in class to understanding, testing, and engineering them more carefully.
