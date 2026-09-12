# Data

`wine.csv` is a cleaned UTF-8 copy of the classic Wine recognition dataset used in the original coursework experiment. The experiments in this repository load the equivalent dataset through `sklearn.datasets.load_wine()` so that the workflow is self-contained and does not depend on local file paths.

The first column is the class label and the remaining 13 columns are continuous chemical-analysis features.

Dataset reference: UCI Machine Learning Repository, **Wine** dataset. The same dataset is distributed through scikit-learn for educational and benchmarking use.
