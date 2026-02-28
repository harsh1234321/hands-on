# hands-on

This repo contains manual implementation of ML algorithms, cost functions, gradients etc. from scratch to dig deep into the maths and their way of working.

## Implemented from scratch (NumPy only)

- `ml_algorithms/knn.py` - K-Nearest Neighbors classifier
- `ml_algorithms/kmeans.py` - K-Means clustering
- `ml_algorithms/linear_regression_gd.py` - Linear Regression using batch Gradient Descent

## Install dependencies (uv)

```bash
uv sync
```

This installs `numpy` from `pyproject.toml` into `.venv`.

## Sample datasets

- `sample_data/knn_classification.csv`
- `sample_data/kmeans_clustering.csv`
- `sample_data/linear_regression.csv`

## Usability check script

Run all algorithms on sample datasets:

```bash
uv run python scripts/verify_algorithms.py
```

This script prints predictions/clusters/parameters so you can quickly validate usage end-to-end.
