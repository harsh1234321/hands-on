"""Run usability checks for KNN, KMeans, and LinearRegressionGD using sample datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_algorithms import KMeans, KNNClassifier, LinearRegressionGD


def _load_csv(path: Path, skip_header: bool = True) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", skip_header=1 if skip_header else 0)


def verify_knn(data_dir: Path) -> None:
    data = _load_csv(data_dir / "knn_classification.csv")
    x_train = data[:, :2]
    y_train = data[:, 2].astype(int)

    model = KNNClassifier(n_neighbors=3).fit(x_train, y_train)
    predictions = model.predict([[1.1, 1.0], [5.1, 5.0]])
    print("KNN predictions:", predictions.tolist())


def verify_kmeans(data_dir: Path) -> None:
    x = _load_csv(data_dir / "kmeans_clustering.csv")

    model = KMeans(n_clusters=3, random_state=42).fit(x)
    print("KMeans labels:", model.labels_.tolist())
    print("KMeans centroids:\n", model.centroids_)
    print("KMeans inertia:", round(model.inertia_, 4))


def verify_linear_regression(data_dir: Path) -> None:
    data = _load_csv(data_dir / "linear_regression.csv")
    x = data[:, :1]
    y = data[:, 1]

    model = LinearRegressionGD(learning_rate=0.05, n_iters=2000).fit(x, y)
    preds = model.predict([[7.0], [8.0]])
    print("LinearRegressionGD weights:", model.weights_.tolist())
    print("LinearRegressionGD bias:", round(model.bias_, 4))
    print("LinearRegressionGD predictions:", preds.round(4).tolist())


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "sample_data"

    verify_knn(data_dir)
    verify_kmeans(data_dir)
    verify_linear_regression(data_dir)


if __name__ == "__main__":
    main()
