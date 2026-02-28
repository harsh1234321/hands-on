"""Manual K-Nearest Neighbors (KNN) implementation using NumPy only."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


class KNNClassifier:
    """A simple KNN classifier for numeric features.

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of nearest neighbors to consider for prediction.
    """

    def __init__(self, n_neighbors: int = 3) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        self.n_neighbors = n_neighbors
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x: Iterable[Iterable[float]], y: Iterable[int]) -> "KNNClassifier":
        """Store training data for neighbor lookups."""
        x_train = np.asarray(x, dtype=float)
        y_train = np.asarray(y)

        if x_train.ndim != 2:
            raise ValueError("x must be a 2D array-like")
        if y_train.ndim != 1:
            raise ValueError("y must be a 1D array-like")
        if len(x_train) != len(y_train):
            raise ValueError("x and y must contain the same number of samples")
        if self.n_neighbors > len(x_train):
            raise ValueError("n_neighbors cannot be greater than number of training samples")

        self._x_train = x_train
        self._y_train = y_train
        return self

    def _euclidean_distances(self, sample: np.ndarray) -> np.ndarray:
        if self._x_train is None:
            raise ValueError("Model is not fitted yet")
        return np.sqrt(np.sum((self._x_train - sample) ** 2, axis=1))

    @staticmethod
    def _majority_vote(labels: np.ndarray) -> int:
        counts = Counter(labels)
        max_count = max(counts.values())
        candidates = [label for label, count in counts.items() if count == max_count]
        return int(min(candidates))

    def predict(self, x: Iterable[Iterable[float]]) -> np.ndarray:
        """Predict labels for each row in x."""
        if self._x_train is None or self._y_train is None:
            raise ValueError("Model is not fitted yet")

        x_test = np.asarray(x, dtype=float)
        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        predictions = []
        for sample in x_test:
            distances = self._euclidean_distances(sample)
            nearest_indices = np.argsort(distances)[: self.n_neighbors]
            nearest_labels = self._y_train[nearest_indices]
            predictions.append(self._majority_vote(nearest_labels))

        return np.array(predictions)
