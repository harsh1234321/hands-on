"""Manual K-Means clustering implementation using NumPy only."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class KMeans:
    """A simple K-Means clustering implementation.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of update iterations.
    tol : float, default=1e-4
        Convergence threshold based on centroid movement.
    random_state : int | None, default=None
        Seed for reproducible centroid initialization.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if tol < 0:
            raise ValueError("tol must be non-negative")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(x), size=self.n_clusters, replace=False)
        return x[indices].copy()

    @staticmethod
    def _assign_labels(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(x[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, x: np.ndarray, labels: np.ndarray, old_centroids: np.ndarray) -> np.ndarray:
        centroids = np.zeros((self.n_clusters, x.shape[1]), dtype=float)
        for k in range(self.n_clusters):
            cluster_points = x[labels == k]
            if len(cluster_points) == 0:
                centroids[k] = old_centroids[k]
            else:
                centroids[k] = cluster_points.mean(axis=0)
        return centroids

    @staticmethod
    def _compute_inertia(x: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
        return float(np.sum((x - centroids[labels]) ** 2))

    def fit(self, x: Iterable[Iterable[float]]) -> "KMeans":
        """Compute K-Means clustering."""
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D array-like")
        if self.n_clusters > len(x_arr):
            raise ValueError("n_clusters cannot exceed number of samples")

        centroids = self._initialize_centroids(x_arr)

        for _ in range(self.max_iter):
            labels = self._assign_labels(x_arr, centroids)
            new_centroids = self._update_centroids(x_arr, labels, centroids)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift <= self.tol:
                break

        final_labels = self._assign_labels(x_arr, centroids)
        self.centroids_ = centroids
        self.labels_ = final_labels
        self.inertia_ = self._compute_inertia(x_arr, centroids, final_labels)
        return self

    def predict(self, x: Iterable[Iterable[float]]) -> np.ndarray:
        """Assign nearest cluster index for each sample."""
        if self.centroids_ is None:
            raise ValueError("Model is not fitted yet")
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        return self._assign_labels(x_arr, self.centroids_)
