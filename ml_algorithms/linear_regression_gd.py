"""Manual Linear Regression with batch Gradient Descent using NumPy only."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class LinearRegressionGD:
    """Linear regression optimized with batch gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient updates.
    n_iters : int, default=1000
        Number of gradient descent iterations.
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iters: int = 1000,
        fit_intercept: bool = True,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iters <= 0:
            raise ValueError("n_iters must be a positive integer")

        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def fit(self, x: Iterable[Iterable[float]], y: Iterable[float]) -> "LinearRegressionGD":
        """Fit model parameters using gradient descent."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D array-like")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array-like")
        if len(x_arr) != len(y_arr):
            raise ValueError("x and y must contain the same number of samples")

        n_samples, n_features = x_arr.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iters):
            y_pred = x_arr @ self.weights_ + (self.bias_ if self.fit_intercept else 0.0)
            errors = y_pred - y_arr

            dw = (1 / n_samples) * (x_arr.T @ errors)
            db = float((1 / n_samples) * np.sum(errors))

            self.weights_ -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias_ -= self.learning_rate * db

            loss = self._mse(y_arr, y_pred)
            self.loss_history_.append(loss)

        return self

    def predict(self, x: Iterable[Iterable[float]]) -> np.ndarray:
        """Predict target values."""
        if self.weights_ is None:
            raise ValueError("Model is not fitted yet")

        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        return x_arr @ self.weights_ + (self.bias_ if self.fit_intercept else 0.0)
