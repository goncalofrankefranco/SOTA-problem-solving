"""NOAI Singapore 2025, Section 2 Question 1.

Implements the requested batch gradient descent using only NumPy.  The input
matrix is expected to include its intercept column, as in the task statement.
"""

from __future__ import annotations

import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    """Fit linear-regression coefficients by batch gradient descent.

    Parameters follow the official task: ``X`` already contains an intercept
    column, and ``y`` contains one target per row.  Coefficients start at zero
    and the gradient is the derivative of mean squared error (with the common
    1/2 factor, which cancels the derivative's 2).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D matrix aligned with y")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")

    m, n = X.shape
    theta = np.zeros((n, 1), dtype=float)
    for _ in range(iterations):
        errors = X @ theta - y
        theta -= alpha * (X.T @ errors) / m

    return np.round(theta.ravel(), 4)


if __name__ == "__main__":
    X = np.array([[2, 2], [2, 4], [2, 6]], dtype=float)
    y = np.array([2, 4, 6], dtype=float)
    print(linear_regression_gradient_descent(X, y, alpha=0.01, iterations=1000))
