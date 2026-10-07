"""Reference implementation for NOAI Singapore 2026 Programming Task 1.

Only NumPy and scikit-learn's standard DecisionTreeRegressor are used, as in
the released task.  The code covers the assessed L1/L2 gradient, initialization,
boosting, prediction, and early-stopping components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def l1_negative_gradient(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Negative gradient of absolute error, using sign at zero as 0."""
    return np.sign(np.asarray(y) - np.asarray(y_pred))


def l2_negative_gradient(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Negative gradient of one-half squared error."""
    return np.asarray(y) - np.asarray(y_pred)


def l1_initial_prediction(y: np.ndarray) -> np.float64:
    """Best constant L1 predictor: the target median."""
    return np.float64(np.median(y))


def l2_initial_prediction(y: np.ndarray) -> np.float64:
    """Best constant L2 predictor: the target mean."""
    return np.float64(np.mean(y))


class BaseGradientBoosting(ABC):
    """Shared logic for gradient-boosting regressors."""

    def _compute_negative_gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.criterion == "l1":
            return l1_negative_gradient(y, y_pred)
        if self.criterion == "l2":
            return l2_negative_gradient(y, y_pred)
        raise ValueError(f"Unknown criterion: {self.criterion!r}")

    def _compute_initial_prediction(self, y: np.ndarray) -> np.float64:
        if self.criterion == "l1":
            return l1_initial_prediction(y)
        if self.criterion == "l2":
            return l2_initial_prediction(y)
        raise ValueError(f"Unknown criterion: {self.criterion!r}")

    def _subsample_indices(self, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        if self.subsample < 1.0:
            count = max(1, int(n_samples * self.subsample))
            return rng.choice(n_samples, size=count, replace=False)
        return np.arange(n_samples)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.init_prediction is None:
            raise ValueError("Model not fitted yet")
        X = np.asarray(X)
        predictions = np.full(X.shape[0], self.init_prediction, dtype=float)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the estimator."""


class GradientBoostingRegressor(BaseGradientBoosting):
    """Small, transparent gradient-boosting regressor for L1 or L2 loss."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        criterion: str = "l2",
        subsample: float = 1.0,
        random_state: int | None = 42,
    ) -> None:
        if n_estimators < 0:
            raise ValueError("n_estimators must be non-negative")
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if criterion not in {"l1", "l2"}:
            raise ValueError("criterion must be 'l1' or 'l2'")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.criterion = criterion
        self.subsample = subsample
        self.random_state = random_state
        self.trees: list[DecisionTreeRegressor] = []
        self.init_prediction: np.float64 | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be 2D and have the same number of rows as y")

        self.trees = []
        self.init_prediction = self._compute_initial_prediction(y)
        current = np.full(y.shape, self.init_prediction, dtype=float)
        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_estimators):
            residual = self._compute_negative_gradient(y, current)
            indices = self._subsample_indices(X.shape[0], rng)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=rng.randint(0, 10000),
            )
            tree.fit(X[indices], residual[indices])
            current += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self


class GBREarlyStop(GradientBoostingRegressor):
    """Gradient boosting that retains the validation-best tree prefix."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> "GBREarlyStop":
        if patience < 1:
            raise ValueError("patience must be at least 1")
        X_train, X_val = np.asarray(X_train), np.asarray(X_val)
        y_train, y_val = np.asarray(y_train).reshape(-1), np.asarray(y_val).reshape(-1)
        if X_train.ndim != 2 or X_train.shape[0] != y_train.size:
            raise ValueError("X_train and y_train have incompatible shapes")
        if X_val.ndim != 2 or X_val.shape[0] != y_val.size:
            raise ValueError("X_val and y_val have incompatible shapes")

        self.trees = []
        self.init_prediction = self._compute_initial_prediction(y_train)
        train_pred = np.full(y_train.shape, self.init_prediction, dtype=float)
        val_pred = np.full(y_val.shape, self.init_prediction, dtype=float)
        rng = np.random.RandomState(self.random_state)
        best_loss = float("inf")
        best_count = 0
        stale = 0

        for round_idx in range(self.n_estimators):
            residual = self._compute_negative_gradient(y_train, train_pred)
            indices = self._subsample_indices(X_train.shape[0], rng)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=rng.randint(0, 10000),
            )
            tree.fit(X_train[indices], residual[indices])
            train_pred += self.learning_rate * tree.predict(X_train)
            val_pred += self.learning_rate * tree.predict(X_val)
            self.trees.append(tree)

            errors = y_val - val_pred
            val_loss = float(np.mean(errors ** 2 if self.criterion == "l2" else np.abs(errors)))
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_count = round_idx + 1
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        self.trees = self.trees[:best_count]
        self.n_estimators = best_count
        return self
