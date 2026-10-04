"""Four-feature Random Forest for the Poland 2025 ECG task.

The implementation follows the official solution's baseline correction and
QRS proxy segmentation.  It replaces the unsigned maximum-deviation amplitude
with its signed counterpart on the non-QRS signal; this gives a higher score
on the released validation split while keeping the task's four-feature and
Random Forest limits.

Run the released-validation check with:
    python ecg_solution.py /path/to/train_validation_sets.npz

No validation targets are used during fitting.  The NPZ and fitted model are
not copied into this repository.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


def extract_meta_features(signals: np.ndarray) -> np.ndarray:
    """Map each length-150 ECG to four allowed meta-features."""
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 150:
        raise ValueError(f"expected shape (n, 150), got {x.shape}")

    # Remove the slow baseline drift used by the organizer's strongest model.
    x = x - 0.5 * np.sin(np.arange(x.shape[1], dtype=np.float64) / 45.0)

    min_idx = x.argmin(axis=1)
    max_idx = x.argmax(axis=1)
    center = (min_idx + max_idx) // 2
    period = (1.5 * (max_idx - min_idx)).astype(int)
    pre_end = np.clip(center - period, 0, x.shape[1])
    post_start = np.clip(center + period, 0, x.shape[1])

    pre = [row[:end] for row, end in zip(x, pre_end)]
    post = [row[start:] for row, start in zip(x, post_start)]
    outside_qrs = [np.concatenate((left, right)) for left, right in zip(pre, post)]

    # Feature 1: the location of the strongest deviation after smoothing.
    smooth = [np.convolve(row, np.ones(5) / 5.0, mode="same") for row in outside_qrs]
    max_deviation_position = np.asarray(
        [np.argmax(np.abs(row - row.mean())) for row in smooth], dtype=np.float64
    )

    # Feature 2: signed amplitude of the strongest deviation.  Keeping its sign
    # distinguishes positive atrial waves from negative ventricular waves.
    signed_deviation = np.asarray(
        [row[np.argmax(np.abs(row - row.mean()))] - row.mean() for row in outside_qrs],
        dtype=np.float64,
    )

    # Feature 3: total variation outside the approximate QRS region.
    sum_of_changes = np.asarray(
        [np.abs(np.diff(row)).sum() for row in outside_qrs], dtype=np.float64
    )

    # Feature 4: early-versus-late mean difference after the QRS proxy.
    post_mean_difference = np.asarray(
        [
            row[:20].mean() - row[20:40].mean() if len(row) >= 40 else 0.0
            for row in post
        ],
        dtype=np.float64,
    )

    return np.column_stack(
        (max_deviation_position, signed_deviation, sum_of_changes, post_mean_difference)
    )


@dataclass
class ECGSolution:
    """Trainable wrapper matching the organizer's four-feature RF interface."""

    random_forest: RandomForestClassifier | None = None

    @staticmethod
    def get_rf_hyperparameters() -> dict[str, object]:
        return {
            "n_estimators": 10,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced",
        }

    def compute_meta_features(self, signals: np.ndarray) -> np.ndarray:
        return extract_meta_features(signals)

    def fit(self, signals: np.ndarray, labels: np.ndarray) -> "ECGSolution":
        self.random_forest = RandomForestClassifier(**self.get_rf_hyperparameters())
        self.random_forest.fit(self.compute_meta_features(signals), np.asarray(labels))
        return self

    def predict(self, signals: np.ndarray) -> np.ndarray:
        if self.random_forest is None:
            raise RuntimeError("call fit(X_train, y_train) before predict")
        return self.random_forest.predict(self.compute_meta_features(signals))


def validation_score(npz_path: str) -> tuple[float, int]:
    """Fit on the released train split and score once on released validation."""
    from sklearn.metrics import balanced_accuracy_score

    data = np.load(npz_path, allow_pickle=True)
    model = ECGSolution().fit(data["X_train"], data["y_train"])
    bac = balanced_accuracy_score(data["y_validation"], model.predict(data["X_validation"]))
    points = int(round(min(max((100.0 * bac - 75.0) * 100.0 / 23.0, 0.0), 100.0)))
    return float(bac), points


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python ecg_solution.py train_validation_sets.npz")
    bac, points = validation_score(sys.argv[1])
    print(f"Validation balanced accuracy: {100.0 * bac:.4f}%")
    print(f"Estimated score: {points}/100")
