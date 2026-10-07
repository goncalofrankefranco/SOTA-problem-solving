#!/usr/bin/env python3
"""Train a default-focused classifier for the 2025 CAIO loan task.

Expected files: train.csv and test.csv from the official CAIO Drive folder.
The optional test_with_target.csv is a post-contest release and is used only
for a final, explicitly labeled evaluation; it never enters model fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "loan_paid_back"
ID_CANDIDATES = ("id", "ID", "loan_id")


def make_pipeline(frame: pd.DataFrame, seed: int) -> Pipeline:
    """Build a compact one-hot + histogram-gradient-boosting pipeline."""
    categorical = list(frame.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric = [c for c in frame.columns if c not in categorical]

    transform = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=450,
        max_leaf_nodes=31,
        l2_regularization=2.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=seed,
    )
    return Pipeline([("features", transform), ("model", classifier)])


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Return the threshold that maximizes F1 for default (encoded as 1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if thresholds.size == 0:
        return 0.5, float(f1_score(y_true, probability >= 0.5, zero_division=0))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def default_target(series: pd.Series) -> np.ndarray:
    """Map the supplied label (1 = repaid, 0 = default) to 1 = default."""
    paid = pd.to_numeric(series, errors="raise").astype(int).to_numpy()
    values = set(np.unique(paid).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0/1; found {sorted(values)}")
    return 1 - paid


def id_column(frame: pd.DataFrame) -> str | None:
    return next((name for name in ID_CANDIDATES if name in frame.columns), None)


def feature_frame(frame: pd.DataFrame, label: bool) -> pd.DataFrame:
    drop = [c for c in [id_column(frame), TARGET if label else None] if c is not None]
    return frame.drop(columns=drop).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--released-test-target", type=Path, help="Optional post-contest test_with_target.csv")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    if TARGET not in train:
        raise ValueError(f"Training data must contain {TARGET!r}")

    y = default_target(train[TARGET])
    X = feature_frame(train, label=True)
    X_test = feature_frame(test, label=False).reindex(columns=X.columns)
    counts = np.bincount(y, minlength=2)
    print(f"train={len(train):,}; features={X.shape[1]}; defaults={counts[1]:,} ({counts[1]/len(y):.3%})")
    print(f"categorical columns={list(X.select_dtypes(include=['object', 'category', 'bool']).columns)}")

    train_idx, valid_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=args.seed, stratify=y
    )
    validation_model = make_pipeline(X.iloc[train_idx], args.seed)
    validation_model.fit(X.iloc[train_idx], y[train_idx])
    p_valid = validation_model.predict_proba(X.iloc[valid_idx])[:, 1]
    threshold, best_f1 = best_f1_threshold(y[valid_idx], p_valid)
    y_valid_hat = (p_valid >= threshold).astype(int)
    print(
        f"holdout default-F1={best_f1:.6f}; threshold={threshold:.6f}; "
        f"precision={precision_score(y[valid_idx], y_valid_hat, zero_division=0):.6f}; "
        f"recall={recall_score(y[valid_idx], y_valid_hat, zero_division=0):.6f}"
    )

    final_model = make_pipeline(X, args.seed)
    final_model.fit(X, y)
    p_test = final_model.predict_proba(X_test)[:, 1]

    if args.released_test_target:
        released = pd.read_csv(args.released_test_target)
        if TARGET not in released:
            raise ValueError(f"Released test file must contain {TARGET!r}")
        p_eval = final_model.predict_proba(feature_frame(released, label=True).reindex(columns=X.columns))[:, 1]
        y_eval = default_target(released[TARGET])
        print(f"released-test default-F1 at holdout-selected threshold={f1_score(y_eval, p_eval >= threshold, zero_division=0):.6f}")
        released_id = id_column(released)
        if released_id and id_column(test) and not np.array_equal(released[released_id], test[id_column(test)]):
            print("warning: released-label rows are not in the same order as test.csv; evaluation was on the released file order")

    predicted_default = (p_test >= threshold).astype(int)
    output = pd.DataFrame()
    test_id = id_column(test)
    output["id"] = test[test_id] if test_id else np.arange(len(test))
    # The official target is 1 for paid back, so invert the tuned default label.
    output[TARGET] = 1 - predicted_default
    output.to_csv(args.output, index=False)
    print(f"wrote {len(output):,} rows to {args.output}")


if __name__ == "__main__":
    main()
