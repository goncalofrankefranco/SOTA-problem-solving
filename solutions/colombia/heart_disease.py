#!/usr/bin/env python3
"""F1-oriented solution for Colombia AI Olympiad 2025 Heart Disease Risk."""

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

TARGET = "heart_disease_risk"
ID_CANDIDATES = ("id", "ID", "patient_id")


def make_pipeline(frame: pd.DataFrame, seed: int) -> Pipeline:
    categorical = list(frame.select_dtypes(include=["object", "category", "bool"]).columns)
    # Numeric sex/smoking flags are nominal and should not acquire a false order.
    for column in frame.columns:
        name = column.lower()
        if any(token in name for token in ("gender", "sex", "smok")) and column not in categorical:
            categorical.append(column)
    numeric = [column for column in frame.columns if column not in categorical]

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
        max_iter=400,
        max_leaf_nodes=31,
        l2_regularization=1.5,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=seed,
    )
    return Pipeline([("features", transform), ("model", classifier)])


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if thresholds.size == 0:
        return 0.5, float(f1_score(y_true, probability >= 0.5, zero_division=0))
    values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.argmax(values))
    return float(thresholds[index]), float(values[index])


def id_column(frame: pd.DataFrame) -> str | None:
    return next((name for name in ID_CANDIDATES if name in frame.columns), None)


def features(frame: pd.DataFrame, labeled: bool) -> pd.DataFrame:
    drop = [name for name in (id_column(frame), TARGET if labeled else None) if name is not None]
    return frame.drop(columns=drop).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--released-test-target", type=Path, help="Optional labeled test file for post-contest evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    if TARGET not in train:
        raise ValueError(f"Training data must contain {TARGET!r}")
    y = pd.to_numeric(train[TARGET], errors="raise").astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0/1 labels")
    X = features(train, labeled=True)
    X_test = features(test, labeled=False).reindex(columns=X.columns)
    print(f"train={len(train):,}; test={len(test):,}; features={X.shape[1]}; high-risk={y.mean():.3%}")
    print(f"numeric columns={list(X.select_dtypes(include=np.number).columns)}")
    print(f"categorical columns={list(X.select_dtypes(include=['object', 'category', 'bool']).columns)}")

    train_idx, valid_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=args.seed, stratify=y
    )
    validation_model = make_pipeline(X.iloc[train_idx], args.seed)
    validation_model.fit(X.iloc[train_idx], y[train_idx])
    p_valid = validation_model.predict_proba(X.iloc[valid_idx])[:, 1]
    threshold, tuned_f1 = best_f1_threshold(y[valid_idx], p_valid)
    y_valid_hat = p_valid >= threshold
    print(
        f"holdout-F1={tuned_f1:.6f}; threshold={threshold:.6f}; "
        f"precision={precision_score(y[valid_idx], y_valid_hat, zero_division=0):.6f}; "
        f"recall={recall_score(y[valid_idx], y_valid_hat, zero_division=0):.6f}"
    )

    model = make_pipeline(X, args.seed)
    model.fit(X, y)

    if args.released_test_target:
        released = pd.read_csv(args.released_test_target)
        if TARGET not in released:
            raise ValueError(f"Released test file must contain {TARGET!r}")
        p_released = model.predict_proba(features(released, labeled=True).reindex(columns=X.columns))[:, 1]
        y_released = pd.to_numeric(released[TARGET], errors="raise").astype(int).to_numpy()
        print(f"released-test F1 at holdout-selected threshold={f1_score(y_released, p_released >= threshold, zero_division=0):.6f}")

    p_test = model.predict_proba(X_test)[:, 1]
    output = pd.DataFrame()
    test_id = id_column(test)
    output["id"] = test[test_id] if test_id else np.arange(len(test))
    output[TARGET] = (p_test >= threshold).astype(int)
    output.to_csv(args.output, index=False)
    print(f"wrote {len(output):,} rows to {args.output}")


if __name__ == "__main__":
    main()
