"""Sim-to-real sequence regression candidate for NOAI China 2026 Task 2.

The official trajectory CSV is supplied by the task environment and is not
included here. This script learns corrections from the visible real prefix and
the complete simulated trajectory, then writes only the hidden rows.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

REQUIRED = {"index", "task_index", "frame_index", "timestamp", "simulation_positions", "observation_state"}


def parse_state(value) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "[]", "{}"}:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, (list, tuple, np.ndarray)):
            array = np.asarray(parsed, dtype=np.float64).reshape(-1)
        else:
            array = np.asarray([float(token) for token in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", text)], dtype=np.float64)
    if array.size != 6 or not np.isfinite(array).all():
        return None
    return array


def load_frame_table(path: str | Path, training: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    frame = frame.copy()
    frame["_sim"] = frame["simulation_positions"].map(parse_state)
    frame["_obs"] = frame["observation_state"].map(parse_state)
    if frame["_sim"].isna().any():
        raise ValueError(f"Could not parse simulation_positions in {path}")
    if training and frame["_obs"].isna().any():
        raise ValueError(f"Training file has missing observation_state values: {path}")
    frame["frame_index"] = pd.to_numeric(frame["frame_index"], errors="raise")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise")
    return frame


def task_arrays(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = group.sort_values(["frame_index", "timestamp"], kind="stable")
    sim = np.stack(ordered["_sim"].to_numpy())
    obs = np.stack([x if x is not None else np.full(6, np.nan) for x in ordered["_obs"]])
    frame = ordered["frame_index"].to_numpy(dtype=np.float64)
    times = ordered["timestamp"].to_numpy(dtype=np.float64)
    return ordered, sim, obs, np.column_stack([frame, times])


def _prefix_stats(sim: np.ndarray, obs: np.ndarray, prefix_n: int, time: np.ndarray) -> np.ndarray:
    residual = obs[:prefix_n] - sim[:prefix_n]
    t = time[:prefix_n]
    t0, t1 = float(t[0]), float(t[-1])
    scale = max(t1 - t0, 1e-8)
    u = (t - t0) / scale
    centered = u - u.mean()
    denom = float(np.dot(centered, centered)) + 1e-8
    slope = (centered[:, None] * (residual - residual.mean(axis=0))).sum(axis=0) / denom
    return np.concatenate([
        residual[0], residual[-1], residual.mean(axis=0), residual.std(axis=0), slope,
        np.asarray([prefix_n / max(1, len(sim)), t0, t1, scale]),
    ])


def build_features(sim: np.ndarray, prefix_obs: np.ndarray, prefix_n: int, frame: np.ndarray, timestamp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(sim)
    if not 0 < prefix_n < n:
        raise ValueError(f"Invalid visible prefix length {prefix_n} for {n} frames")
    # Finite differences capture local simulated motion; timestamps are used
    # rather than assuming uniform frame spacing.
    dt = np.maximum(np.gradient(timestamp), 1e-6)
    velocity = np.gradient(sim, axis=0) / dt[:, None]
    acceleration = np.gradient(velocity, axis=0) / dt[:, None]
    t0, t1 = float(timestamp[0]), float(timestamp[-1])
    time_scale = max(t1 - t0, 1e-8)
    t_norm = (timestamp - t0) / time_scale
    prefix_t = timestamp[prefix_n - 1]
    since_prefix = np.maximum(timestamp - prefix_t, 0.0) / time_scale
    stats = _prefix_stats(sim, prefix_obs, prefix_n, timestamp)
    prefix_sim_last = sim[prefix_n - 1]
    features = np.column_stack([
        sim, velocity, acceleration,
        np.repeat(t_norm[:, None], 1, axis=1),
        np.repeat(since_prefix[:, None], 1, axis=1),
        sim - prefix_sim_last,
        np.repeat(stats[None, :], n, axis=0),
    ]).astype(np.float32)
    return features[prefix_n:], features


def make_training_examples(frame: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_parts, y_parts, group_parts = [], [], []
    for task_id, group in frame.groupby("task_index", sort=False):
        ordered, sim, obs, _ = task_arrays(group)
        n = len(ordered)
        if n < 5:
            continue
        # Validation and test reveal about 30% of each trajectory. Jitter the
        # synthetic cut slightly so the fit is not tied to one exact boundary.
        visible_fraction = float(rng.uniform(0.27, 0.33))
        cut = int(np.clip(round(n * visible_fraction), 2, n - 1))
        hidden_features, _ = build_features(sim, obs, cut, ordered["frame_index"].to_numpy(dtype=float), ordered["timestamp"].to_numpy(dtype=float))
        x_parts.append(hidden_features)
        y_parts.append(obs[cut:] - sim[cut:])
        group_parts.append(np.full(n - cut, task_id))
    if not x_parts:
        raise ValueError("No usable training trajectories")
    return np.vstack(x_parts), np.vstack(y_parts).astype(np.float32), np.concatenate(group_parts)


def fit_models(x: np.ndarray, residual: np.ndarray, seed: int = 42):
    models = []
    for dimension in range(6):
        estimator = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.07, max_iter=140,
            max_leaf_nodes=19, min_samples_leaf=35, l2_regularization=3.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
            random_state=seed + dimension,
        )
        estimator.fit(x, residual[:, dimension])
        models.append(estimator)
    return models


def predict_hidden_rows(frame: pd.DataFrame, models) -> pd.DataFrame:
    outputs = []
    for _, group in frame.groupby("task_index", sort=False):
        ordered, sim, obs, _ = task_arrays(group)
        observed = np.isfinite(obs).all(axis=1)
        # The task exposes a visible prefix. Stop at the first hidden row so a
        # malformed later observation cannot leak information into predictions.
        prefix_n = 0
        while prefix_n < len(ordered) and observed[prefix_n]:
            prefix_n += 1
        if prefix_n == len(ordered):
            continue
        if prefix_n < 2:
            raise ValueError("Each trajectory needs at least two visible real states")
        hidden_features, _ = build_features(sim, obs, prefix_n, ordered["frame_index"].to_numpy(dtype=float), ordered["timestamp"].to_numpy(dtype=float))
        residual_pred = np.column_stack([model.predict(hidden_features) for model in models])
        state_pred = sim[prefix_n:] + residual_pred
        for row_idx, state in zip(range(prefix_n, len(ordered)), state_pred):
            row = ordered.iloc[row_idx]
            outputs.append({
                "index": row["index"],
                "task_index": row["task_index"],
                "frame_index": row["frame_index"],
                "timestamp": row["timestamp"],
                "observation_state": "[" + ", ".join(f"{float(v):.8f}" for v in state) + "]",
            })
    result = pd.DataFrame(outputs, columns=["index", "task_index", "frame_index", "timestamp", "observation_state"])
    input_order = {str(value): i for i, value in enumerate(frame["index"].tolist())}
    result["_input_order"] = result["index"].map(lambda value: input_order.get(str(value), len(input_order)))
    return result.sort_values("_input_order", kind="stable").drop(columns="_input_order").reset_index(drop=True)


def organizer_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
    return float(np.mean(np.exp(-10.0 * rmse)))


def validate(path: str | Path) -> None:
    frame = load_frame_table(path, training=True)
    x, residual, groups = make_training_examples(frame)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_rows, val_rows = next(split.split(x, residual, groups))
    models = fit_models(x[train_rows], residual[train_rows])
    # The constructed examples retain only task ids, so score the held-out
    # residual rows against the corresponding targets directly.
    pred_residual = np.column_stack([model.predict(x[val_rows]) for model in models])
    score = organizer_score(residual[val_rows], pred_residual)
    print(f"grouped held-out mean row score={score:.6f}")


def resolve_split_csv(data_dir: str | Path, split: str, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    root = Path(data_dir)
    names = ("val.csv", "validation.csv", "val_data.csv", "validation_data.csv") if split == "val" else ("test.csv", "test_data.csv")
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    tokens = ("val", "valid") if split == "val" else ("test",)
    candidates = [p for p in root.glob("*.csv") if any(token in p.stem.lower() for token in tokens)]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not identify the {split} CSV in {root}; pass --{split}-csv explicitly")


def submit(train_path: str | Path, data_dir: str | Path, output_dir: str | Path, val_path: str | None = None, test_path: str | None = None) -> None:
    train = load_frame_table(train_path, training=True)
    x, residual, _ = make_training_examples(train)
    models = fit_models(x, residual)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, explicit in (("val", val_path), ("test", test_path)):
        frame = load_frame_table(resolve_split_csv(data_dir, split, explicit), training=False)
        predictions = predict_hidden_rows(frame, models)
        predictions.to_csv(output_dir / f"submission_{split}.csv", index=False)
        print(f"{split}: wrote {len(predictions)} hidden rows")
    with zipfile.ZipFile(output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output_dir / "submission_val.csv", "submission_val.csv")
        archive.write(output_dir / "submission_test.csv", "submission_test.csv")
    print(f"Wrote {output_dir / 'submission.zip'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "submit"), default="submit")
    parser.add_argument("--train-csv", default=os.environ.get("TRAIN_CSV", "train.csv"))
    parser.add_argument("--data-dir", default=os.environ.get("DATA_PATH", "."))
    parser.add_argument("--val-csv", default=os.environ.get("VAL_CSV"))
    parser.add_argument("--test-csv", default=os.environ.get("TEST_CSV"))
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    if args.mode == "validate":
        validate(args.train_csv)
    else:
        submit(args.train_csv, args.data_dir, args.output_dir, args.val_csv, args.test_csv)


if __name__ == "__main__":
    main()
