"""Predict organometallic reaction half-life from initial concentrations.

The competition data stay external. Point --train-table at the organizer's
training_data.dat and pass validation/test question tables in submit mode.
"""
from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split


EPS = 1e-12


def read_table(path: Path) -> pd.DataFrame:
    """Read the organizer's comma/tab/space-delimited .dat tables."""
    try:
        raw = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
    except Exception as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc
    if raw.empty:
        raise ValueError(f"Empty data table: {path}")
    first_is_data = pd.to_numeric(raw.iloc[0], errors="coerce").notna().all()
    if first_is_data:
        return raw
    return raw.iloc[1:].reset_index(drop=True)


def select_inputs(frame: pd.DataFrame) -> np.ndarray:
    if frame.shape[1] < 4:
        raise ValueError(f"Expected experiment id and three concentrations, got {frame.shape[1]} columns")
    values = frame.iloc[:, 1:4].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Initial concentrations contain missing or non-numeric values")
    return values


def features(x: np.ndarray) -> pd.DataFrame:
    """Simple kinetic features for reactant, donor, and inhibitor concentrations."""
    x = np.asarray(x, dtype=np.float64)
    a, d, l = x[:, 0], x[:, 1], x[:, 2]
    cols: dict[str, np.ndarray] = {}
    for i, v in enumerate((a, d, l)):
        cols[f"c{i}"] = v
        cols[f"log_c{i}"] = np.log(np.maximum(v, 0) + EPS)
        cols[f"sqrt_c{i}"] = np.sqrt(np.maximum(v, 0))
    for name, v in {
        "a_over_d": a / (d + EPS), "a_over_l": a / (l + EPS),
        "d_over_a": d / (a + EPS), "d_over_l": d / (l + EPS),
        "l_over_a": l / (a + EPS), "l_over_d": l / (d + EPS),
        "a_times_d": a * d, "a_times_l": a * l, "d_times_l": d * l,
        "a_d_over_l": a * d / (l + EPS), "a_l_over_d": a * l / (d + EPS),
        "d_l_over_a": d * l / (a + EPS), "min_reactants": np.minimum(a, d),
        "reactant_sum": a + d, "all_sum": a + d + l,
        "stoich_imbalance": np.abs(a - d) / (a + d + EPS),
        "inhibitor_fraction": l / (a + d + l + EPS),
    }.items():
        cols[name] = v
        if name.endswith("_over_l") or name.endswith("_over_d") or name.endswith("_over_a"):
            cols[f"log_{name}"] = np.log(np.maximum(v, 0) + EPS)
    return pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_models(seed: int):
    return [
        ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, max_features=0.9, n_jobs=-1, random_state=seed),
        RandomForestRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.9, n_jobs=-1, random_state=seed + 1),
        HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=250, max_leaf_nodes=12,
                                      min_samples_leaf=12, l2_regularization=1.0, random_state=seed + 2),
    ]


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray, seed: int = 17) -> np.ndarray:
    xt, xp = features(x_train), features(x_pred)
    preds = []
    for model in make_models(seed):
        model.fit(xt, y_train)
        preds.append(np.asarray(model.predict(xp), dtype=np.float64))
    # Trees preserve local neighborhoods; the absolute-error booster estimates
    # the central conditional response. A blend reduces variance at n=1,000.
    pred = 0.4 * preds[0] + 0.3 * preds[1] + 0.3 * preds[2]
    return np.maximum(pred, 0.0)


def official_score(pred: np.ndarray, true: np.ndarray) -> float:
    error = np.abs(np.asarray(pred) - np.asarray(true))
    return float(np.mean(np.maximum(0.0, 1.0 - np.log1p(0.1 * error) / 5.0)))


def write_submission(pred: np.ndarray, path: Path) -> None:
    frame = pd.DataFrame({"Experiment Number": np.arange(len(pred)), "t12": [f"{v:.4e}" for v in pred]})
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "submit"), default="validate")
    parser.add_argument("--train-table", type=Path, required=True, help="Organizer training_data.dat")
    parser.add_argument("--val-table", type=Path)
    parser.add_argument("--test-table", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("submission"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train = read_table(args.train_table)
    if train.shape[1] < 6:
        raise ValueError("Expected training table columns with t12 at zero-based column 5")
    x = select_inputs(train)
    y = pd.to_numeric(train.iloc[:, 5], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(y)
    x, y = x[valid], y[valid]
    if args.mode == "validate":
        ix, iv = train_test_split(np.arange(len(y)), test_size=0.2, random_state=args.seed)
        pred = fit_predict(x[ix], y[ix], x[iv], args.seed)
        error = np.abs(pred - y[iv])
        print(f"heldout={len(iv)} mean_score={official_score(pred, y[iv]):.6f} mae={error.mean():.6g} median_ae={np.median(error):.6g}")
        return
    if args.val_table is None or args.test_table is None:
        raise SystemExit("submit mode requires --val-table and --test-table")
    val_x, test_x = select_inputs(read_table(args.val_table)), select_inputs(read_table(args.test_table))
    val_pred = fit_predict(x, y, val_x, args.seed)
    test_pred = fit_predict(x, y, test_x, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_csv, test_csv = args.output_dir / "submission_val.csv", args.output_dir / "submission_test.csv"
    write_submission(val_pred, val_csv)
    write_submission(test_pred, test_csv)
    with zipfile.ZipFile(args.output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(val_csv, val_csv.name)
        zf.write(test_csv, test_csv.name)
    print(f"wrote {args.output_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
