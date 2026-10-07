"""Feature-based candidate for NOAI China 2026 Task 4.

Reads the official CSV assets when available and writes the required zip.  The
organizer's files are not included in this repository.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

N = 30
CELL_ID = {".": 0, "#": 1, "?": 2, "S": 3, "T": 4}


def read_mazes(path: str | Path) -> list[str]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            value = line.strip().strip('"')
            if not value or value.lower() in {"maze", "data"}:
                continue
            if "," in value:
                fields = value.split(",")
                value = "".join(field.strip().strip('"') for field in fields)
            if len(value) != N * N:
                continue
            if any(ch not in CELL_ID for ch in value):
                raise ValueError(f"Unexpected maze symbol in {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No {N}x{N} mazes found in {path}")
    return rows


def read_targets(path: str | Path) -> np.ndarray:
    y = np.loadtxt(path, delimiter=",", dtype=np.float64, encoding="utf-8-sig")
    y = np.atleast_2d(y)
    if y.shape[1] != 4:
        raise ValueError(f"Expected four target columns in {path}, got {y.shape}")
    return y


def _bfs(mask: np.ndarray, start: tuple[int, int]) -> tuple[np.ndarray, int]:
    dist = np.full((N, N), -1, dtype=np.int16)
    r0, c0 = start
    if not mask[r0, c0]:
        return dist, 0
    queue = [(r0, c0)]
    dist[r0, c0] = 0
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < N and 0 <= nc < N and mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist, len(queue)


def _components(mask: np.ndarray) -> tuple[int, list[int]]:
    seen = np.zeros((N, N), dtype=bool)
    sizes: list[int] = []
    for r in range(N):
        for c in range(N):
            if not mask[r, c] or seen[r, c]:
                continue
            _, size = _bfs_unseen(mask, seen, (r, c))
            sizes.append(size)
    return len(sizes), sizes


def _bfs_unseen(mask: np.ndarray, seen: np.ndarray, start: tuple[int, int]) -> tuple[np.ndarray, int]:
    queue = [start]
    seen[start] = True
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < N and 0 <= nc < N and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return np.empty(0), len(queue)


def maze_features(maze: str) -> np.ndarray:
    """Encode cell layout and exact topology bounds implied by observed cells."""
    grid = np.asarray([CELL_ID[ch] for ch in maze], dtype=np.int8).reshape(N, N)
    walls, unknown = grid == 1, grid == 2
    known_open = (grid == 0) | (grid == 3) | (grid == 4)
    possible_open = known_open | unknown
    s_pos = tuple(np.argwhere(grid == 3)[0])
    t_pos = tuple(np.argwhere(grid == 4)[0])

    s_known, reachable_known = _bfs(known_open, s_pos)
    s_possible, reachable_possible = _bfs(possible_open, s_pos)
    t_known, _ = _bfs(known_open, t_pos)
    t_possible, _ = _bfs(possible_open, t_pos)
    known_components, known_sizes = _components(known_open)
    possible_components, possible_sizes = _components(possible_open)

    path_known = int(s_known[t_pos])
    path_possible = int(s_possible[t_pos])
    known_path_exists = path_known >= 0
    path_possible_exists = path_possible >= 0
    path_unknowns = 0
    if path_possible_exists:
        # Count unknowns lying on at least one shortest optimistic path.
        shortest_corridor = (s_possible >= 0) & (t_possible >= 0)
        shortest_corridor &= (s_possible + t_possible == path_possible)
        path_unknowns = int(np.count_nonzero(shortest_corridor & unknown))

    features: list[float] = [
        float(walls.sum()), float(unknown.sum()), float(known_open.sum()),
        float(reachable_known), float(reachable_possible),
        float(known_components), float(possible_components),
        float(max(known_sizes, default=0)), float(np.mean(known_sizes) if known_sizes else 0),
        float(max(possible_sizes, default=0)), float(np.mean(possible_sizes) if possible_sizes else 0),
        float(path_known if known_path_exists else 2 * N * N),
        float(path_possible if path_possible_exists else 2 * N * N),
        float(known_path_exists), float(path_possible_exists), float(path_unknowns),
        float(np.count_nonzero(unknown & (s_possible >= 0))),
        float(np.count_nonzero(unknown & (s_known >= 0))),
        float(np.count_nonzero(unknown & (t_possible >= 0))),
        float(np.count_nonzero(unknown & (t_known >= 0))),
        float(s_pos[0]), float(s_pos[1]), float(t_pos[0]), float(t_pos[1]),
    ]
    # Cell-state pattern keeps spatial information that global counts discard.
    features.extend(grid.astype(np.float32).ravel().tolist())
    # Row/column summaries expose corridors, borders and concentrated missingness.
    for state_mask in (walls, unknown, known_open):
        features.extend(state_mask.sum(axis=1).astype(float).tolist())
        features.extend(state_mask.sum(axis=0).astype(float).tolist())
    # Distances from S and T summarize connectivity without making the features
    # depend on a particular choice for each unknown cell.
    for dist in (s_known, s_possible, t_known, t_possible):
        finite = dist[dist >= 0]
        features.extend([
            float(finite.min()) if finite.size else 0.0,
            float(finite.mean()) if finite.size else 0.0,
            float(finite.max()) if finite.size else 0.0,
            float(np.count_nonzero(dist < 0)),
        ])
    return np.asarray(features, dtype=np.float32)


def feature_matrix(mazes: list[str]) -> np.ndarray:
    return np.vstack([maze_features(maze) for maze in mazes])


def fit_models(x: np.ndarray, y: np.ndarray, random_state: int = 42):
    models = []
    for target in range(4):
        # A log target makes the fit focus more closely on relative error,
        # which is what the competition's percentage-error metric rewards.
        model = ExtraTreesRegressor(
            n_estimators=350,
            max_features=0.8,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state + target,
        )
        model.fit(x, np.log1p(np.maximum(y[:, target], 0.0)))
        models.append(model)
    return models


def predict(models, x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.expm1(model.predict(x)) for model in models]).clip(0.0)


def competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    scores = []
    for col in range(4):
        ape = np.abs(y_pred[:, col] - y_true[:, col]) / np.maximum(np.abs(y_true[:, col]), 1e-12)
        tail = np.mean(np.sort(ape)[-max(1, int(np.ceil(0.1 * len(ape)))):])
        scores.append(float(0.2 * np.exp(-np.mean(ape)) + 0.05 * np.exp(-tail)))
    return float(sum(scores)), scores


def validate(train_x_path: str, train_y_path: str) -> None:
    mazes, y = read_mazes(train_x_path), read_targets(train_y_path)
    if len(mazes) != len(y):
        raise ValueError(f"Mazes and labels differ: {len(mazes)} vs {len(y)}")
    x = feature_matrix(mazes)
    train_i, val_i = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    models = fit_models(x[train_i], y[train_i])
    pred = predict(models, x[val_i])
    total, components = competition_score(y[val_i], pred)
    print(f"random holdout score={total:.6f}; target scores={components}")


def submit(train_x_path: str, train_y_path: str, data_dir: str, output_dir: str) -> None:
    mazes, y = read_mazes(train_x_path), read_targets(train_y_path)
    if len(mazes) != len(y):
        raise ValueError(f"Mazes and labels differ: {len(mazes)} vs {len(y)}")
    models = fit_models(feature_matrix(mazes), y)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, filename in (("val", "val_data.csv"), ("test", "test_data.csv")):
        hidden = read_mazes(Path(data_dir) / filename)
        pred = predict(models, feature_matrix(hidden))
        np.savetxt(out / f"submission_{split}.csv", pred, delimiter=",", fmt="%.6f")
    with zipfile.ZipFile(out / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(out / "submission_val.csv", "submission_val.csv")
        archive.write(out / "submission_test.csv", "submission_test.csv")
    print(f"Wrote {out / 'submission.zip'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "submit"), default="validate")
    parser.add_argument("--train-data", default="/bohr/train-4mzz/v1/train_data.csv")
    parser.add_argument("--train-answer", default="/bohr/train-4mzz/v1/train_answer.csv")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_PATH", "/bohr/mazeval-7zx2/v1"))
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    if args.mode == "validate":
        validate(args.train_data, args.train_answer)
    else:
        submit(args.train_data, args.train_answer, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
