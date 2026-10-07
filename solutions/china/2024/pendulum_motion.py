"""PyTorch-initialized pendulum fitting for NOAI China 2024 task 4.

The method reconstructs the missing time grid, uses ``torch.linalg.lstsq`` on
finite differences for regression initialization, then refines the nonlinear
ODE fit with SciPy's ``least_squares`` and ``solve_ivp``. It reports candidates;
official hidden A/B recordings are required for the three-file submission run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


GRAVITY = 9.8


def find_gap(time: np.ndarray) -> tuple[int, int, float]:
    steps = np.diff(time)
    nominal = float(np.median(steps[steps < np.quantile(steps, 0.75) * 2]))
    indices = np.flatnonzero(steps > max(0.25, 5.0 * nominal))
    if len(indices) != 1:
        raise ValueError(f"Expected one missing interval; found {len(indices)}")
    left = int(indices[0])
    return left, left + 1, nominal


def reconstruct_grid(time: np.ndarray, nominal_step: float) -> tuple[np.ndarray, np.ndarray]:
    count = int(round((time[-1] - time[0]) / nominal_step)) + 1
    grid = np.linspace(time[0], time[-1], count, dtype=np.float64)
    indices = np.rint((time - time[0]) / (grid[1] - grid[0])).astype(np.int64)
    if np.max(np.abs(grid[indices] - time)) > 2e-4:
        raise ValueError("Recorded times are not aligned to a recoverable regular grid")
    return grid, indices


def torch_initial_guess(
    theta: np.ndarray,
    observed_indices: np.ndarray,
    grid: np.ndarray,
    left_index: int,
    right_index: int,
) -> tuple[float, float, float]:
    """Use local finite differences and torch.linalg.lstsq for an initial fit."""
    values = torch.as_tensor(theta, dtype=torch.float64)
    grid_step = float(grid[1] - grid[0])
    rows: list[list[float]] = []
    targets: list[float] = []
    segment_ids: list[int] = []
    for j in range(1, len(observed_indices) - 1):
        center = int(observed_indices[j])
        previous = int(observed_indices[j - 1])
        following = int(observed_indices[j + 1])
        if center - previous != 1 or following - center != 1:
            continue
        if center < left_index:
            segment = 0
        elif center > right_index:
            segment = 1
        else:
            continue
        velocity = (values[j + 1] - values[j - 1]) / (2.0 * grid_step)
        acceleration = (values[j + 1] - 2.0 * values[j] + values[j - 1]) / (grid_step**2)
        if segment == 0:
            rows.append([-float(velocity), -float(torch.sin(values[j])), 0.0])
        else:
            rows.append([-float(velocity), 0.0, -float(torch.sin(values[j]))])
        targets.append(float(acceleration))
        segment_ids.append(segment)
    if len(rows) < 10 or set(segment_ids) != {0, 1}:
        raise ValueError("Insufficient contiguous observations before and after the gap")
    design = torch.tensor(rows, dtype=torch.float64)
    target = torch.tensor(targets, dtype=torch.float64).unsqueeze(1)
    solution = torch.linalg.lstsq(design, target).solution[:, 0]
    mu = max(0.01, float(solution[0]))
    beta_before = max(0.1, float(solution[1]))
    beta_after = max(beta_before + 0.01, float(solution[2]))
    length = float(np.clip(GRAVITY / beta_before, 0.3, 40.0))
    force = float(np.clip(length * (beta_after - beta_before), 0.1, 1000.0))
    return length, mu, force


def simulate_observations(
    time: np.ndarray,
    initial_angle: float,
    length: float,
    damping: float,
    force: float,
    force_time: float,
) -> tuple[np.ndarray, float]:
    """Integrate before/after the unknown force switch at full precision."""
    gap_mask = time > force_time
    before_time = time[~gap_mask]
    after_time = time[gap_mask]

    def equation(beta: float):
        def derivative(_t: float, state: np.ndarray) -> tuple[float, float]:
            theta_value, omega_value = state
            return omega_value, -damping * omega_value - beta * np.sin(theta_value)

        return derivative

    beta_before = GRAVITY / length
    beta_after = GRAVITY / length + force / length
    before = solve_ivp(
        equation(beta_before),
        (float(time[0]), force_time),
        (initial_angle, 0.0),
        method="DOP853",
        t_eval=np.concatenate((before_time, [force_time])),
        rtol=1e-9,
        atol=1e-11,
        max_step=0.03,
    )
    state_at_force = before.y[:, -1]
    predicted = np.empty_like(time)
    predicted[~gap_mask] = before.y[0, :-1]
    after = solve_ivp(
        equation(beta_after),
        (force_time, float(time[-1])),
        state_at_force,
        method="DOP853",
        t_eval=after_time,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.03,
    )
    predicted[gap_mask] = after.y[0]
    return predicted, float(after.y[1, -1])


def next_zero_crossing(
    start_time: float,
    angle: float,
    velocity: float,
    length: float,
    damping: float,
    force: float,
) -> float:
    beta = GRAVITY / length + force / length

    def equation(_t: float, state: np.ndarray) -> tuple[float, float]:
        theta, omega = state
        return omega, -damping * omega - beta * np.sin(theta)

    def zero(_t: float, state: np.ndarray) -> float:
        return float(state[0])

    zero.terminal = True
    zero.direction = 0
    result = solve_ivp(
        equation,
        (start_time, start_time + 40.0),
        (angle, velocity),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        events=zero,
        max_step=0.01,
    )
    if len(result.t_events[0]) == 0:
        return float("nan")
    crossing = float(result.t_events[0][0])
    if crossing <= start_time + 1e-7:
        # If the last recorded angle is effectively zero, return the next crossing.
        result = solve_ivp(
            equation,
            (start_time + 1e-5, start_time + 40.0),
            result.y[:, -1],
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
            events=zero,
            max_step=0.01,
        )
        if len(result.t_events[0]) == 0:
            return float("nan")
        crossing = float(result.t_events[0][0])
    return crossing


def solve_recording(path: Path, *, max_nfev: int = 250) -> dict[str, float]:
    frame = pd.read_csv(path)
    if not {"t", "theta"}.issubset(frame.columns):
        raise ValueError(f"Expected t and theta columns in {path}")
    frame = frame.sort_values("t")
    time = frame["t"].to_numpy(dtype=np.float64)
    theta = frame["theta"].to_numpy(dtype=np.float64)
    left_observed, right_observed, nominal = find_gap(time)
    grid, observed_indices = reconstruct_grid(time, nominal)
    left = int(observed_indices[left_observed])
    right = int(observed_indices[right_observed])
    init_length, init_damping, init_force = torch_initial_guess(
        theta, observed_indices, grid, left, right
    )
    gap_left = float(grid[left])
    gap_right = float(grid[right])
    bounds = (
        [0.3, 0.0, 0.0, gap_left + 1e-7],
        [40.0, 20.0, 2000.0, gap_right - 1e-7],
    )
    best_fit = None
    for fraction in (0.2, 0.5, 0.8):
        initial = [init_length, init_damping, init_force, gap_left + fraction * (gap_right - gap_left)]

        def residual(parameters: np.ndarray) -> np.ndarray:
            prediction, _ = simulate_observations(
                time, float(theta[0]), *parameters
            )
            return prediction - theta

        fit = least_squares(
            residual,
            initial,
            bounds=bounds,
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        if best_fit is None or np.dot(fit.fun, fit.fun) < np.dot(best_fit.fun, best_fit.fun):
            best_fit = fit
    assert best_fit is not None
    length, damping, force, force_time = [float(p) for p in best_fit.x]
    predicted, velocity_end = simulate_observations(
        time, float(theta[0]), length, damping, force, force_time
    )
    residual_rmse = float(np.sqrt(np.mean((predicted - theta) ** 2)))
    zero_time = next_zero_crossing(
        float(time[-1]), float(theta[-1]), velocity_end, length, damping, force
    )
    return {
        "l": length,
        "miu": damping,
        "F": force,
        "t_nextzerotheta": zero_time,
        "t_Fput": force_time,
        "observed_angle_rmse": residual_rmse,
        "gap_start": gap_left,
        "gap_end": gap_right,
    }


OUTPUT_COLUMNS = ("l", "miu", "F", "t_nextzerotheta", "t_Fput")


def write_result_csv(result: dict[str, float], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{key: result[key] for key in OUTPUT_COLUMNS}]).to_csv(output, index=False)


def solve_batch(
    train_path: Path,
    test_a_path: Path,
    test_b_path: Path,
    output_dir: Path,
    *,
    max_nfev: int,
) -> None:
    """Fit the visible training trace and both hidden recordings into named CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = (
        ("submission_train.csv", train_path),
        ("submissionA.csv", test_a_path),
        ("submissionB.csv", test_b_path),
    )
    for filename, input_path in inputs:
        result = solve_recording(input_path, max_nfev=max_nfev)
        output = output_dir / filename
        write_result_csv(result, output)
        print(f"{filename}: { {key: round(result[key], 8) for key in OUTPUT_COLUMNS} }")
        print(f"saved {output}")
    archive_path = output_dir / "submission.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in ("submissionA.csv", "submissionB.csv"):
            archive.write(output_dir / filename, arcname=filename)
    print(f"saved {archive_path} (A/B CSVs; training CSV remains separate)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Single CSV containing one t/theta recording")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="Single-result CSV output")
    parser.add_argument("--train", type=Path, help="Training recording for the three-file submission mode")
    parser.add_argument("--test-a", type=Path, help="Hidden A recording CSV")
    parser.add_argument("--test-b", type=Path, help="Hidden B recording CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("submission"), help="Directory for named batch CSVs")
    parser.add_argument("--max-nfev", type=int, default=250, help="Maximum solver evaluations per initial event-time guess")
    args = parser.parse_args()
    torch.set_num_threads(1)
    batch_paths = (args.train, args.test_a, args.test_b)
    if any(path is not None for path in batch_paths):
        if not all(path is not None for path in batch_paths):
            parser.error("batch mode requires all of --train, --test-a, and --test-b")
        solve_batch(*batch_paths, args.output_dir, max_nfev=args.max_nfev)
    else:
        if args.input is None:
            parser.error("provide --input for one recording or all three batch paths")
        result = solve_recording(args.input, max_nfev=args.max_nfev)
        write_result_csv(result, args.output)
        print("candidate:", {k: round(v, 8) for k, v in result.items()})
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
