# NOAI China 2026 — Embodied Intelligence Sim2Real State Prediction

**Problem domain:** Time series, multivariate sequence regression, sim-to-real adaptation  
**Evaluation metric:** Mean per-hidden-frame score `exp(-10 × RMSE)` over six joint-state coordinates; maximum 1.0.

## Abridged statement

For each robot trajectory, use the complete simulated six-joint position sequence and the visible prefix of real robot states to predict the real states for every remaining hidden frame. The task supplies timestamps and trajectory/frame identifiers. This is a paraphrase of the SOTA summary; the linked official statement controls exact details.

## Dataset analysis and EDA

SOTA reports 336 fully labeled training trajectories with 176,675 rows. Validation has 72 trajectories and 35,136 rows, of which 24,562 need predictions; test has 72 trajectories and 37,632 rows, of which 26,310 need predictions. The observed prefix is about 30% of the validation/test rows in aggregate. The task page does not state the exact CSV filenames or publish the files outside its Bohrium links. Neither the data nor an authenticated Bohrium workspace is available locally, so trajectory plots, per-joint residual analysis, and local validation scores are unavailable.

The important structure is that every row has both a simulation state and time, but real states are visible only at the start of evaluation trajectories. A single global simulator-to-robot mapping may leave trajectory-specific offset or drift. The visible prefix can estimate these residual patterns, while frame velocity and acceleration provide local motion context.

## Experiments and solution strategy

No task-data experiments were run. The candidate in [the solution script](../../../solutions/china/2026/isaac_sim_state_prediction.py) creates pseudo-hidden prefixes on the fully labeled training trajectories, learns each joint's residual (`real − simulated`) using a histogram gradient-boosting regressor, and predicts hidden rows from:

- The simulated joint positions, finite-difference velocities, and accelerations.
- Normalized trajectory time and time since the observed prefix ends.
- Prefix residual statistics for each joint: first/last value, mean, spread, and fitted trend.
- The current simulated state relative to the final observed simulated state.

The script includes a group-held-out validation mode using the official per-row score formula. The model settings have not been tuned against task data. After the files are available, compare residual boosting with per-trajectory affine correction, ridge regression, retrieval of similar training trajectories, and blends; select using grouped validation so frames from one trajectory never appear in both sides.

## Score evidence and limits

| Result | Mean row score | Provenance |
|---|---:|---|
| This candidate | Not measured | No training/validation files or Bohrium scoring access in this workspace. |
| Organizer baseline B | 0.5647 | Reported by SOTA. |
| Scientific Committee reference B | 0.7861 | Reported by SOTA; not reproduced here. |

These reference numbers are organizer-reported benchmark values, not results from this candidate. No score or optimality claim is made.

## Implementation and diagnostics

Run `python solutions/china/2026/isaac_sim_state_prediction.py --mode validate --train-csv /path/to/train.csv` to get a grouped holdout estimate once the official training CSV is available. For submission, set `TRAIN_CSV` or pass `--train-csv`, set `DATA_PATH` to the evaluation directory, and run `--mode submit`; if the split file names are not `val.csv` and `test.csv`, pass `--val-csv` and `--test-csv` (or set `VAL_CSV`/`TEST_CSV`). The script writes `submission.zip` with `submission_val.csv` and `submission_test.csv`, each containing exactly the hidden rows and the five required columns: index, task index, frame index, timestamp, and a six-number state array.

The SOTA page lists CPU-only execution, a 25-minute train/inference limit, no external data, no LLM API, no internet or package installation, and a prohibition on accessing hidden labels. The implementation trains only from the supplied complete training trajectories and uses only the visible prefix from validation/test inputs. Since exact file paths are not stated on the SOTA page, the training file path must be supplied explicitly; validation/test resolution also needs confirmation against the official workspace.

## Compute and footprint

The proposed fit uses six `HistGradientBoostingRegressor` models on CPU. The reported training set has roughly 177,000 frames; no runtime or memory measurement is available here. The solution requires NumPy, pandas, and scikit-learn, and bundles no data or model checkpoint.

## Alternatives to compare

- Per-trajectory ridge or robust affine mapping from simulated to real joint states, estimated on the visible prefix.
- A simple residual offset/trend correction; it is inexpensive and useful as a baseline.
- Retrieval of similar fully labeled trajectories, as explicitly suggested by the task summary.
- A sequence model or dynamic time warping if repeated motion patterns are present; it must respect the CPU and time budget.
- A validation-tuned ensemble of local prefix correction and a global model.

## Progressive hints

1. Compute the error between simulated and real states on training trajectories.
2. Do errors vary by joint, trajectory, or time?
3. How can the visible real prefix estimate the current trajectory's offset and drift?
4. Train and validate by trajectory, then compare a global residual model with prefix-based corrections and similar-trajectory retrieval.

**One-line solution:** Predict the simulation-to-real residual from the simulated motion and the visible prefix, then add that residual to each hidden simulated state.

## Sources and reuse

- [SOTA task page and official links](https://checklist.sota-ai.org/problems/noai-china-2026-round-2-sim2real-state-prediction/)
- The page links English/Chinese statements, the official baseline notebook, Bohrium task pages, and source files.

SOTA reports the source license as **not stated**. This report paraphrases the task and links to the official materials; no dataset or notebook is copied.
