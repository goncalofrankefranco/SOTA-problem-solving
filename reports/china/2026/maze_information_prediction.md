# NOAI China 2026 — Maze Information Prediction

**Problem domain:** Grid topology, multi-output regression  
**Evaluation metric:** Four percentage-error sub-scores, summed for a maximum of 1.0. Each target contributes `0.2·exp(−MAPE) + 0.05·exp(−Max10PE)`, where `Max10PE` is the mean percentage error among the worst-scoring 10% of rows.

## Abridged statement

Each input is a partially observed four-connected 30×30 maze flattened into 900 characters. Known cells are start `S`, end `T`, open `.`, or wall `#`; `?` hides a cell that is either open or a wall in the completed maze. Predict the completed maze's wall count, number of open cells reachable from `S`, number of connected open regions, and shortest `S`-to-`T` path length. The true maze always connects `S` to `T`. This is a paraphrase; the Chinese statement is authoritative where translations differ.

## Dataset analysis and EDA

SOTA reports 5,000 labeled observed mazes and 5,000 corresponding target rows. Validation and test each contain 3,000 unlabelled mazes. The local workspace has no maze files, so no distribution plots, target ranges, unknown-cell proportions, or validation estimates were measured.

The unknown cells are not arbitrary missing numeric features: each is a hidden binary topology decision. The wall count is bounded by the number of observed walls and that count plus the number of unknowns. Treating unknowns as open gives the largest possible reachable set and a lower bound on shortest-path length; treating them as walls gives a smaller reachable set, while a known-open `S`–`T` path—if one exists—is a feasible route in the completed maze. Connected-component counts are not monotone under opening unknowns, because opening a cell can add an isolated region or merge existing regions; the two completions are therefore alternative topology features, not bounds. The four targets have different error distributions, so a single multi-output squared-error fit may overemphasize large-valued outputs.

## Experiments and solution strategy

No task-data experiment was run. The candidate in [the solution script](../../../solutions/china/2026/maze_information_prediction.py) extracts both the flattened cell layout and engineered graph features: counts of known/unknown/wall cells, row/column summaries, connected-component sizes under optimistic and pessimistic treatments, reachable-region sizes, and optimistic/pessimistic `S`–`T` path measurements. Four separate ExtraTrees regressors fit log-transformed targets, making the fit more responsive to relative error than an unscaled squared-error model. The script reports the competition score on a random 20% holdout when the supplied data are available.

With data access, compare the row-level split with maze-generation-aware splits if available, then test per-target model families and direct metric tuning. Check whether unknown placement and maze topology differ between train and leaderboard data. Any submission must preserve row order and output four finite values per maze.

## Score evidence and limits

| Result | Total score | Provenance |
|---|---:|---|
| This candidate | Not measured | Training/evaluation files and Bohrium grader are not available locally. |
| Organizer baseline B | 0.4508 | Reported by SOTA. |
| Scientific Committee reference B | 0.8653 | Reported by SOTA; not reproduced by this candidate. |

The reference score is an organizer benchmark, not a result from the repository's code. No full-score or optimality claim is made.

## Implementation and diagnostics

The candidate expects `/bohr/train-4mzz/v1/train_data.csv` and `train_answer.csv`, and uses `DATA_PATH` (default `/bohr/mazeval-7zx2/v1`) for `val_data.csv` and `test_data.csv`. `--mode validate` trains on an 80% random holdout split of the supplied training rows and prints the overall and per-target metric. `--mode submit` fits all supplied labels and writes `submission.zip` containing headerless `submission_val.csv` and `submission_test.csv`, four values per row in source order.

No organizer-environment runtime or actual score was available. The code's feature extraction includes exact BFS/component calculations for two completions of the observed maze; that is a diagnostic basis, not evidence that the learned predictions are accurate. The organizer baseline is a 900–4096–512–4 neural regressor trained with MSE.

## Compute and footprint

The proposed model is CPU-only and uses scikit-learn ExtraTrees, fitting one model per target. No pretrained weights or external data are required. The task limit is 25 minutes for training plus inference; runtime and memory were not measured in this environment.

## Alternatives to compare

- The released fully connected neural-network baseline on encoded maze cells.
- A direct physical estimator using unknown-cell probability and optimistic/pessimistic BFS bounds.
- Gradient boosting on engineered topology summaries and a separate model per target.
- Monte Carlo maze completions followed by exact target computation; use a validation-tuned completion model so the assumed open/wall probabilities match the observed data.
- Ensembles of raw-grid and graph-feature models selected on the actual percentage-error score.

## Progressive hints

1. Which target can be estimated from known cells plus the number of unknown cells?
2. How do the reachable set and shortest path change when every `?` is treated as open versus blocked?
3. What topological features distinguish many isolated openings from one large connected region?
4. Combine exact graph bounds with a separately validated regression model for each target, and tune it using the exponential percentage-error score.

**One-line solution:** Extract grid and connectivity bounds from both open/blocked interpretations of unknown cells, then fit per-target models for the four maze statistics.

## Sources and reuse

- [SOTA task page and English baseline translation](https://checklist.sota-ai.org/problems/noai-china-2026-round-2-maze-information-prediction/)
- The page links English/Chinese statements, baseline notebook, Bohrium task pages, and source files.

SOTA reports the source license as **not stated**. This report paraphrases the task and links to official materials; the maze data and organizer notebook are not redistributed.
