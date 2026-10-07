# NOAI China 2025 — Chemical Reaction Kinetics Simulation

**Problem domain:** Tabular regression with concentration-time reaction traces  
**Metric:** Mean of `max(0, 1 - ln(1 + 0.1 × |prediction − target|) / 5)` over experiments  
**Official task page:** [SOTA checklist: Chemical Reaction Kinetics Simulation](https://checklist.sota-ai.org/problems/noai-china-2025-round-2-chemical-reaction-kinetics/) · [Bohrium task](https://www.bohrium.com/en/competitions/25136176824)

## Abridged task

Predict the half-life `t1/2` of reactant L2M from the initial concentrations of L2M, D and L. The reaction includes reversible and irreversible elementary steps. The released training package contains traces for 1,000 simulated experiments and a summary table; each hidden validation/test experiment provides only the three initial concentrations. The validation set has 100 experiments and the hidden test set has 412. The expected archive contains `submission_val.csv` and `submission_test.csv`, each with experiment number and `t12` columns. The official statement linked from the checklist controls exact column names and rules.

## Data and license

The NOAI training traces and validation/test inputs were not mounted in this workspace. The task page lists the organizer source license as **not stated**. The linked statement and baseline notebook were not retrievable from this environment, and the Bohrium answer files are protected by the competition runtime. No organizer data or notebook is copied here.

## Dataset analysis and EDA

The task summary describes 1,000 simulated training experiments, with three initial concentrations and a reaction trace for each experiment. The hidden validation and test tables contain 100 and 412 experiments, respectively, and expose only the initial concentrations. Since the tables were unavailable locally, I could not inspect concentration ranges, target skew, trace shapes, missing values, or outliers. The column positions below follow the linked baseline's description and still need confirmation against the organizer files.

## Method

The candidate in [chemical_reaction_kinetics.py](../../../solutions/china/2025/chemical_reaction_kinetics.py) uses three tree regressors blended together: Extra Trees, Random Forest, and an absolute-error HistGradientBoosting model. Features include raw and log concentrations, concentration ratios, pairwise products, reactant imbalance, limiting-reactant concentration, and an inhibitor fraction. This captures nonlinear interactions while remaining practical for the 1,000-row training set.

The script reads `training_data.dat` and selects input columns 1–3 and target column 5, matching the baseline notebook's positional description. A random 80/20 split reports the competition score, MAE and median absolute error. Submit mode predicts the validation and test tables and writes the expected CSVs and ZIP.

## Experiments and score evidence

| Result | Score | Provenance |
|---|---:|---|
| NOAI validation leaderboard | Not measured | Bohrium credentials and the encrypted validation data were unavailable. |
| NOAI hidden test leaderboard | Not measured | Hidden labels and final evaluation were inaccessible. |
| Local candidate holdout | Not measured | The official training table was not available locally. |

No organizer baseline or scientific-committee score is reported as this candidate's result. The linked Bohrium competition window is listed as ending on 20 June 2026, so this candidate was not submitted to its A/B leaderboard.

## Implementation, diagnostics, and compute

The script uses CPU scikit-learn tree models, with parallel tree fitting enabled through `n_jobs=-1`. The competition tables were not available, so there is no measured runtime, memory footprint, holdout result, or output-format check against the organizer's helper. The positional parser and expected CSV names are candidate assumptions, not verified facts; confirm them from the official statement and sample files before running.

## Alternatives considered

- Fit a mechanistic rate law or reaction ODE directly, which could extrapolate more naturally from concentration inputs.
- Regress on a log-transformed half-life and compare it with the raw-target blend.
- Compare repeated cross-validation and simpler gradient-boosted trees using the exact clipped logarithmic score.

## Run

```bash
python solutions/china/2025/chemical_reaction_kinetics.py \
  --mode validate --train-table /path/to/training_data.dat

python solutions/china/2025/chemical_reaction_kinetics.py \
  --mode submit --train-table /path/to/training_data.dat \
  --val-table /path/to/val_data_question.dat \
  --test-table /path/to/test_data_question.dat \
  --output-dir submission
```

Requires NumPy, pandas and scikit-learn. The code does not download the organizer data or attempt to read `ANSWER_PATH`; files are supplied explicitly.

## Limitations and next steps

- Check that the actual summary file uses the stated column positions and delimiter before running; I could not verify the organizer statement or table headers locally.
- Compare the ensemble with a direct kinetic-law fit and log-target models using repeated splits. With only 1,000 experiments, report split variance as well as the mean.
- Evaluate the exact requested file formatting and score against the public validation grader if Bohrium access becomes available. Do not select the model on the hidden test.

## Progressive hints

1. Identify which initial concentrations correspond to the reactant, donor, and inhibitor before fitting a model.
2. Inspect concentration ranges and how the target changes with each concentration; the reaction is nonlinear.
3. Add ratios, products, and limiting-reactant features, then compare tree models with a kinetic-law fit.
4. Select models using the clipped logarithmic score from the statement, not MAE alone.

**One-line solution:** Blend tree regressors on concentration and kinetic-interaction features; the candidate and both NOAI leaderboard scores remain unmeasured until the organizer data are available.

## Sources and reuse

- [NOAI task page, metric, data sizes and submission format](https://checklist.sota-ai.org/problems/noai-china-2025-round-2-chemical-reaction-kinetics/) — source license not stated.
- [Official IOAI article confirming the 2 June 2025 NOAI finals used four tasks and A/B evaluation](https://ioai-official.org/noai-china-finals-2025-8-team-members-selected-for-chinas-national-team-for-ioai-2025/).
