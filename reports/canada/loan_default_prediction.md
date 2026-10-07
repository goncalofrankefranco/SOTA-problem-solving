# Applied Problem Solving: Loan Default Prediction — CAIO 2025

**Problem domain:** Tabular machine learning, imbalanced binary classification  
**Official metric:** F1 score for the default class (`loan_paid_back = 0`) on 5,000 undisclosed rows selected from a 20,000-row test set; code quality is also considered.

## Abridged statement

Predict whether a loan is repaid using borrower and loan information. The released notebook describes 100,000 labeled training applications and 20,000 test applications. `loan_paid_back` is 1 for repayment and 0 for default; approximately 80% of training examples are repaid. The contest instructions prohibit external data. See the [SOTA task summary](https://checklist.sota-ai.org/problems/caio-canada-2025-national-qualifier-loan-default-prediction/) and [organizer's 2025 materials](https://iaiocanada.com/prepare/).

## Dataset analysis and EDA

The official summary describes 12 predictors: annual income, debt-to-income ratio, credit score, loan amount, interest rate, gender, marital status, education level, employment status, loan purpose, grade, and subgrade. The grade is A–F and the subgrade is 1–5. The target is imbalanced toward repayment (roughly 4:1).

The organizer published `train.csv`, `test.csv`, and a post-contest `test_with_target.csv`. Those files are linked from the SOTA task page and CAIO page, but Google Drive requests from this execution environment lead to sign-in or are denied. Therefore row-level missingness, category counts, outliers, duplicate IDs, and feature/target distributions were not measured here. The script prints sample counts, class balance, and categorical columns once run against the official files.

## Experiments

| Experiment | Dataset / validation | Result |
|---|---|---|
| Stratified 80/20 validation with threshold tuning | Not run: source CSV could not be downloaded | No score |
| Post-contest released-test evaluation | Not run: `test_with_target.csv` could not be downloaded | No score |

No leaderboard or organizer score is claimed. In particular, the 100/100 maximum possible task score is not evidence that this implementation achieved full credit.

## Chosen method

[The solution script](../../solutions/canada/loan_default.py) uses a scikit-learn pipeline with median imputation for numeric columns, one-hot encoding for categorical columns, and `HistGradientBoostingClassifier`. It maps the official label to `default = 1`, selects a probability threshold that maximizes default-class F1 on a stratified 20% holdout, then fits on all released training rows and writes binary `loan_paid_back` predictions for the test IDs.

Threshold selection matters because F1 is evaluated on defaults, the minority class; a generic 0.5 cutoff is not guaranteed to maximize that score. The script can optionally evaluate against post-contest `test_with_target.csv`, but that file is never included in fitting.

## Score and provenance

**Measured score: none.** The metric and split design are organizer/checklist-reported. No local score was possible because the official notebook and CSV files were inaccessible from this execution environment. To reproduce a score, run the script on the source files; pass `--released-test-target` to evaluate on the published target-bearing test file. If that file contains all 20,000 labels, its full-file F1 is a post-contest local measurement, not necessarily the official score on the organizer's undisclosed 5,000-row subset.

## Implementation and diagnostics

The implementation is CPU-compatible, deterministic under the selected seed, keeps the held-out rows out of training during threshold selection, and uses no external data. It reports holdout F1, precision, recall, threshold, feature count, and class balance. No prediction file or trained model was generated here. The official notebook's baseline uses standardization, one-hot encoding, and a class-balanced decision tree; the source description says accuracy/code quality were general applied-round criteria, while this task's stated target score is default-class F1.

## Compute and footprint

The chosen scikit-learn model is intended for a CPU and a 100,000-row, 12-feature dataset. Runtime and peak memory were not measured because no CSV was available. The pipeline and requirements are small; it does not need a GPU or pretrained weights.

## Alternatives considered

- The organizer's class-balanced decision tree is a direct baseline but can be unstable and may leave F1 below a boosted-tree model.
- Logistic regression would be a useful interpretable baseline after one-hot encoding, but may underfit nonlinear income, credit, and grade interactions.
- CatBoost or LightGBM could be strong for mixed tabular data, but add dependencies and were not empirically compared.
- Returning the majority repayment class would score poorly on default F1.

These are methodological expectations, not measured comparisons on the unavailable data.

## Progressive hints

1. Read the metric carefully: the positive class for the score is default, even though the source target uses 0 for default.
2. Inspect the class ratio and feature types; treat grades, subgrades, and borrower categories as categorical.
3. Use a stratified holdout and tune the threshold for default F1 rather than assuming 0.5.
4. After choosing the threshold and settings, refit on all 100,000 labeled rows and emit predictions in the exact test ID order.

**One-line summary:** Tune a mixed-feature boosted classifier for default-class F1; the runnable solution is ready, but a score is unverified because the official data could not be retrieved.

## Sources and reuse

- [SOTA task page, including data links and task rules](https://checklist.sota-ai.org/problems/caio-canada-2025-national-qualifier-loan-default-prediction/)
- [CAIO Prepare page](https://iaiocanada.com/prepare/)
- [Official Round 2 Colab notebook](https://colab.research.google.com/drive/1oaOCrbiVDurUUoPLtVQ3WS_hYTUuShdL?usp=sharing)
- [Official Round 2 Drive folder](https://drive.google.com/drive/folders/1mCuUNUscTnuHO8tEKHsX8RKfdSM66DsL?usp=drive_link)
- [Post-contest `test_with_target.csv`](https://drive.google.com/file/d/1yhYJTqx6jZX3KVP-cdS8Ct7st6ijhW6J/view?usp=sharing)

The source page does not state a data license. The report links the organizer materials and does not copy data or notebook contents. The contest prohibits external data and AI assistance during the live exam; this implementation uses only the official feature files for its intended historical practice workflow.
