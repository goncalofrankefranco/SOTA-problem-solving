# Heart Disease Risk Prediction — Colombia AI Olympiad 2025, Problem 1

**Problem domain:** Tabular machine learning, binary classification  
**Official metric:** F1 score

## Abridged statement

Given synthetic patient attributes, predict whether the patient has high five-year heart-disease risk. The target column is `heart_disease_risk` (`1` high, `0` low). The [SOTA task page](https://checklist.sota-ai.org/problems/colombia-ai-olympiad-2025-problem-1-heart-disease-risk/) describes a Kaggle task titled “Colombian AI Olympiad 2025 (Pr1) Heart Attack.”

## Dataset analysis and EDA

The released competition files are `train.csv`, `test.csv`, and `sample_submission.csv`. The task summary describes a Python-generated synthetic dataset inspired by dependencies in the Framingham Heart Study. It names age, gender, smoking, cholesterol and yearly cholesterol change, stress level, and diet quality; it says roughly 40% of samples are high risk. The described dependency structure includes gender affecting smoking, smoking affecting cholesterol and cholesterol change, and age affecting baseline cholesterol. Some categories are ordinal and distributions overlap.

Exact dataset size, feature spellings beyond the target, category levels, missingness, and correlations were not measured here because the Kaggle data endpoint returned no downloadable contents in this environment. The solution prints the actual row count, class balance, numeric columns, and categorical columns once the files are available.

## Experiments

| Experiment | Result |
|---|---|
| Stratified 80/20 validation with F1 threshold tuning | Not run: competition files could not be downloaded |
| Released/Kaggle hidden evaluation | Not submitted; no organizer score available here |

## Chosen method

[The solution script](../../solutions/colombia/heart_disease.py) uses a scikit-learn `HistGradientBoostingClassifier` with median imputation, one-hot encoding for string categories and nominal gender/smoking fields, and a deterministic stratified 80/20 validation split. It selects the probability threshold that maximizes high-risk F1 on the validation fold, refits on all labeled rows, and writes `id,heart_disease_risk` predictions.

This is a compact CPU model for nonlinear interactions in mixed tabular features. The threshold is chosen against F1 rather than accuracy, which is the organizer's stated metric. Ordinal variables that are supplied as numeric levels remain ordered numeric features; string-valued levels are one-hot encoded.

## Score and provenance

**Measured score: none.** F1 is the organizer-reported metric. The local score and Kaggle score are unknown because neither the competition data nor a released labeled test file was retrievable from this execution environment. No perfect score is claimed.

## Implementation and diagnostics

The script prints train/test sizes, target prevalence, feature dtypes, validation threshold, precision, recall, and F1. It drops the ID from model features, aligns test columns to the training schema, and writes IDs in test order. It can optionally evaluate a released labeled test CSV with `--released-test-target`; such a measurement should be labeled post-contest and must not be confused with a live Kaggle score.

The dataset is synthetic, so feature dependencies may be stronger and more structured than in clinical records. A simple train/validation split estimates performance on the same synthetic generator; it does not establish clinical validity.

## Compute and footprint

The selected model runs on CPU. No runtime, memory, or hardware measurement was possible without the data. It does not require a GPU, outside medical data, or pretrained weights.

## Alternatives considered

- Logistic regression is interpretable but may miss the stated nonlinear and cascading dependencies.
- ExtraTrees or random forests provide useful baselines for synthetic interactions but were not measured.
- CatBoost/LightGBM can be strong on mixed tabular features and may be worth testing if available; no comparison was possible here.
- Maximizing accuracy or using the default 0.5 threshold may sacrifice the F1 score.

These are candidate experiments rather than results.

## Progressive hints

1. Identify the exact target, ID, and submission columns from the sample submission before fitting.
2. Check the approximately 40% high-risk prevalence and distinguish nominal from ordinal categories.
3. Use stratified validation and optimize the decision threshold for F1; report precision and recall alongside it.
4. Fit the selected model on all labeled rows and preserve test ID order in the submission.

**One-line summary:** Tune a mixed-feature boosted classifier for high-risk F1; code is ready, while score and data diagnostics remain unverified because Kaggle files were inaccessible.

## Sources and reuse

- [SOTA task page, statement summary and rules](https://checklist.sota-ai.org/problems/colombia-ai-olympiad-2025-problem-1-heart-disease-risk/)
- [Official Kaggle competition](https://www.kaggle.com/competitions/colombian-ai-olympiad-2025-pr-1-heart-attack)

The SOTA page says reuse is subject to Kaggle competition rules. This repository links the competition and does not copy its data or statement files.
