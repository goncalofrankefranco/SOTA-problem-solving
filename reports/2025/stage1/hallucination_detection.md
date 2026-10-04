# Poland 2025 Stage I — Wykrywanie Halucynacji (Hallucination Detection)

**Domain:** NLP uncertainty estimation and binary classification  
**Metric:** ROC AUC. The scoring rule gives 0 points at AUC ≤0.70 and 100 points at AUC ≥0.82.  
**Official sources:** [SOTA checklist](https://checklist.sota-ai.org/) · [starter notebook](https://github.com/OlimpiadaAI/II-OlimpiadaAI/tree/main/1_etap/2_wykrywanie_halucynacji) · [organizer worked solution](https://github.com/OlimpiadaAI/II-OlimpiadaAI/blob/main/1_etap/2_wykrywanie_halucynacji/2_wykrywanie_halucynacji_modelowe_rozwiazanie.ipynb)

## Abridged statement

Classify whether an LLM's answer to a factual question is correct. Each record contains the question and main answer, generated tokens, four higher-temperature supporting answers, their tokens and token probabilities, and a trusted `is_correct` label. The final evaluation runs on CPU within five minutes using the permitted tabular libraries.

## Dataset analysis and EDA

The release has 2,967 training records and 990 validation records, with eight columns and no null fields. Main answers average 19.44 words in train and 19.46 in validation; the four supporting answers average 20.34 words in both. Labels are imbalanced, with roughly twice as many incorrect answers as correct answers. The organizer's data checks found one malformed answer tag in train, none in validation, no empty tagged answers, and 439 training records whose supporting token lists cannot reconstruct the supporting answer. A `\n\n` inside one token list precedes unrelated generated text and misaligned probabilities; the worked notebook trims this tail before feature extraction.

## Experiments

The worked notebook explores six feature groups and an XGBoost classifier:

- Token-probability statistics: minima, means, maxima, standard deviations, lengths, variance, and generalized negative log likelihood, both for answer spans and for each supporting response.
- Semantic features: TF-IDF, question/answer similarity, and lexical overlap.
- Cross-answer agreement: number of unique extracted answers, count of the modal answer, and agreement ratio.
- Style and structure: answer length, sentence-length variation, and longest-common-subsequence similarity.
- Question type: regex flags for who/what/where/when/why/how/which.
- Data cleanup: remove malformed tags and truncate the bad token/probability tails described above.

The organizer reports AUC 0.7117 using only statistical features, 0.7417 semantic, 0.7277 cross-answer consistency, 0.6141 style, 0.5987 question type, and 0.8178 answer-probability features. A random baseline is 0.5000. Excluding answer-probability features from the full set falls to 0.7775; other single-group ablations range from 0.8129 to 0.8221. The all-feature XGBoost model with early stopping scores 0.8226. A separate three-fold search over 256 parameter sets (768 fits) reports best parameters `max_depth=3`, `n_estimators=500`, `learning_rate=0.01`, `subsample=0.5`, `colsample_bytree=0.8`, `min_child_weight=2`, `gamma=0.1` and validation AUC 0.8263.

## Selected solution and validation evidence

The standalone implementation in [hallucination_detection.py](../../../solutions/2025/stage1/hallucination_detection.py) reworks the organizer's six feature groups and the best grid-searched XGBoost configuration. It fits TF-IDF on training text and fits the classifier only on training labels; validation labels are used only for the final metric. The selected tree count and hyperparameters came from three-fold cross-validation on the training set. It requires the task-permitted `xgboost` package.

The organizer's selected all-feature XGBoost implementation reports **AUC 0.8226**, above the full-credit threshold, for **100/100 organizer-reported points**. The separate searched model reports 0.8263. The notebook also reports the most important features as the number of nonmatching answer spans, number of matching answer spans, answer probability standard deviation, other-answer probability standard deviation, and unique extracted answer count.

These scores are copied from the organizer's executed notebook outputs; they were not independently rerun. `train.json` and `valid.json` are hosted only on Google Drive. Attempts to reach those links failed with `HTTP 403: CONNECT tunnel failed`, and the files are absent from the official GitHub repository and local mirrors. The local Python environment also lacks `xgboost`, so no local score is available. No hidden labels or secret-test claims are used.

## Compute and runtime

The official limit is five minutes on CPU. The executed notebook records 768 three-fold parameter-search fits but no total wall-clock time. The final prediction is a compact XGBoost model over engineered features; feature-extraction and inference timings are not reported.

## Alternatives

The notebook measures each feature family alone and removes each family from the full feature set. The answer-probability group is the strongest single family; removing it causes the largest ablation drop. A random classifier provides the 0.50 baseline. External semantic encoders are not used in the selected solution.

## Progressive hints

1. Compare token probabilities for the answer spans across the four alternative generations.
2. Measure whether the four generations agree on the extracted factual answer.
3. Clean malformed token tails before calculating probability and span features.
4. Combine probability, semantic, answer-agreement, style, and question-type features in a small tabular classifier.

**One-line summary:** An all-feature XGBoost model reaches organizer-reported AUC 0.8226 (100/100) after cleaning corrupted alternative-token tails and engineering answer-span probability features.
