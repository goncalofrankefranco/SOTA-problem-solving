# Poland 2026 Stage 2 — Drzewa decyzyjne

**Problem name:** Drzewa decyzyjne (Decision Trees)  
**Problem domain:** Classical ML, geometric preprocessing, model selection  
**Evaluation metric:** Number of datasets on which a decision tree matches or beats the supplied baseline accuracy; 1.2 points per dataset, capped at 100  
**Abridged statement:** For each of many independent two-dimensional binary classification datasets, preprocess the labeled training points and unlabeled test points, choose decision-tree hyperparameters, and predict the test labels. The released collection A is available for development; collection B is used for the hidden evaluation.

## Dataset analysis and EDA

Collection A contains 100 separate datasets. Each has two numeric features and binary labels. Training-set sizes range from 74 to 2,043 points (median 677.5); the minority class is 25%–50% of each training set. The supplied baseline accuracies range from 0.572 to 0.976, with a median of 0.933.

Scatterplots show substantial variation in scale, orientation, and class geometry. Examples include diagonal bands, clusters, and overlapping or noisy boundaries. Many patterns are poorly aligned with the original feature axes, so a tree that only makes axis-aligned splits in raw coordinates can underperform. The two features remain the full input; preprocessing must preserve both matrix shapes.

**EDA performed:** point-cloud scatterplots, class-balance and sample-size summaries, feature covariance/eigenvalue inspection, and score comparisons for raw coordinates, whitening, ICA-style rotations, PCA, polar and polynomial maps, and spectral embeddings.

## Experiments

| Approach | Collection A result | Notes |
|---|---:|---|
| Starter notebook: pooled whitening, moment-based ICA rotation, repeated-CV pruning | 49/100 wins; 58.8 points; 88.54% mean accuracy | Reproduced from the released notebook implementation. |
| Train-only, 16-bin mutual-information rotation with the original CV policy | 51/100 wins; 61.2 points; 88.68% mean accuracy | Improved over the starter, but later replaced. |
| Pooled, 24-bin mutual-information rotation; global alpha prior | 54/100 wins; 64.8 points | CV tolerance 0.001 and fixed prior 0.0015. |
| Pooled, 24-bin rotation; prior conditioned on training-only best-CV alpha | **55/100 wins; 66.0 points; 88.70% mean accuracy** | Selected implementation below. |
| Expanded CV grid over pruning strength and minimum leaf size | 48/100 wins; 57.6 points | More hyperparameter choices did not improve the released A score. |
| Spectral embedding, sample of 34 datasets | 0/34 wins with 8 or 12 neighbors; 2/34 with 20 neighbors | Tested and discarded; this is a subset result, not a full-A score. |
| PCA, polar, and simple polynomial coordinate maps | Did not improve the selected approach | Tested as alternatives; not used in the final pipeline. |

The final preprocessing rotation minimizes a discretized mutual-information estimate between the two whitened projections. It uses 24 equal-frequency bins per projection and searches rotations from 0° through 179°. Both whitening and the rotation objective use pooled train and test features without labels; the same transform is applied to both arrays.

The tree searches `ccp_alpha` values from 0.0015 to 0.0045 by repeated stratified cross-validation (up to four folds, two repeats). It finds the best mean CV alpha, then sets one global prior: 0.0015 when the CV-best alpha is at most 0.003, otherwise 0.00375. Among candidates within 0.1 percentage points of the best mean CV accuracy, it chooses the alpha closest to that prior. This condition uses only the current dataset’s training labels and CV scores.

## Solution and validation

Implementation: [decision_trees.py](../solutions/stage2/decision_trees.py)

On the 100 released A datasets, the final functions beat or match `baseline_acc` on 55 datasets, for **66.0/100 points**. Mean test accuracy is 88.70%. The official shape assertions passed: preprocessed training and test arrays retain their input shapes.

Each dataset’s tree hyperparameters are chosen using only its training labels and stratified cross-validation. The unlabeled test features are used only with training features to estimate the shared whitening transform and rotation. The released A output labels were used to compare global solution variants during development, so 66.0/100 is a development/validation score, not an untouched estimate of collection B performance. Per-dataset oracle choices made after inspecting test labels were excluded from the final code and score. A diagnostic leave-one-family-out assessment of the prior policy scored 54/100 (64.8 points); family IDs were used only to create that diagnostic split, never by the solution.

**Full credit was not reached.** Collection B is hidden and its score is unknown. The high and varied baseline accuracies suggest the selected single-tree interface cannot match every geometry with one simple preprocessing strategy.

## Compute and footprint

The solution uses CPU NumPy and scikit-learn only; it requires no GPU or external model. Preprocessing searches 180 two-dimensional rotations per dataset, followed by small decision-tree CV fits. It preserves the task’s two-feature input and output contract.

## Hints

1. Plot each dataset before choosing a tree strategy; the task is a collection of different geometries, not one shared distribution.
2. Check feature scale, covariance, and orientation. Can a data-driven rotation make the tree’s axis-aligned splits more useful?
3. Use the unlabeled test points to stabilize transductive feature preprocessing, while keeping their labels out of training and model selection.
4. Select pruning strength with stratified CV, and be cautious about tiny CV-score differences on small datasets.

**TL;DR:** Symmetrically whiten pooled features, rotate them to reduce 24-bin histogram-estimated mutual information, then select pruning by repeated stratified CV with a prior based on the CV-best alpha; measured result: **66.0/100 on released collection A**.
