# NOAI Singapore 2026 — Gradient Boosting from Scratch

**Problem domain:** Classical ML, supervised regression, gradient boosting

**Evaluation metric:** 25 marks total: 8 for gradients and initial predictions, 12 for the estimator and fitting algorithm, plus a 5-mark early-stopping bonus. The notebook also prints MSE on synthetic data, but no public leaderboard score or hidden-grader result is available.

## Abridged statement

Implement a gradient-boosting regressor for L1 and L2 losses. Compute the loss-specific negative gradients and constant starting predictions, then build an estimator that fits a decision tree to each round's negative gradient and adds its learning-rate-scaled predictions. Add a validation-based early-stopping estimator for the bonus.

## Data analysis

There is no released environmental-sensor dataset or held-out evaluation split. The notebook checks functions with random 3-by-5 arrays and demonstrates fitting on `make_regression` data with 1,000 rows, 10 features, and noise 10, split into training and test subsets. These are synthetic examples rather than evidence about generalisation to a separate task dataset. The useful analysis is mathematical: L1 uses `sign(y - prediction)` and starts at the target median; L2 uses the residual and starts at the target mean.

## Experiments

The local implementation was checked on synthetic arrays for both negative gradients, both initial predictions, and fit/predict with each loss. The recorded assertions passed. The early-stopping API was also exercised; its simple train/validation case retained all 30 requested trees, so that run did not verify that patience actually truncates the ensemble. The official notebooks were inspected for formulas and expected behavior but were not executed as a reference comparison.

## Solution

[The implementation](../../../solutions/singapore/2026/gradient_boosting.py) uses NumPy and `sklearn.tree.DecisionTreeRegressor`. It initializes the ensemble with the median or mean, fits each next tree to the current negative gradient, updates the training predictions by the learning-rate-scaled tree output, and adds each tree contribution during prediction. `GBREarlyStop` checks validation loss after each round and retains the best tree prefix.

## Score evidence and limits

**No mark score was measured.** The official maximum is 25 marks, but no organizer grader result or released hidden tests were available. Passing local functional assertions is evidence that the exercised code paths run; it does not establish a mark total. In particular, early-stopping truncation remains unverified by the recorded simple example.

## Implementation and diagnostics

The implementation follows the notebook's stated gradients: `np.sign(y - y_pred)` for L1, and `y - y_pred` for L2. Its constant initial predictions are respectively `median(y)` and `mean(y)`. Each boosting round uses the current residual-like target and updates predictions without refitting prior trees. The bonus estimator evaluates validation MAE for L1 or MSE for L2 and stops after the configured patience.

**Rules and dependencies:** The task calls for NumPy and decision trees and rules out XGBoost and LightGBM. The solution uses NumPy, scikit-learn's `DecisionTreeRegressor`, and Python's standard `abc` module; it does not call a packaged gradient-boosting estimator.

## Compute and footprint

The local synthetic checks used CPU. No runtime benchmark, memory benchmark, or organizer-environment result was recorded. No GPU, pretrained weights, or external data are needed.

## Alternatives considered

- `sklearn.ensemble.GradientBoostingRegressor`, XGBoost, or LightGBM would bypass the requested from-scratch boosting logic or violate the stated library restriction.
- Stochastic gradient boosting is supported through row subsampling, but it does not replace the required basic fitting loop.
- L1 tree-leaf line search can use median corrections in some boosting formulations; the released task asks for trees fitted to the specified pseudo-gradients, so the implementation follows that stated loop.

## Progressive hints

1. Choose the loss-specific negative gradient and best constant starting value before fitting any trees.
2. At each round, fit the next tree to the current negative gradient, not directly to the original target.
3. Add the tree prediction multiplied by the learning rate and keep the tree for later predictions.
4. For early stopping, track the best validation-loss round and return that tree prefix.

**One-line summary:** Start from the median or mean, fit each tree to the current L1 or L2 negative gradient, and retain the validation-best prefix for early stopping.

## Sources and reuse

- [SOTA task summary](https://checklist.sota-ai.org/problems/noai-singapore-2026-final-gradient-boosting-from-scratch/)
- [Official NOAI 2026 Programming Task 1 notebook](https://aisingapore.org/wp-content/uploads/question_1_v20022026.ipynb)
- [Released solution notebook](https://github.com/AISGNUSNOAI/NOAI-2026/blob/main/question_1_solution.ipynb)
- [Official NOAI learning-resources page](https://noai.aisingapore.org/learning-resources/)

The source does not state a reuse license. This report paraphrases the task and the solution code is independently written; the notebooks are linked rather than copied into this repository.
