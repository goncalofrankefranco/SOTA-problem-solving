# NOAI Singapore 2025 — Linear Regression Using Gradient Descent

**Problem domain:** Classical ML, supervised regression, numerical optimization  
**Evaluation metric:** Return the coefficient vector rounded to four decimal places; the task is worth 15 marks. No public automated grader or hidden test set was found.

## Abridged statement

Implement batch gradient descent for linear regression. Given a feature matrix that already includes an intercept column, a target vector, a learning rate, and an iteration count, initialize the coefficients at zero and repeatedly update them using the mean error gradient. Return the coefficient vector rounded to four decimals.

## Data analysis

The released example has three rows and two columns: a constant intercept column and a single feature with values 2, 4, and 6. Targets are 2, 4, and 6, so the underlying relation is a zero intercept and unit slope. The example is tiny and deterministic; it has no separate training, validation, or test data and no missing values.

## Experiments

I used the provided `alpha = 0.01`, `iterations = 1000`, and zero initialization. The result was compared directly to the official Section 2 solution notebook:

| Method | Coefficients |
|---|---|
| Local implementation | `[0.0030, 0.9987]` |
| Official solution notebook | `[0.0030, 0.9987]` |

The small nonzero intercept and slope error are the expected finite-iteration result, not a coding discrepancy.

## Solution

The implementation is [linear_regression_gradient_descent.py](../../../solutions/singapore/2025/linear_regression_gradient_descent.py). It computes `errors = X @ theta - y`, applies `theta -= alpha * X.T @ errors / m`, and rounds only the returned vector.

## Score evidence and limits

**Measured result:** Exact match to the official notebook's output on the only released example. The task carries 15 marks, but there is no published grader or additional scoring examples. This is evidence for the expected answer, not an organizer-issued `15/15` result.

## Implementation and diagnostics

The batch gradient is vectorized with NumPy and supports either a one-dimensional target or a column-shaped target. The intercept is not added internally because the statement says it is already part of `X`. The local exact-output comparison passed.

## Compute and footprint

This is a handful of small matrix operations on CPU. Across the released-example checks for all four 2025 Singapore solutions, the measured in-process runtime was about `0.013 s`; no GPU or external package beyond NumPy is needed.

## Alternatives considered

- The normal equation can solve this tiny example directly, but it would not follow the requested gradient-descent implementation.
- Stochastic or mini-batch updates change the specified batch update rule.
- More iterations would move coefficients closer to `[0, 1]`, but would not reproduce the released example's specified 1,000-step output.

## Progressive hints

1. Keep the intercept as a column of ones supplied in `X`.
2. Compute predictions for every row at once, then take the mean gradient over all rows.
3. Initialize all coefficients to zero and update them simultaneously from the old parameter vector.
4. Apply the four-decimal rounding after the requested number of iterations.

**One-line summary:** Batch gradient descent returns **`[0.0030, 0.9987]`**, exactly matching the released reference example.

## Sources and reuse

- [SOTA task summary](https://checklist.sota-ai.org/problems/noai-singapore-2025-final-linear-regression-gradient-descent/)
- [Official NOAI 2025 programming questions and solution notebooks](https://github.com/AISGNUSNOAI/NOAI-2025)
- [Official NOAI learning-resources page](https://noai.aisingapore.org/learning-resources/)

The source page does not state a license. The linked task calls for Python 3.9 standard library and NumPy. This report paraphrases the task and the implementation is independently written; the official notebook is not copied into this repository.
