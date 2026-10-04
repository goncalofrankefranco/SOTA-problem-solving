# Pruning

## Task and metric

- **Domain:** regression and neural-network compression.
- **Metric:** `(1 - MSE/1000)^1.5 * sparsity^1.5`; score above 0.95 earns the maximum 1.5 points.
- **Abridged statement:** keep the exact 128–1024–Sigmoid–10 MLP architecture, set as many of its weights and biases to zero as possible, and retain low MSE. The notebook allows up to five minutes on Colab GPU.

## EDA

The released training arrays contain 8,000 examples of 128 features and 10 continuous targets; released validation has 2,000 examples. Features are standardized (mean near 0, standard deviation near 1), while target standard deviation is about 177.7. A linear least-squares fit on training data gives MSE 0.9804 on train and 1.0165 on released validation, indicating that the target function is almost linear. The validation labels were used only to compute the final reported metric, never for fitting weights.

## Experiments and measured score

| Method | Validation MSE | Sparsity | Score |
|---|---:|---:|---:|
| Cached organizer example pruning (100 first-layer zeros per row and 256 output zeros per row) | 321.071 | 0.7374 | 0.3542 |
| Sparse MLP trained from scratch, 2 active hidden units | 4,572.72 | 0.9980 | 0.0000 |
| Sparse MLP trained from scratch, 4 active hidden units | 1,860.25 | 0.9960 | 0.0000 |
| Sparse MLP trained from scratch, 6 active hidden units | 623.53 | 0.9941 | 0.2289 |
| Sparse MLP trained from scratch, 8 active hidden units | 277.896 | 0.9921 | 0.6064 |
| Sparse MLP trained from scratch, 10 active hidden units | 226.46 | 0.9902 | 0.6703 |
| Sparse MLP trained from scratch, 12 active hidden units | 202.74 | 0.9882 | 0.6993 |
| Sparse MLP trained from scratch, 16 active hidden units | 196.268 | 0.9843 | 0.7037 |
| Sparse MLP trained from scratch, 24 active hidden units | 203.53 | 0.9765 | 0.6859 |
| Sparse MLP trained from scratch, 32 active hidden units | 201.42 | 0.9687 | 0.6804 |
| Sparse MLP trained from scratch, 48 active hidden units | 201.29 | 0.9531 | 0.6641 |
| Sparse MLP trained from scratch, 64 active hidden units | 163.849 | 0.9374 | 0.6940 |
| Dense Adam run, best observed checkpoint around epoch 50 | 11.420 | 0.0000 | 0.0000 |
| Rank-9 linear fit encoded as paired sigmoid units (`c=1e-4`) | **1.1895** | **0.9825** | **0.9721** |

The full sparse-from-scratch sweep over 2–64 active units showed poor accuracy at very small widths and a score peak near 16 units. The dense Adam run reached validation MSE 11.42 at epoch 50, then became unstable under the selected learning rate; it motivated inspecting the simpler data-generating relationship rather than serving as the final model.

The final result was also checked by the cloned official `validation_script.py` against the released validation arrays: **0.9721 score, 1.5/1.5 points**. The script passed with the generated model parameters. Validation labels were used only by the evaluator for scoring, not in fitting the coefficients.

## Method and runtime

Fit a ten-output linear regression on training arrays only, then truncate the training coefficient matrix to rank 9 by SVD. Each retained linear component is represented by two hidden units with opposite small input weights: `sigmoid(c·t) - sigmoid(-c·t) ≈ c·t/2`. Opposite output weights scale the difference back to the linear component. All unused hidden units and biases are set exactly to zero. This leaves 2,494 nonzero parameters out of 141,322. The fit and 128×10 SVD complete in a few seconds on the local CPU; the generation run, including importing PyTorch in the local cache environment, took about 3.5 seconds. Inference is the original MLP.

## Alternatives considered

Magnitude pruning alone scored 0.3542. Sparse training with only 8–64 active units did not approximate the function as well. Dense training could fit the near-linear function but was wasteful and unstable at the tested learning rate. The paired-sigmoid encoding uses the mandated activation while exploiting the observed low-rank linear structure.

## Progressive hints

1. Inspect target scale and compare a train-only linear regression baseline.
2. Use the singular spectrum of the fitted 128×10 coefficient matrix to estimate effective rank.
3. Approximate a linear hidden response with a small-slope sigmoid pair of opposite signs.
4. Keep the validation labels out of coefficient fitting; compute the official score only after saving the final weights.

**One-line summary:** a train-only rank-9 regression encoded into paired sigmoid units scores 0.9721 on released validation and reaches full points.
