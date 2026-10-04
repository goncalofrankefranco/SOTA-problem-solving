# Kredytobranie — Taking Out a Loan

**Problem domain:** Counterfactual explanations for a two dimensional credit classifier  
**Official metric:** `100 × V × (D + P) / 2`, rounded to an integer. `V` scales validity from 0 at 50% to 1 at 100%; `P` scales the share above the supplied class-1 KDE threshold the same way; `D` is 1 when mean L2 change is below 0.22 and falls linearly to 0 at 0.30.

## Abridged statement

For each rejected application, return a nearby point that the supplied classifier assigns to class 1 and that has sufficient probability under the supplied class-1 density model. The function may use the explanation points and the two supplied models; the official notebook explicitly prohibits access to `X_train` and `y_train`.

## Data analysis

- Official data: 800 training points (416 class 1, 384 class 0) and 118 explanation points. The two features are in `[0, 1]` for training; explanation features range from about `-0.033` to `0.957`.
- Only 25 of 118 explanation points initially cross the classifier's class-1 decision threshold.
- The class-1 KDE contains 416 support points, uses bandwidth `0.05`, and the released plausibility threshold is `1.6425596476`.
- The final method does not read the released training arrays. The training points inside the supplied KDE are used only through the allowed generative model.

## Experiments

All scores below are local measurements on the released `x_explain.npy`, `disc_model.pth`, `gen_model.pth`, and threshold, using the notebook's metric formula and a vectorized expression mathematically equivalent to its KDE.

| Objective weights `(validity, distance², plausibility)` | Steps / learning rate | Mean L2 | Validity | Plausibility | Score |
|---|---:|---:|---:|---:|---:|
| `(25, 300, 79)` — supplied mirror candidate | 1,000 / 0.01 | 0.2030 | 0.9915 | 0.9237 | 91 |
| `(10, 30, 20)` | 1,000 / 0.01 | 0.2661 | 1.0000 | 0.9661 | 68 |
| `(5, 10, 10)` | 1,000 / 0.01 | 0.2837 | 1.0000 | 1.0000 | 60 |
| `(10, 10, 10)` | 1,000 / 0.01 | 0.3091 | 1.0000 | 0.9915 | 49 |
| `(30, 10, 10)` | 2,000 / 0.01 | 0.3482 | 1.0000 | 0.9915 | 49 |
| `(30, 30, 30)` | 2,000 / 0.01 | 0.3094 | 1.0000 | 0.9831 | 48 |
| `(50, 10, 20)` | 2,000 / 0.01 | 0.3692 | 1.0000 | 0.9831 | 48 |
| `(25, 5, 10)` | 3,000 / 0.005 | 0.3649 | 1.0000 | 0.9915 | 49 |
| `(25, 300, 200)` | 1,000 / 0.01 | 0.2074 | 1.0000 | 0.9661 | 97 |
| `(25, 300, 500)` | 1,000 / 0.01 | 0.2138 | 1.0000 | 1.0000 | **100** |
| `(40, 300, 500)` | 1,000 / 0.01 | 0.2325 | 1.0000 | 0.9915 | 91 |

The distance and plausibility objectives need to be balanced: larger plausibility weights moved every point above the threshold while preserving mean distance below 0.22. A larger validity weight was not helpful after validity reached 100%; it pushed the mean distance above the full-credit cutoff.

## Chosen method

[The solution](../../../solutions/2025/stage2/kredytobranie.py) runs 1,000 batched Adam steps. It minimizes binary cross-entropy for the requested class, squared distance to each original, and a ReLU penalty for falling below the supplied log-density threshold (with a 0.01 margin). It evaluates the provided KDE in one vectorized batch; this matches the notebook's Gaussian-kernel log density while avoiding its per-point Python loop.

## Measured score and evidence

Best local result: **100/100** at mean L2 `0.2138`, validity `1.000`, and plausibility `1.000` on all 118 released explanation points. An exact-code rerun using the official scoring functions reproduced 100/100 in `44.43` seconds on this CPU-only host; its minimum log density was `1.64582`, above the `1.64256` cutoff. CUDA is unavailable here, so GPU runtime was not measured. Hidden-test performance is unknown because the competition supplies new classifier and KDE models.

The official notebook supplies the scoring code and an empty starter function. The available mirror's example uses weights `(25, 300, 79)`; its output does not include a recorded score. The table contains our local runs, not an organizer result.

## Alternatives considered

- Returning the original points fails on most samples (only 25/118 valid initially).
- Lower plausibility penalties can meet the distance cutoff but leave some points below the density threshold.
- Higher classifier emphasis and weaker distance regularization keep validity high but exceed the mean-distance limit.
- A per-point grid search could exploit the two dimensional space, but the vectorized gradient method already reached full released-validation credit and adapts to replacement models.

## Progressive hints

1. The target is class 1 for every explanation point; optimize the classifier's raw logit with binary cross-entropy.
2. Use the supplied class-1 KDE to penalize low-density counterfactuals rather than moving directly across the decision boundary.
3. Keep the aggregate L2 change under 0.22; after validity is high, increase plausibility pressure before increasing classifier pressure.
4. A small margin above the density threshold reduces borderline failures, and batching the KDE makes iterative optimization inexpensive.

**One-line summary:** Optimize each point against validity, KDE plausibility, and change size; measured result is **100/100 on released validation**.
