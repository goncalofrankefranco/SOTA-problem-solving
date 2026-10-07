# Solving the Pendulum Motion with Missing Data — NOAI China 2024 Round 2

**Problem domain:** Time-series physics parameter estimation  
**Evaluation metric:** Score = (S1 + S2 + 2S3 + 2S4 + 2S5) / 8, where S1 = exp(−10|l̂−l|), S2 = exp(−10|μ̂−μ|), S3 = exp(−|F̂−F|), S4 = exp(−10|t̂_nextzero−t_nextzero|), and S5 = exp(−10|t̂_Fput−t_Fput|). A wrongly formatted submission scores 0.

## Abridged statement

From an interrupted angle recording, infer a pendulum's rope length l, damping coefficient μ (miu in the required CSV), the magnitude F of a downward force that begins at unknown time t_Fput during a missing interval, and the next time the angle reaches zero after recording ends. The task uses g = 9.8; the motion is governed by θ'' = −μ θ' − (g/l + F/l · I[t ≥ t_Fput]) sin(θ) for unit mass. PyTorch is required for regression. The full submission writes the five quantities to the prescribed CSV columns.

## Dataset analysis and EDA

The community mirror's pendulum_train.csv has 734 observed points from t=0 to t=15 seconds. The normal step is about 0.015015 seconds. One 4.009-second gap runs from t=3.003 to t=7.012; the second visible segment continues to the end of the recording. The two hidden evaluation recordings/labels were not accessible.

## Experiments and selected method

[pendulum_motion.py](../../../solutions/china/2024/pendulum_motion.py) detects the interval from the sampling-time jump, reconstructs its regular grid, and uses `torch.linalg.lstsq` regression on finite-difference derivatives to initialize the physical coefficients. It then refines the piecewise nonlinear ODE fit with SciPy `least_squares`/`solve_ivp` and estimates the next zero crossing from the fitted terminal state. Thus PyTorch is used for the required regression initialization, while nonlinear refinement and integration use SciPy. The reported parameter fit uses all visible train angles.

| Result | Value | Provenance |
|---|---:|---|
| Rope length l | 4.89923 | Local fit to the visible mirrored training trajectory |
| Damping μ | 0.50077 | Local fit; written as miu in the output CSV |
| Force F | 39.95445 | Local fit |
| Force-start time t_Fput | 4.41649 s | Local fit inside the missing interval |
| Next zero crossing | 15.87686 s | Extrapolated after the recording ends |
| RMSE on observed angles | 0.000270 rad | Fit diagnostic only; not the task score |
| Participant leaderboard A/B score | 0.8710 / 0.9186 | Values reported by the third-party participant notebook, not this implementation |

The mirror's own submission_train.csv contains a separate candidate estimate (4.91793, 0.49931, 40.13448, 15.8825, 4.41800). This is another candidate output, not ground truth, so its closeness is only a sanity check. **No parameter score was measured here because the organizer's true parameters and hidden A/B test recordings are unavailable.**

## Result, compute, and limits

The visible-curve fit completed in about 16 seconds on CPU. Its small angle RMSE shows that the piecewise dynamics explain the mirrored training recording, but it does not verify accuracy on different hidden parameters. In batch mode the script reads one training file and two evaluation files, writes `submission_train.csv` separately, writes one-row `submissionA.csv` and `submissionB.csv` files with the exact columns `l,miu,F,t_nextzerotheta,t_Fput`, and zips A/B as `submission.zip`. Example: `python solutions/china/2024/pendulum_motion.py --train /path/pendulum_train.csv --test-a /path/pendulum_testA.csv --test-b /path/pendulum_testB.csv --output-dir submission`. The hidden A/B inputs were unavailable, so only the training candidate was produced locally; the batch output and archive were not generated on hidden data. Bohrium data/grader access was unavailable.

The SOTA page lists the organizer source license as unstated. The community GitHub repository has license: null. The mirror and notebook were kept under /tmp and are not included here.

## Alternatives considered

- Directly smoothing and differentiating the full trace is fragile around the missing interval; derivative regression is limited to contiguous samples.
- A linearized small-angle model simplifies fitting but introduces bias when the initial angle is not small.
- Fitting the exact nonlinear pendulum equation and switching the forcing at a profiled time better uses the visible waveform, though it cannot reveal the hidden-test score.

## Progressive hints

1. Detect the gap from the time differences, not from angle values.
2. Fit damping and the pre-force frequency using only contiguous pre-gap observations.
3. Use the post-gap phase to estimate the changed frequency and force start.
4. Integrate the fitted post-force system beyond the final sample to find the next zero.

**One-line summary:** The visible mirrored training curve is fit to 2.70e−4 rad RMSE with a PyTorch-initialized physical model, but the official parameter score is unmeasured.

## Sources and reuse

- [SOTA task page and official-material links](https://checklist.sota-ai.org/problems/noai-china-2024-round-2-pendulum-motion/)
- [Bohrium task page](https://www.bohrium.com/en/competitions/1723157880)
- [Community notebook and mirrored training/output files](https://github.com/jaredliw/ioai-tsp-2025/blob/main/noai-china-2024/pendulum-motion/pendulum-motion.ipynb)

The original materials have no license stated on the SOTA page, and the community repository declares no license. This report paraphrases the task and links to the sources without reproducing the notebook or any organizer data.
