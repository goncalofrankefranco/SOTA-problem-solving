# Predicting the Shooting Percentage of Basketball Stars — NOAI China 2024 Round 2

**Problem domain:** Binary classification from two spatial features  
**Evaluation metric:** Accuracy; a submission that violates the architecture rules receives 0.

## Abridged statement

Train a PyTorch MLP to predict whether a basketball shot is made. The model input must be exactly **loc_x** and **loc_y**, the output is one made/missed label, and only linear layers and allowed activations may be used. It may have at most three **nn.Linear** layers with at most eight neurons per layer; define the model class `MyModel` directly rather than wrapping layers in **nn.Sequential**. The task lists `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.ELU`, `nn.LeakyReLU`, and `nn.PReLU` as allowed activations. The score is test accuracy when constraints are met and zero otherwise. The official submission is `submission.zip` containing `submission_model.py` and `submission_dic.pth`.

## Dataset analysis and EDA

The SOTA task summary describes 20,000 training rows and about 5,000 hidden test rows. The public community mirror linked below instead contains a 25,000-row CSV with labels on every row. In that mirror, 13,814 shots are misses and 11,186 are made shots (44.74% positive); the majority-class accuracy is 55.26%. The coordinate ranges are **loc_x** −250 to 248 and **loc_y** −44 to 791. The source and split discrepancy means this file cannot be treated as the official 20k training split or as hidden-test evidence.

Only the two permitted location columns enter the model. The other columns in the mirror are not used. No organizer data files are stored in this repository.

## Experiments and selected method

The authored model in [basketball_shooting.py](../../../solutions/china/2024/basketball_shooting.py) is a 2 → 8 → 8 → 1 MLP with Tanh hidden activations and a logits output. It trains with BCEWithLogitsLoss and Adam. Standardization is fit inside each training fold and then folded into the first linear layer, so the saved model accepts the raw loc_x, loc_y pair without adding another model layer.

| Result | Accuracy | Provenance |
|---|---:|---|
| Five-fold stratified OOF, default logit threshold 0 | 0.5978 | Locally measured on the 25,000-row community mirror; ROC-AUC 0.6102. Fold accuracies: 0.5832–0.6088. |
| OOF threshold sweep, threshold 0.27 | 0.6088 | Locally measured on the same OOF logits; the threshold was selected on those same folds, so this is an optimistic tuning result rather than an unbiased estimate. The training command can fold this threshold into the output bias. |
| Organizer leaderboard A/B | 0.5604 / 0.5677 | Accuracy reported in the linked participant notebook; neither value is a result from this repository. |

The default OOF result is 4.52 percentage points above the mirror's majority-class baseline. The threshold sweep slightly improves the same OOF predictions, but should be reselected from an inner validation split before claiming an unbiased threshold-tuned estimate.

In train mode, the script now writes the required `submission.zip` with a standalone `MyModel` definition and the raw model state dictionary under the exact required names. The model source takes two raw features; its final bias includes the selected 0.27-logit threshold shift. Example: `python solutions/china/2024/basketball_shooting.py --mode train --data /path/to/data_train.csv --output submission.zip`. This packaging path has not been run on organizer data because those assets are unavailable.

## Result, compute, and limits

The five-fold run used CPU PyTorch and completed in about 8 seconds in this workspace. **No official score was measured here.** Bohrium's data and hidden-test grader require access unavailable in this environment, and the only readable copy was the third-party mirror. Its GitHub repository metadata reports no license (license: null), while SOTA lists the organizer source license as unstated. The mirrored data were used only from /tmp; they are not redistributed here. The public leaderboard values above are explicitly participant-reported reference results, not a verified organizer scorecard export.

## Alternatives considered

- A majority-class prediction scores 55.26% on the mirror but ignores the position signal.
- shot_distance, minutes_remaining, and shot_id could improve prediction but are outside the stated input restriction.
- Wider or deeper MLPs violate the eight-neuron/three-linear-layer cap.
- A threshold chosen from the hidden test would be leakage; the code only exposes a train-time threshold parameter.

## Progressive hints

1. Start with the made-shot fraction as an accuracy baseline.
2. Scale the two position coordinates using training data only.
3. Use a small nonlinear MLP to approximate the spatial success surface.
4. Check the exact layer count and output threshold before packaging the model.

**One-line summary:** A compliant two-feature 2 → 8 → 8 → 1 MLP reaches 0.5978 default-threshold OOF accuracy on an unlicensed community mirror; organizer hidden-test performance is unmeasured.

## Sources and reuse

- [SOTA task page and official-material links](https://checklist.sota-ai.org/problems/noai-china-2024-round-2-basketball-shooting/)
- [Bohrium task page](https://www.bohrium.com/en/competitions/5135119121)
- [Community notebook and mirrored assets](https://github.com/jaredliw/ioai-tsp-2025/blob/main/noai-china-2024/basketball-shooting/basketball-shooting.ipynb)

The task statement/data license is not stated on the SOTA page; the community repository has no declared license. This report paraphrases the rules and links to the sources. It does not reproduce the statement, dataset, or notebook.
