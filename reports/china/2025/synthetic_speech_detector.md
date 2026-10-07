# NOAI China 2025 — Synthetic Speech Detector

**Problem domain:** Binary classification of log-Mel spectrograms  
**Metric:** F1 score  
**Official task page:** [SOTA checklist: Synthetic Speech Detector](https://checklist.sota-ai.org/problems/noai-china-2025-round-2-synthetic-speech-detection/) · [Bohrium task](https://www.bohrium.com/en/competitions/18645233825)

## Abridged task

Classify a speech spectrogram as genuine (`bonafide`, label 0) or synthesized (`spoof`, label 1). The data are `.pt` tensors, usually shaped `[1, 128, 94]` for roughly three seconds of audio. Training files are labeled by folder; validation and test tensors are exposed only through the encrypted competition environment. Submit one 0/1 prediction per file in `submissionA.csv` and `submissionB.csv`, without headers, inside `submission.zip`. The official statement linked from the task page is authoritative.

## Data and license

No NOAI `ANSWER_PATH` or Bohrium grader was available. The NOAI checklist page lists the source license as **not stated**. I found a separate official [IOAI-2025 GAITE task repository](https://github.com/IOAI-official/IOAI-2025) with a similarly named task and data under its CC BY 4.0 repository license. That public copy has 10,995 training tensors and released validation/test sets of 1,374/1,375 tensors with a reference metric file. This is a related-task proxy; I did not verify that its files or sample order are identical to the NOAI Bohrium mount. The repository's datasets, label files, and notebooks are not included here.

## Dataset analysis and EDA

The NOAI summary describes labeled spectrogram tensors stored by class and hidden validation/test files exposed through the competition environment. The separate GAITE data use one-channel tensors around `[1, 128, 94]`; that shape and its split counts are proxy-source details, not confirmed NOAI properties. No NOAI class balance, tensor range, recording overlap, or spectrogram plots were available for local inspection.

## Method

The candidate in [synthetic_speech_detector.py](../../../solutions/china/2025/synthetic_speech_detector.py) adapts ResNet18 to one-channel spectrograms and uses mild time/frequency masking. It scales and clips the log-energy values, trains with cross-entropy, keeps the best epoch on an internal stratified split, and emits the required ordered CSV predictions. By default it uses randomly initialized weights and no downloads. An optional `--pretrained` flag uses torchvision ResNet18 weights only if already cached locally; the script deliberately does not download them.

## Experiments and score evidence

| Result | Score | Provenance |
|---|---:|---|
| NOAI validation leaderboard | Not measured | No NOAI credentials or encrypted task data were available. |
| NOAI hidden test leaderboard | Not measured | The final labels and grader were inaccessible. |
| Candidate validation split | Not measured yet | The run command below reports macro and positive-class F1 on an internal stratified split. |
| IOAI GAITE public test proxy | Not measured | The related task has a public reference metric file, but no candidate run is claimed here. |

The official IOAI GAITE task description reports its own scientific-committee baseline and maximum scores. Those are not scores for this code or for NOAI China and are not reproduced as candidate results.

## Implementation, diagnostics, and compute

The script uses a one-channel ResNet18 and can run on CPU or CUDA. No NOAI or GAITE candidate run was measured, so this report has no model score, runtime, or memory result. It uses numeric filename ordering for output; confirm that ordering and the exact F1 averaging rule against the official helper before submitting.

## Alternatives considered

- A compact CNN on log-Mel tensors can reduce the parameter count and overfitting risk.
- Compare time/frequency masking and normalization choices on a speaker-aware validation split if speaker IDs are available.
- Use cached pretrained weights only if the competition rules permit them and the validation score improves.

## Run

```bash
python solutions/china/2025/synthetic_speech_detector.py \
  --mode validate --train-dir /path/to/training_set

python solutions/china/2025/synthetic_speech_detector.py \
  --mode submit --train-dir /path/to/training_set \
  --val-dir /path/to/validation_set --test-dir /path/to/testing_set \
  --output-dir submission
```

Requires PyTorch, torchvision, pandas, scikit-learn, NumPy and `.pt` tensors. The prediction order is numeric filename order. Verify it against the official helper and statement before submitting; the validation/test labels are never used by the code.

## Limitations and next steps

- Verify whether the NOAI and IOAI GAITE datasets are the same before using proxy data to tune the NOAI solution.
- Run the internal stratified validation and compare the default threshold with thresholds chosen only on that validation split.
- If cached designated weights are permitted by the exact NOAI rules, compare the optional pretrained model; do not assume internet access during evaluation.
- Confirm the expected F1 averaging convention and exact sort order from the organizer's statement/helper before upload.

## Progressive hints

1. Treat each tensor as a time-frequency view of speech and verify the class labels and tensor ranges.
2. Check whether train/validation examples share speakers or recording sources before splitting.
3. Train a small two-dimensional CNN with class-aware validation and report the required F1 variant.
4. Tune threshold, normalization, and masking only on validation data; preserve the organizer's file order in the submission.

**One-line solution:** Adapt ResNet18 to one-channel Mel images and train with light SpecAugment; no NOAI score is claimed.

## Sources and reuse

- [NOAI task summary, data description and scoring](https://checklist.sota-ai.org/problems/noai-china-2025-round-2-synthetic-speech-detection/) — source license not stated.
- [Official IOAI-2025 repository](https://github.com/IOAI-official/IOAI-2025) — separate GAITE task materials under CC BY 4.0; proxy data identity is unverified.
- [Official IOAI news on NOAI China Finals 2025 and A/B grading](https://ioai-official.org/noai-china-finals-2025-8-team-members-selected-for-chinas-national-team-for-ioai-2025/).
