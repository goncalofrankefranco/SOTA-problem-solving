# NOAI China 2025 — Compound Word Segmentation

**Problem domain:** German compound word segmentation, character-level sequence labeling  
**Metric:** Mean per-word F1 on exact constituent intervals  
**Official task page:** [SOTA checklist: Compound Word Segmentation](https://checklist.sota-ai.org/problems/noai-china-2025-round-2-compound-word-segmentation/) · [Bohrium contest page](https://www.bohrium.com/en/competitions/53892361357)

## Abridged task

Given a concatenated German compound, return one 0/1 label per character. A `1` marks the final character of a constituent; a `0` marks other characters. The released training file has 94,306 labeled words. The 11,788 validation and 11,789 test words have empty target arrays and are read in the competition environment. Each word is scored by F1 over exact boundary/interval matches, then scores are averaged over words. The validation leaderboard is public; the test leaderboard is the final result. See the official statement linked from the task page for the controlling specification.

## Data and license

No NOAI data or Bohrium `ANSWER_PATH` was mounted in this workspace. The SOTA task page lists the source license as **not stated**. It also links the training data through Bohrium. I did not include any organizer data or notebooks here.

## Dataset analysis and EDA

The task summary lists 94,306 labeled training words, 11,788 validation words, and 11,789 test words. The labels encode constituent endings as one binary value per character. The separate IOAI GAITE repository contains a same-sized labeled file, but identity with the NOAI files and split order has not been established. No NOAI character frequencies, word-length distribution, or boundary-rate summary could be computed locally.

The [official IOAI-2025 repository](https://github.com/IOAI-official/IOAI-2025) contains a similarly named GAITE word-segmentation task and a 94,306-word training file under the repository's CC BY 4.0 license. I used that public file for a local holdout experiment below. Its identity with the NOAI Bohrium files is not verified, so the score is proxy evidence only. The script accepts the NOAI training JSON directly so it can be evaluated without changing the model code.

## Method

The candidate in [compound_word_segmentation.py](../../../solutions/china/2025/compound_word_segmentation.py) combines two signals:

- A two-layer bidirectional character LSTM predicts boundary probabilities with a binary cross-entropy objective.
- A dynamic program uses the training constituents as a soft lexicon. It gives known constituent spans a frequency-weighted bonus while retaining an unconstrained path for unseen pieces. The last character of every word is always marked as a boundary.

The script includes a random 90/10 word holdout mode that reports the official mean per-word F1. The default threshold and lexicon weight are starting values; tune them on a held-out split before selecting a final run. Submit mode trains on all labeled words and writes the required validation and test JSON files into `submission.zip`.

## Experiments and score evidence

| Result | Score | Provenance |
|---|---:|---|
| NOAI validation leaderboard | Not measured | No Bohrium login, mounted task data, or active submission window was available. |
| NOAI hidden test leaderboard | Not measured | Hidden labels and final evaluation are inaccessible in this workspace. |
| Local GAITE analogue holdout | 0.862135 mean per-word exact-segment F1 | Measured on a seeded random 10% split: 9,430 held-out words; the model trained on the other 84,876 words for 1 epoch, batch size 512, threshold 0.42 and lexicon weight 0.25. This is not a NOAI leaderboard score. |

The public IOAI GAITE task is not counted as a NOAI result unless the data are shown to be identical. This local holdout is a candidate estimate only, not an organizer score. No score from an organizer reference model is attributed to this implementation.

## Implementation, diagnostics, and compute

The 0.862135 proxy run used one training epoch on CPU; it is a metric check on related public data, not evidence of NOAI validation quality. No NOAI runtime or memory measurement is available. The interval-level scorer now compares exact `(start, end)` constituent spans, matching the task metric described on the checklist. The training script remains configurable to rerun validation on the actual NOAI JSON files.

## Alternatives considered

- Character n-gram classifiers can be faster and provide a strong baseline for boundary decisions.
- A CRF can model adjacent boundary consistency explicitly.
- A lexicon-only dynamic program is useful as a diagnostic but may fail on unseen constituents; compare its held-out score with the BiLSTM and blended variants.

## Run

```bash
python solutions/china/2025/compound_word_segmentation.py \
  --mode validate \
  --train-json /path/to/IOAI-2025/GAITE-Contest/Word_Segmentation/train.json \
  --epochs 1 --batch-size 512 --seed 17

python solutions/china/2025/compound_word_segmentation.py \
  --mode submit --train-json /path/to/train.json \
  --val-json /path/to/val.json --test-json /path/to/test.json \
  --output-dir submission
```

Requires PyTorch and NumPy. The script does not fetch the dataset or access encrypted files automatically; pass the official file paths explicitly. No dataset, checkpoint, or organizer notebook is stored in this repository.

## Limitations and next steps

- Run the built-in holdout evaluation on the exact NOAI training file, then tune the boundary threshold and lexicon bonus without consulting the hidden test labels.
- Confirm the character-label and JSON serialization requirements against the official Chinese statement before uploading; the statement overrides the abridged checklist.
- The BiLSTM may miss rare constituents and spelling/linking variants. Character n-gram models or a CRF head are reasonable extensions if holdout F1 improves.

## Progressive hints

1. Treat the target as character-level boundary labels and decode them into constituent intervals.
2. Inspect word lengths, boundary frequency, and repeated substrings in the training set.
3. Use a character model to estimate boundary probabilities, then add training constituents as a soft lexicon.
4. Tune the decoder with mean per-word exact-interval F1; boundary accuracy alone does not match the task score.

**One-line solution:** A character BiLSTM predicts endings, and a lexicon-aware dynamic program turns them into consistent German compound segments; the NOAI score remains unmeasured.

## Sources and reuse

- [NOAI task summary and linked official statement/data](https://checklist.sota-ai.org/problems/noai-china-2025-round-2-compound-word-segmentation/) — license not stated on the source page.
- [Official IOAI-2025 task repository](https://github.com/IOAI-official/IOAI-2025) — separate GAITE task materials; repository license is CC BY 4.0. It is cited as a related source, not as proof of NOAI data identity.
