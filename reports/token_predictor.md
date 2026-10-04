# Poland 2026 Stage 2 — Predyktor tokenów (Token Predictor)

**Problem domain:** NLP, language-model context features, candidate-token classification, weakly supervised learning from a text stream  
**Evaluation metric:** Mean per-example balanced accuracy: the average of recall on good-token IDs and recall on bad-token IDs, then averaged over examples. Accuracy at or below 0.62 earns 0 points; at or above 0.68 earns 100, with linear scaling between those thresholds.

## Abridged statement

Given a 512-token Shakespeare prompt and a candidate list, divide the candidate token IDs into two disjoint lists. A good token appears in the next 128 tokens of the source text; a bad token does not. The 301,966-token training stream and a pretrained GPT-2 XL are provided. The released validation set has 99 examples, and the hidden test has 100. In final evaluation, inference must finish within five minutes on the competition GPU. The allowed libraries are NumPy and PyTorch.

## Dataset analysis and EDA

The training input is one contiguous 301,966-token stream encoded with GPT-2 byte-level BPE, drawn from Shakespeare plays. The validation records contain a 512-token prompt, good and bad candidate IDs, and a shuffled candidate list. The good and bad set sizes vary across records: in the released validation archive, good-set sizes range from 52 to 90 and bad-set sizes from 54 to 197. Thus, candidate-set size and class balance are not constant. The 99 validation prompts come from a different, sequential passage of the same source text, so adjacent examples are related rather than independent.

EDA inspected decoded text, prompt suffixes, candidate frequency, token frequency in the full training stream, and counts in progressively shorter prompt windows. I also tested exact suffix n-gram lookup for lengths 2–8. The source stream contains repeated vocabulary and local structure, but exact long suffix matches are sparse for these validation prompts. Global token frequency and prompt occurrence are informative, while neither alone is strong enough to solve the task. A possible GPT-2 XL candidate-logit feature could not be tested because the linked checkpoint was absent from the workspace and could not be downloaded here.

## Solution strategy and experiments

The selected implementation builds per-candidate features from the training-stream unigram count, token ID, counts in the full prompt and its last 8–512 tokens, exponentially recency-weighted counts, and prompt-presence/repetition flags. A small logistic ranker was fit offline on all 1,178 stride-256 synthetic 512-to-128 windows available from the released training stream. Good IDs are the unique IDs in each following 128-token horizon. For each synthetic example, negatives are the unique IDs from a separate random 512-token passage, with overlap against the positive set removed. Fixed coefficients are embedded in the submitted NumPy implementation; no model is fit on the validation labels at inference.

The positive-list size prior is also estimated from those training-only windows: the mean number of unique IDs in a 128-token horizon is 82.53, so the implementation assigns the 83 highest-ranked candidates to `good_answer`. This prior does not assume that the negative pool has the same size as the positive set.

| Experiment | Validation result | Decision |
|---|---:|---|
| Rank candidates by global training-stream unigram frequency | Candidate AUC 0.7075; best per-case balanced accuracy about 0.651 | Keep as a useful baseline feature, not as the sole model. |
| Earlier synthetic ranker (mixed uniform-vocabulary and 256-token text negatives), top half | Mean balanced accuracy 0.674894; 91/100 points | Previous baseline; improved by matching negative examples to broader text distractor pools. |
| 1,178-window ranker with separate 512-token text negatives, top half | Candidate AUC 0.74860; mean balanced accuracy 0.68215; 100/100 points | Improved the ranking model using train-stream-only synthetic examples. |
| Same ranker, select 83 candidates using the train-stream mean horizon size | **Mean balanced accuracy 0.684145; 100/100 points** | Selected. The count is rounded from the training-only mean of 82.53 positive IDs per horizon. |
| Train-only Ridge prediction of per-example positive-set size | MAE 8.20 IDs; mean balanced accuracy 0.68074 | Tested for adaptive set size; lower than the train-derived fixed count. |
| Select a fixed count of 87 after comparing counts on validation | Mean balanced accuracy 0.68476 | Validation-tuned diagnostic only; not used because the count was selected from validation labels. |
| Set size from the sum of candidate probabilities, with a 0.95 scale | Earlier ranker: MAE 13.13 IDs; mean balanced accuracy 0.67333 | Tested and discarded. |
| Exact suffix n-gram continuation features, lengths 2–8 | Grouped-CV candidate AUC 0.7420 / threshold balanced accuracy 0.6730, versus 0.7431 / 0.6736 for numeric prompt-frequency features | Sparse exact matches did not help; excluded from the final scorer. |
| Logistic candidate ranker trained on validation labels, 5-fold grouped CV over contiguous blocks of about 20 examples with a fold-fitted size predictor | Mean balanced accuracy 0.68104 | Validation-only diagnostic, not used by the submitted method. It uses labeled validation examples and is not a deployable or untouched test estimate. |

The final fixed count of 83 is derived from training-stream horizons, not the validation labels. It is not a claim that every example has the same positive-set size. A train-only per-example size regressor was also tested, but its validation score was lower. Validation labels were used only to calculate development metrics and compare global strategies; the final prediction method does not read them, identify validation rows, or use sample-specific answers.

## Results and diagnostics

The final file is [`solutions/stage2/token_predictor.py`](../solutions/stage2/token_predictor.py). It was run against all 99 released validation records using only `train_data` and each record's prompt/candidate list. It reproduced mean per-example balanced accuracy **0.684145**, corresponding to **100/100** under the notebook's scaling and rounding. The outputs formed disjoint partitions and covered every unique candidate ID in all 99 records. Other train-only random-window/C variants scored between 0.683416 and 0.685613 at the same fixed count of 83.

The released validation score reaches the notebook's full-credit threshold. This does **not** establish a 100/100 hidden-test score: selector variants were compared on a public split whose rows are sequential and related. The GPT-2 XL candidate-logit feature remains disabled because the linked checkpoint was unavailable, and no hidden-test score is available.

## Implementation and compute footprint

The solution uses NumPy feature extraction and fixed logistic coefficients; it imports no scikit-learn. It counts the training stream once and caches the resulting unigram table. Each call ranks only the supplied candidate IDs, deduplicates IDs defensively, and returns a disjoint `(bad_answer, good_answer)` partition. The official two-argument notebook call and the requested `classify(model, prompt, token_list)` form are both supported. The 99 validation calls took 0.08 seconds on the local CPU in this development environment; this was not a timed competition-GPU run. No GPU is needed for the selected method.

An optional, disabled GPT feature can aggregate candidate log-probabilities across the last prompt positions if `model` is a callable PyTorch model. It was not run against the actual checkpoint, is not included in the reported score, and should remain disabled unless separately validated within the runtime limit.

## Hints

1. What exactly makes a candidate good: does it occur anywhere in the next 128-token continuation, or only as the very next token?
2. How do global training-stream frequency and occurrences near the end of the prompt compare with one another?
3. Do counts over several prompt suffix lengths or recency-weighted counts rank candidates better than a single frequency?
4. The good and bad set sizes vary. Can a size predictor trained only from the permitted text stream improve balanced accuracy over a fixed selection rule?
5. If the provided GPT-2 XL is available, how should candidate logits be combined with corpus counts, and what validation prevents the added model cost from hurting the score?

**One-line solution:** Rank candidates with a train-stream-fitted logistic score from unigram and multi-scale prompt/recency counts, then assign the top 83 candidates to `good_answer` (measured: **100/100** on the released validation split).

**Official task and starter notebook:** [Predyktor tokenów](https://github.com/OlimpiadaAI/III-OlimpiadaAI/tree/main/2_etap/predyktor_tokenow)
