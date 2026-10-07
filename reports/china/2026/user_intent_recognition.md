# NOAI China 2026 — User Intent Recognition in Zhihu Scenarios

**Problem domain:** NLP, few-shot text classification  
**Evaluation metric:** Weighted F1 over 16 intent classes; maximum 1.0.

## Abridged statement

Assign each Chinese search query or multi-turn dialogue to one of 16 user-intent classes. For dialogue input, earlier turns provide context, but the label is determined by the final user message. The supplied labeled set contains only one example per class. This paraphrase is based on the task summary; use the official statement for exact rules.

## Dataset analysis and EDA

The SOTA task page reports 16 labeled examples, one per intent, plus 2,500 unlabeled validation rows and 2,500 unlabeled test rows. The local workspace has none of these assets or the supplied `bert-base-chinese` checkpoint, so no corpus plots, class-distribution analysis, or validation score could be produced here.

This is an extreme few-shot setting. Several categories are close in meaning—for example, general health knowledge versus finding medical services, and interpreting a legal rule versus asking for legal advice. Dialogue context can help disambiguate a short final query, while using every `usr:` line equally would conflict with the stated labeling rule. The allowed local BERT checkpoint is a useful language prior; the task page also notes that a development-only Qwen model must not be called by the submitted notebook.

## Experiments and solution strategy

No competition-data experiment was run. The candidate in [the solution script](../../../solutions/china/2026/user_intent_recognition.py) combines:

1. Last-user-turn extraction for dialog inputs.
2. Frozen, mean-pooled embeddings from the task-supplied local Chinese BERT checkpoint.
3. A regularized logistic classifier trained on the 16 provided examples plus a small set of authored, label-grounded Chinese anchor queries.
4. A character n-gram classifier blended with the embedding classifier to catch explicit terms and spelling variants.

The blend and anchor set are starting choices, not tuned results. With data access, compare last-turn-only versus context-preserving input, frozen embeddings versus light fine-tuning, anchor weights, and blend weights on the provided validation set while reporting weighted F1. Preserve the 16 official class strings exactly in output.

## Score evidence and limits

| Result | Weighted F1 | Provenance |
|---|---:|---|
| This candidate | Not measured | Training, validation, and supplied checkpoint are not mounted here; no Bohrium grading access. |
| Organizer baseline B | 0.1133 | Reported on the SOTA task page. |
| Scientific Committee reference B | 0.8154 | Reported on the SOTA task page; not reproduced by this repository. |

The reference score is not evidence that this candidate approaches it. No score or optimality claim is made.

## Implementation and diagnostics

The script expects `/bohr/train-a3ld/v1/train.jsonl` and the local `bert-base-chinese` directory there. It reads `val.jsonl` and `test.jsonl` from `DATA_PATH` (default `/bohr`), then writes `submission.zip` containing `submission_val.jsonl` and `submission_test.jsonl`. Each output line is one JSON object with a Chinese `label`, in source order. The code does not use Qwen, network access, an external LLM API, or runtime package installation.

No task-level diagnostic run was possible. A real validation run is needed to identify class confusions, confirm the anchor strategy helps, select model/blend weights, and verify the 25-minute GPU limit in `noai:2026v1.1`.

## Compute and footprint

The candidate uses the supplied BERT checkpoint for embedding extraction and small scikit-learn classifiers. It selects CUDA when available; no extra pretrained model or dataset is bundled. Runtime and peak memory were not measured. The competition allows GPU execution for up to 25 minutes and prohibits internet access, `pip install`, external LLM APIs, and Qwen at submission runtime.

## Alternatives to compare

- A small supervised fine-tune of BERT, with early stopping based on held-out or cross-validated examples; the one-example-per-class regime makes direct fine-tuning easy to overfit.
- Nearest-prototype or nearest-neighbor prediction over BERT embeddings, including Chinese class descriptions.
- High-confidence pseudo-labeling over the unlabeled split, evaluated carefully because class imbalance and errors can reinforce one another.
- A Qwen-assisted static artifact only if produced during the permitted development phase and fully independent of Qwen at runtime; the current candidate does not use this option.

## Progressive hints

1. In a dialogue, which turn determines the gold intent?
2. What information can be learned from one labeled example per class, and what must come from the supplied pretrained model?
3. Which intent pairs are semantically close, and can the final user utterance plus a compact class description separate them?
4. Compare frozen BERT embeddings, a regularized lexical model, and a careful blend on weighted F1.

**One-line solution:** Extract the final user turn, encode it with the supplied Chinese BERT model, and classify with a validation-tuned combination of semantic and character-level features.

## Sources and reuse

- [SOTA task page and English baseline translation](https://checklist.sota-ai.org/problems/noai-china-2026-round-2-user-intent-recognition/)
- The task page links the official English and Chinese statements, Bohrium notebook, competition, and source files.

SOTA reports the source license as **not stated**. This report paraphrases the task and links to its source. The official data, statement, baseline notebook, and model weights are not copied into the repository.
