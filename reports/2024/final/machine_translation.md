# Machine Translation — 2024 Polish AI Olympiad Final

**Problem domain:** Neural machine translation and learned word alignment  
**Evaluation metric:** Tokenized corpus BLEU-4, with validation/test token cross-entropy as diagnostics. The official task has a 13-point project rubric (7 model implementation, 3 training/evaluation, 1 attention visualization, 2 additional experiments); it defines no single accuracy-to-score formula.

## Abridged statement

Implement and train the method from Bahdanau, Cho, and Bengio, “Neural Machine Translation by Jointly Learning to Align and Translate,” on the English–German Multi30k parallel corpus. Track training and validation loss, evaluate on the provided test split with appropriate translation metrics, show successful and unsuccessful translations, visualize attention, and conduct additional experiments. The official starter initializes German as the input vocabulary and English as the output vocabulary, so the implementation below supports German-to-English translation.

## Dataset analysis and EDA

The official loader uses `bentrevett/multi30k`, lowercases SpaCy-tokenized text, retains tokens appearing at least twice in training, and prepends `<sos>` / appends `<eos>`. The Rayan notebook's saved loader output records 29,000 training pairs, 1,014 validation pairs, and 1,000 test pairs. The local cache does not include the Hugging Face dataset, vocabularies, or SpaCy models, so vocabulary sizes, length distributions, unknown-token rates, and sample translations could not be measured here. The official dataloader pads source and target sequences time-major.

## Experiments and selected method

| Method | Evidence | Decision |
|---|---|---|
| Official starter | Model, training, metric, plots, and experiment sections remain TODO; no score | Not a runnable baseline. |
| Rayan participant notebook | No training/evaluation run; model code has unresolved shape and attribute errors, and its training cells remain TODO | Inspected as context, not treated as a result. |
| **Bidirectional GRU encoder + additive Bahdanau attention + GRU decoder with deep maxout output** | Implementation added; no dataset is locally available for a BLEU/loss run | Selected as a faithful and executable paper-based implementation. |

The model retains a hidden annotation at each source position, initializes the decoder from the concatenated final forward/backward encoder states, predicts a soft alignment for each generated token, and conditions a GRU decoder on its previous word and attention context. The deep output layer combines decoder state, previous-word embedding, and context, then applies maxout before vocabulary logits. Padding is masked in attention and ignored by the loss. Training uses Adam, 0.5 teacher forcing, gradient clipping, validation-loss checkpointing, learning-rate reduction, and early stopping. Utilities provide greedy decoding, tokenized BLEU-4, loss plots, and attention plots.

Implementation: [machine_translation.py](../../../solutions/2024/final/machine_translation.py).

## Result, compute, and limits

**BLEU and training score: unverified; no local measurement.** The official repository contains `machine_translation.ipynb` and the cited paper but no corpus files. The local Rayan notebook records the Multi30k split sizes but has no trained model or metrics. The environment does not have the dataset, SpaCy models, or a GPU, and network access to retrieve the required resources is blocked. No claim is made about translation quality or rubric points. The official task's validation-loss monitoring permits using validation references for checkpoint selection and metrics; they are not used for gradient updates. Test references are reserved for final evaluation.

The original task allowed a Colab T4 and did not state a wall-time limit. No runtime was measured; the 512-hidden-unit, 256-dimensional starter configuration should be benchmarked on the competition hardware. The code is model/data-loader agnostic and expects the official prepared token-id batches (`de_ids`, `en_ids`) without downloading data. Corpus BLEU uses add-one smoothing and is a diagnostic implementation; for directly comparable published scores, replace it with the paper/organizer's specified BLEU implementation if one is supplied.

No additional architecture ablations could be measured. Reasonable next experiments are (1) teacher-forcing ratios 1.0 vs. 0.5 vs. 0.0, (2) hidden dimension and dropout sweeps, (3) attention and maxout ablations, and (4) greedy versus beam decoding, selected using validation loss/BLEU only.

## Progressive hints

1. Keep every encoder time-step output; a single fixed sentence vector is the bottleneck the paper addresses.
2. Compute an additive attention distribution from the previous decoder state and each encoder annotation.
3. Feed the weighted context and previous target word into the decoder, masking padded source positions.
4. Ignore target padding in cross-entropy, clip gradients, and track validation metrics separately from training.
5. Inspect attention plots and representative successful/failed translations before drawing conclusions from a single BLEU score.

**One-line summary:** The paper's bidirectional-GRU, additive-attention, maxout model and training/evaluation utilities are implemented, but BLEU and rubric points remain unverified because Multi30k is unavailable locally.
