# Translation Stylization

## Problem, domain, and metric

- **Domain:** English-to-Polish machine translation with terminology preservation.
- **Task:** adapt the MarianMT model `gsarti/opus-mt-tc-en-pl` so selected AI/ML keywords remain in English, using the provided English/Polish training pairs and keyword lists. The evaluator calls `process_example(en, keywords)`, generates a Polish hypothesis, and compares it with one reference.
- **Metric:** the notebook averages NLTK's unsmoothed `sentence_bleu` over examples. The score is 0 at BLEU ≤0.82, 100 at BLEU ≥0.86, and linear in between. The test set has no Polish references available to the solution.
- **Abridged statement:** train a translation model and preprocess each English input so that the target's domain terms follow the corpus style of staying untranslated.

## Released data and EDA

The [official task notebook and datasets](https://github.com/OlimpiadaAI/II-OlimpiadaAI/tree/main/3_etap/3_stylizacja_tlumaczen) provide 2,808 training and 856 validation examples. Mean sentence lengths are 92.3/96.7 characters for training English/Polish and 93.1/97.0 for validation. The training set has 1,312 distinct keyword strings; validation has 993, of which 973 appear in training. Examples have one to five keywords, most often two.

Whole-phrase, case-insensitive matching finds 95.43% of training keyword occurrences in the English source and 94.81% in the Polish reference. For validation the corresponding rates are 95.13% and 93.55%. Thus inline keyword protection has a strong signal, but exact preservation is not universal; blindly forcing every phrase could disagree with some references.

## Experiments and score evidence

| Experiment | BLEU / points | Status |
|---|---|---|
| Starter Marian model with no fine-tuning | Not measured | Marian weights are not cached locally; no starter output is saved in the official notebook |
| Inline keyword markers plus supervised Marian fine-tuning | Not measured | Candidate implementation saved; cannot load base weights in this environment |

The official evaluator and score function were inspected, but there is no official/released BLEU result to cite. Network requests to Hugging Face fail through the environment proxy with HTTP tunnel 403, and the model files are absent locally. This is a model-access blocker, not a BLEU failure. No validation translation targets were used for training.

## Selected candidate

[translation_stylization.py](../../../solutions/2025/stage3/translation_stylization.py) provides `process_example` and `train_translation_model`. Preprocessing wraps each whole-word keyword occurrence in plain-text `[KEEPTERM]` markers, without changing the tokenizer. Fine-tuning uses only the official training pairs, with the original Polish translation as target, full-model AdamW at `2e-5`, batch size 16, and two epochs. The same preprocessing is used during training and inference.

This is an unscored candidate, not a verified 100-point solution. The official starter already loads the specified base model/tokenizer; the function expects those objects rather than downloading or modifying the tokenizer itself.

## Runtime, alternatives, and caveats

The platform allows ten minutes total for training and evaluation on GPU. Runtime was not measured because the base Marian checkpoint could not be loaded and this workspace has CPU-only PyTorch. The 2,808-example training set is small, but whether two full-model epochs fit the official budget remains unverified here.

Alternatives include fine-tuning on raw English only, using LoRA, or adding a textual instruction listing terms. These require a working base checkpoint and BLEU evaluation before they can be ranked. Validation keywords and references are available for normal scoring, but the selected training routine deliberately reads only `train_dataset`.

## Progressive hints

1. The target references already preserve most listed terms verbatim; first measure how often each term occurs in the source and target.
2. Fine-tune the supplied translation model on the provided pairs before adding more complicated prompt instructions.
3. Mark exact keyword spans in source text, use the same transform during training and evaluation, and leave the tokenizer unchanged.
4. Check sentence-level BLEU on released validation because the references do not preserve every listed phrase exactly.

**One-line summary:** a train-only fine-tuning candidate is saved, but missing Marian weights block validation BLEU measurement; the hidden-test score is unknown.
