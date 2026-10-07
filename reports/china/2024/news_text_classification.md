# News Text Classification Task — NOAI China 2024 Round 2

**Problem domain:** Multiclass text classification  
**Evaluation metric:** Macro-F1, the unweighted mean of category-level F1 scores. Training and testing must finish within 10 minutes on CPU; if the per-category F1 cannot be computed or the time limit is exceeded, score 0.

## Abridged statement

Train a PyTorch text classifier using `text` to predict `category`. The task provides 1,000 labeled training articles and 200 unlabeled test articles. CPU training plus testing must finish within 10 minutes, and the score is macro-F1 averaged over categories; failing to calculate per-category F1 or exceeding the time limit gives zero. The official statement recommends word embeddings with an LSTM, but the classifier architecture is otherwise open. Submit `submission.ipynb` containing the training process and a `submission.csv` whose category labels follow the naming and storage convention of `train_news.csv`.

## Dataset analysis and EDA

The community mirror contains 1,000 rows in four categories: business 290, sport 284, tech 221, and entertainment 205. Article lengths range from 742 to 19,138 characters (mean 2,159.8). There are 26 exact duplicate texts; the local validation therefore groups identical text by SHA-256 so copies cannot fall on opposite sides of a fold. The hidden 200-row test file was not present.

## Experiments and selected method

[news_text_classification.py](../../../solutions/china/2024/news_text_classification.py) builds word TF-IDF features (unigrams/bigrams) and character-boundary n-grams (3–5 characters), capped at 30,000 features per vectorizer. It trains a sparse multiclass linear classifier with PyTorch sparse matrix multiplication and CrossEntropyLoss. No pretrained weights or external text are used. Submit mode reads the unlabeled CSV, predicts `category`, preserves the test rows and their order, and writes `submission.csv` with the same label column name as training. This authored script is a reproducible code path, not the official notebook file; the example organizer run is still needed to assemble the complete required `submission.ipynb` and CSV artifacts.

| Result | Macro-F1 | Provenance |
|---|---:|---|
| Five-fold duplicate-aware OOF | 0.9724 | Locally measured on the 1,000-row community mirror using StratifiedGroupKFold and exact-text groups; fold values 0.9666–0.9796. |
| Participant LSTM, leaderboard A/B | 0.8667 / 0.8286 | Per-category F1 and aggregate scores reported in a public community notebook; not reproduced here. |

The folds are stratified by category. Each fold fits its vectorizers on the training portion only, then trains the PyTorch head for 160 full-batch epochs. The OOF score is a validation estimate on the mirror, not a hidden-test score.

## Result, compute, and limits

The complete five-fold diagnostic finished in 82 seconds on CPU. A single fit uses one fold's vectorization and 160 PyTorch epochs, but the official hidden test path and exact grader runtime were not measured. The task's 10-minute CPU limit is therefore not claimed as organizer-verified. **Official leaderboard performance for this implementation is unmeasured.**

The SOTA page lists the organizer source license as unstated. The public community repository also has no declared license (license: null). The mirrored article text was used temporarily from /tmp; no source articles, labels, or notebooks are reproduced in this repository.

## Alternatives considered

- The community LSTM follows the statement's recommendation but reported lower public leaderboard macro-F1.
- Word-only TF-IDF is simpler; character features can preserve useful spelling, suffix, and name patterns.
- A pretrained language model would add download, runtime, and licensing constraints without being needed for this small dataset.

## Progressive hints

1. Check the category counts and text lengths before selecting a model.
2. Fit vectorizers separately inside each validation fold.
3. Combine word and character n-grams, then train a small multiclass PyTorch head.
4. Group exact duplicate articles before measuring cross-validation macro-F1.

**One-line summary:** A PyTorch sparse TF-IDF classifier scores 0.9724 macro-F1 in duplicate-aware local cross-validation; organizer hidden-test score is unmeasured.

## Sources and reuse

- [SOTA task page and official-material links](https://checklist.sota-ai.org/problems/noai-china-2024-round-2-news-text-classification/)
- [Bohrium task page](https://www.bohrium.com/en/competitions/2223242868)
- [Community LSTM notebook and mirrored training data](https://github.com/jaredliw/ioai-tsp-2025/blob/main/noai-china-2024/news-text-classification/news-text-classification-LSTM.ipynb)

This report paraphrases the official task and links to sources. It does not republish the BBC-derived article text, dataset, or notebook; their license is not stated by the cited pages.
