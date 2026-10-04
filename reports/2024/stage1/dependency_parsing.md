# Dependency Parsing — Stage I (2024)

**Problem name:** Dependency Parsing (`dependency_parsing`)

**Domain:** Polish NLP, unsupervised-style dependency induction with labeled training data

**Metric:** Average UUAS (unlabeled undirected attachment score) and root-placement accuracy, each mapped linearly from 0.50 to 0.85 and clipped at that range. The sum is worth up to 2 points; full credit requires both metrics to be at least 0.85.

## Abridged statement

Train a parser for Polish sentences using the released 1,000-sentence labeled training set. Predict a dependency tree and root position for each sentence. The official interface permits the listed HerBERT pretrained model and requires CPU evaluation; training and evaluation have organizer time limits. The 200-sentence validation split is used only for evaluation.

## Dataset analysis and EDA

The official repository contains `train.conll` (1,000 sentences) and `valid.conll` (200 sentences), plus the organizer's `utils.py` and `validation_script.py`. These assets were retrieved for inspection and not copied into the solution directory. The train set has 9,663 tokens (mean 9.66 words/sentence, median 8, maximum 39); validation has 1,821 tokens (mean 9.11, median 8, maximum 45). The annotated root is among the first three positions in 53.4% of training sentences and 55.0% of validation sentences. The code reads gold heads from the CoNLL files and constructs undirected tree-distance and root-depth targets from training sentences only.

## Experiments

- **Official starter / evaluator inspection:** reviewed the starter notebook, its TODO interface, the supplied split files, and `validation_script.py`. No official worked solution or organizer-reported score was found in the inspected materials.
- **Data-only baselines:** no heuristic baseline was assigned a validation score. Root-position frequencies show only a weak early-root prior; they are descriptive, not a measured parser result.
- **HerBERT representation plus distance/depth regressors:** implemented as the selected route. The frozen `allegro/herbert-base-cased` word representations feed one symmetric pairwise tree-distance regressor and one root-depth regressor. A minimum spanning tree recovers undirected edges and the predicted minimum-depth token is selected as root. This code was not run because neither `transformers` nor the allowed HerBERT checkpoint/tokenizer is installed or cached locally, and shell network access is unavailable.

## Selected solution

The candidate implementation is [dependency_parsing.py](../../../solutions/2024/stage1/dependency_parsing.py). It loads the official HerBERT checkpoint through Transformers, averages wordpiece vectors for each word, fits the distance and depth heads from `train.conll`, saves compact head state dictionaries, and evaluates against `valid.conll`. It does not load validation data during training. The source is a complete execution route conditional on the organizer-allowed Transformers package and pretrained checkpoint being available; it has not been dependency-resolved or runtime-verified in this environment.

## Validation score and evidence

**No local validation score was measured.** The exact blocker is that this environment has no `transformers` installation and no cached `allegro/herbert-base-cased` weights/tokenizer, while outbound shell downloads are unavailable. The official 200-sentence validation split and scoring script are available, but scoring without the required model would not be meaningful. No organizer-reported score was identified. Therefore there is no basis to claim full credit or hidden-test performance.

## Compute and runtime

The supplied route freezes HerBERT and trains only two small regression heads, with CPU selected automatically when CUDA is unavailable. The code defaults to 40 epochs per head. Because the pretrained model could not be loaded, training time, CPU evaluation time, and compliance with the official training/evaluation time limits remain unverified. The task allows only CPU evaluation; no GPU-specific route is required.

## Alternatives

Potential routes include a majority root-position prior, a supervised biaffine arc parser with HerBERT features, or a transition-based parser. None was trained or scored here. A cached organizer-allowed HerBERT checkpoint is required to evaluate the selected approach and make a fair comparison.

## Progressive hints

1. Parse the CoNLL head column and check how many roots each sentence has.
2. Dependency structure can be represented through pairwise tree distances and each token's depth from the root.
3. Contextual word representations can provide features for both distance and root-depth prediction.
4. Recover an undirected tree from predicted pairwise distances with an MST, then choose the token with minimum predicted depth as root.

**One-line solution:** Use frozen HerBERT word vectors to predict pairwise tree distances and root depths, then recover the tree with an MST and choose the shallowest token as root.
