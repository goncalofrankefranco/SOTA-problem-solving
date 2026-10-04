# Ciphers — 2024 Polish AI Olympiad Final

**Problem domain:** Unsupervised substitution-cipher recovery / character language modeling  
**Evaluation metric:** Character accuracy. The official hidden evaluation averages four independent ciphers; each contributes `2 * max(character_accuracy - 0.50, 0)`, with a final score from 0 to 1. Runtime is limited to five minutes without Internet access.

## Abridged statement

Each plaintext character is replaced independently by one of three emoji glyphs assigned to that character. Given clear Polish text and encrypted lines, return the encrypted lines decoded to characters. The public corpus includes clear lines, 30,000 encrypted lines, and a decoded ground-truth file for evaluation; the ground truth must not be used to fit the decoder.

## Dataset analysis and EDA

The [official notebook and validation script](https://github.com/OlimpiadaAI/I-OlimpiadaAI/tree/main/final_stage/ciphers) are available in the organizers' repository. The linked `cipher.zip` assets (`clear_lines.txt`, `ciphered_lines.txt`, and `ciphered_lines_ground_truth.txt`) were not present in the local mirrors, and Google Drive access returned HTTP 403. Therefore glyph counts, alphabet size, line lengths, corpus statistics, and any local character-accuracy score could not be measured. The official checker reads the ground-truth file only after calling `decipher_corpus` and compares concatenated characters.

## Experiments and selected method

| Method | Evidence | Decision |
|---|---|---|
| Official identity skeleton | No score recorded | Baseline only; returns the encrypted glyphs unchanged. |
| Mirror write-up: Word2Vec + fixed-size constrained K-means + unigram-frequency alignment | No score or runtime recorded; adds six manual swaps | Inspected but not copied. The swaps have no documented provenance and may depend on a particular validation cipher. |
| **Context PPMI/SVD embeddings + exact three-per-group Hungarian clustering + frequency alignment + clear-text trigram refinement** | **Unverified**: required corpus absent | Implemented as a fully offline candidate; it reads no ground-truth file. |

The implementation first embeds glyphs from weighted left/right context co-occurrences. A constrained assignment repeatedly matches every glyph to one of three slots per cluster. It maps clusters to the most frequent clear-text characters by unigram rank, then hill-climbs character swaps using a trigram language model learned only from `clear_corpus`. Language-model refinement uses a deterministic sample of encrypted lines without labels. No validation targets or Internet resources are required.

Implementation: [ciphers.py](../../../solutions/2024/final/ciphers.py).

## Result, compute, and limits

**Score: unverified, not zero.** The source repository's validator cannot run without the three corpus files, and no participant notebook in the available local material records a score. The four-case hidden-test result is unknown. The implementation uses NumPy, SciPy, and scikit-learn, which are available in the supplied Python environment and are standard offline packages; it does not require gensim or `k_means_constrained`.

No runtime was measured. The implementation's major costs are corpus co-occurrence accumulation, several small exact-size Hungarian assignments, and 600 trigram-guided mapping swaps over up to 2,000 ciphertext lines. The official five-minute limit remains unverified until the actual corpus is available. Potential failure modes are context clusters that mix rare characters, frequency-rank drift between clear and encrypted lines, and local optima in the trigram swap search.

The mirror write-up's Word2Vec/constrained-KMeans idea is a reasonable alternative, but its manual swaps are not reproduced. A better validation set would compare both clustering approaches and tune the number of context neighbors while leaving ground truth out of selection.

## Progressive hints

1. Treat each emoji as a noisy spelling of a hidden plaintext character; all three variants should have similar character contexts.
2. Group glyphs from neighboring-glyph distributions, enforcing groups of exactly three.
3. Align common groups and common clear-text characters by frequency.
4. Use a character n-gram model trained on the clear corpus to resolve frequency ties and improve the substitution map.
5. Evaluate with the supplied ground truth only after the mapping is fixed; repeat across the four independent ciphers.

**One-line summary:** The offline contextual-clustering and trigram-refinement decoder is implemented, but its score and five-minute runtime remain unverified because the cipher corpus is missing.
