# Ekstrakcja źródeł — Source Extraction

**Problem domain:** Scientific claim-to-paper retrieval  
**Official metric:** Mean nDCG@10 across 300 validation queries. A match at zero-based rank `r` contributes `1 / log2(r + 2)`; matches below rank 10 contribute zero. The point score is zero through nDCG `0.20`, 100 at `0.50` or higher, and linear between them.

## Abridged statement

Map each claim and scientific paper to a 768-dimensional real vector so cosine search ranks the matching paper among the top ten. The corpus has titles and abstracts; only the released validation file has matching paper IDs. The notebook allows the supplied SGPT GPT-2 checkpoint and requires inference to work offline within ten minutes on GPU.

## Data analysis

- The official release contains 5,183 papers and 300 validation claims; all 300 gold IDs exist in the corpus, with 204 unique gold papers.
- Claims average 12.0 whitespace-delimited words; median length is 82 characters and maximum is 179.
- Paper titles have median 94 characters. Abstracts have median 1,330 characters, 95th percentile 2,323, and maximum 10,000.
- The validation labels were used only to calculate nDCG and compare global representation choices. No matching IDs were used to fit embeddings or corpus statistics.

## Experiments

The following are local exploratory measurements on the released validation set. These first baselines used a 768-bucket signed `HashingVectorizer` plus corpus-fitted TF-IDF in a temporary analysis script; the submission implementation uses built-in Python hashing and NumPy/PyTorch, so it does not depend on scikit-learn.

| Local representation | nDCG@10 | Recall@10 | Point score |
|---|---:|---:|---:|
| Hashed word unigrams, title repeated 2×, abstract prefix | 0.37895 | 0.533 | 60 |
| Hashed word 1–2 grams | 0.33806 | 0.450 | 46 |
| Hashed word 1–3 grams | 0.29810 | 0.367 | 33 |
| Hashed character 3–5 grams | 0.38941 | 0.537 | 63 |
| Initial multi-block lexical fallback (words, bigrams, character grams) | 0.41575 | — | 72 |
| Unhashed BM25 diagnostic, Porter stemming, title weight 0.25, `k1=1.2`, `b=0.75` | 0.52751 | — | 100* |
| 768D signed-hash BM25 representation, title and abstract weights 1:1, query weight `sqrt(IDF)`, `k1=2`, `b=0.75` | **0.50365** | — | **100** |

The first five rows are exploratory hashed lexical baselines. BM25 tuning varied stemming, stopword filtering, `k1` in `{0.6, 1.2, 2.0}`, `b` in `{0.35, 0.75}`, and title-to-abstract weights; the hashed-vector sweep also varied query weighting between binary, IDF, and square-root IDF. The direct BM25 row is a lexical diagnostic only: its pairwise scores cannot be submitted through the required embedding interface. The 768D signed-hash BM25 row is the final fallback representation and is measured through the official notebook's cosine search and scoring functions. The local transformer checkpoint could not be downloaded: Hugging Face access returned proxy `403 Forbidden`, and no model files are cached. `*` applies the task's score curve to the diagnostic nDCG; it is not an official-interface result.

## Chosen method

[The solution](../../../solutions/2025/stage2/ekstrakcja_zrodel.py) prefers the exact SGPT model named by the official notebook when its files are already cached, using weighted mean pooling. If the checkpoint is unavailable, it returns 768D signed-hash vectors for Porter-stemmed BM25-style title and abstract terms. It fits field-specific document frequencies and average lengths from the unlabeled `corpus.jsonl`, uses the abstract IDF for query weights, and combines title and abstract vectors equally. It copies no model weights or data.

## Score, runtime, and evidence

Official notebook: the starter `Embedder` has placeholder encoders and reports no organizer benchmark result. The Rayan mirror contains a separate Qwen-based implementation, but its model is not present in the local cache. The final local score is from the lexical fallback, not a transformer result.

Using the official `search_topk_texts`, `evaluate_retrieval_ndcg`, and `compute_score` functions, the final fallback scored **nDCG@10 0.50365 / 100 points** on all 300 released validation queries. Embedder initialization and corpus-statistics fitting took `5.96` seconds; encoding and search took `4.70` seconds on CPU. CUDA is unavailable, so GPU runtime was not measured. This score is for the fallback; the SGPT branch could not be tested because the checkpoint was unavailable. Hidden-test performance is unknown.

The hidden-test score is unknown. The offline competition image's SGPT cache status is also unknown; when the model is unavailable, the fallback is self-contained and reproduced full credit on released validation.

## Alternatives considered

- A pretrained dense SGPT representation is the official intended path, but its weights are not available on this host, so its score could not be measured.
- Hashed word bigrams and trigrams lowered nDCG in the first sweep, likely due to collisions in only 768 dimensions.
- Character n-grams improved the first unigram baseline but underperformed the tuned BM25-style fallback.
- Corpus-only direct BM25 achieved 0.52751 nDCG in a diagnostic pairwise ranker, but it is not deployable through the task's 768D embedding API; a signed-hash BM25 approximation retained 0.50365 in the official cosine-search path.
- A larger neural language model would require large weights and more compute, contrary to the instruction not to copy checkpoints and the offline evaluation constraint.

## Progressive hints

1. Use a shared 768D space and L2-normalize outputs because search uses cosine similarity.
2. Encode the title and abstract; title-only retrieval leaves useful abstract evidence unused.
3. Fit title and abstract BM25 statistics from the unlabeled corpus; stem terms so morphological variants match.
4. Hash BM25-weighted unigram vectors into the required 768D space, and give the query an IDF-derived weight.
5. Test all encoders against the released query-to-paper labels, while keeping those labels out of model fitting.

**One-line summary:** A self-contained 768D signed-hash BM25 fallback scores **100/100 on released validation**; SGPT and hidden-test performance remain unverified.
