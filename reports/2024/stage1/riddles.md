# Riddles

## Task and metric

- **Domain:** Polish-language NLP, dictionary retrieval and semantic ranking.
- **Metric:** mean reciprocal rank at K=20; the task awards 1.5 points for MRR above 0.3.
- **Abridged statement:** given a tokenized Polish riddle, rank up to 20 single-word candidate answers. The official notebook states that every answer is present in a corpus of Wiktionary definitions and supplies morphology mappings plus Polish Word2Vec embeddings. The final environment is CPU-only and must answer 100 riddles in two minutes without internet.

## EDA and asset availability

The [official notebook and validator](https://github.com/OlimpiadaAI/I-OlimpiadaAI/tree/main/first_stage/riddles) describe roughly 2,000 example riddles, definitions for 8,094 frequent nouns, a word-to-lemma file, and a Gensim Word2Vec model. These assets are loaded from Google Drive, but the text, morphology, and embedding files were not present in the available local mirrors. The official validator evaluates the first 100 query/answer pairs after loading those assets.

## Experiments, method, and score

No local MRR can be reported: the input riddles, answer labels, definition dictionary, and embedding model are all absent. The score is **unmeasured**, not zero. I drafted a deterministic retrieval method for the notebook globals: normalize query tokens to lemmas; rank candidates by the strongest definition-level combination of IDF-weighted lexical overlap and cosine similarity between query and definition Word2Vec centroids; return the top K distinct dictionary keys. This uses the intended offline resources and requires no network or GPU. It could not be benchmarked for accuracy or runtime in this workspace.

## Alternatives considered

Exact token overlap alone is fast but misses paraphrases. A hosted language model violates the offline evaluation constraint. A large generative model is unnecessary if the provided definitions and embeddings can be retrieved efficiently.

## Progressive hints

1. Treat the dictionary definitions as a candidate index rather than generating answers.
2. Lemmatize both riddle and definition tokens so inflection does not hide matches.
3. Give rare overlapping terms more weight through the supplied inverse-document-frequency values.
4. Add Word2Vec cosine similarity for paraphrases, aggregate by the best candidate sense, and cache candidate vectors for the 100-query run.

**One-line summary:** the retrieval implementation is drafted, but the required corpus and embeddings are missing, so accuracy and runtime remain unverified.
