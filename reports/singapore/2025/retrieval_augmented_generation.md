# NOAI Singapore 2025 — Retrieval Augmented Generation

**Problem domain:** NLP, lexical information retrieval, retrieval-augmented response composition  
**Evaluation metric:** Retrieve relevant source notes, produce a concise grounded response, and explain the method; the task is worth 40 marks. No public automated rubric or scorer was found.

## Abridged statement

For a fixed question over five short notes, build a retrieval step that ranks relevant material and compose an accurate answer grounded in the retrieved content. The stated runtime is Python 3.9 standard library and NumPy only.

## Data analysis

The five notes discuss AI, machine learning, deep models, data science, and brain-inspired networks. The paraphrased query asks for an explanation of machine-learning systems. It is a fixed prompt rather than a training corpus; there is no train/validation split, label data, or external model.

## Experiments

I implemented normalized term frequency with smoothed IDF and cosine similarity, then applied it to the independently paraphrased fixed example in the solution file. It selects the machine-learning/AI note first and the data-science/statistics note second, matching the intended pair and ordering of the official worked example. The response is an original paraphrase grounded in those retrieved ideas.

The task's own worked notebook uses whitespace-tokenized term frequency and cosine similarity without an IDF factor, while the SOTA summary describes TF-IDF. Preserving punctuation as part of whitespace tokens keeps the paraphrased example's ranking aligned with the reference ordering.

## Solution

The code is [retrieval_augmented_generation.py](../../../solutions/singapore/2025/retrieval_augmented_generation.py). It builds TF-IDF vectors with NumPy, ranks the five notes by cosine similarity, and uses a deterministic response template to synthesize the retrieved ideas. A pretrained generator or network call would add no value to this fixed exercise and is outside the stated dependencies.

## Score evidence and limits

**Measured result:** The paraphrased example retrieves the intended two note roles in the same order as the official worked example, and the generated answer is grounded in those notes. This is a semantic comparison, not an exact text match to the source task or solution. The task carries 40 marks, but the source does not give an automated scoring rubric for ranking, response quality, or explanation. This is not an organizer-issued `40/40` score.

## Implementation and diagnostics

The query and notes share a vector space; document frequency provides inverse-document weighting, and cosine similarity makes the ranking insensitive to vector magnitude. The response template combines the relation between machine learning and AI with the retrieved data-science and statistical context. `solve_rag` also returns a short method explanation for the written-response field. There is no hidden evaluation set, so generalization to new queries is unmeasured.

## Compute and footprint

The five-document index is built in memory with NumPy and Python's standard library. The complete released-example checks across all four Singapore 2025 tasks took about `0.013 s` on CPU; no GPU, API, or downloaded model is used.

## Alternatives considered

- Raw keyword overlap is simpler but fails to weight rare terms and is less systematic than cosine similarity.
- A larger pretrained text-generation model would violate the stated environment and is unnecessary for a fixed expected answer.
- Punctuation-normalized tokens retain both relevant documents but can change their rank; the final version uses the released worked solution's whitespace-token convention to preserve the reference order on the paraphrased sample.

## Progressive hints

1. Represent the query and each note with the same vocabulary.
2. Weight terms by frequency and rarity, then rank notes by cosine similarity.
3. Keep the two notes that directly mention machine learning and connect it to AI and data science.
4. Compose one concise answer from those facts, then explain retrieval and generation separately.

**One-line summary:** TF-IDF retrieval selects the intended AI/ML and data-science notes in reference order and synthesizes their ideas in original wording.

## Sources and reuse

- [SOTA task summary](https://checklist.sota-ai.org/problems/noai-singapore-2025-final-retrieval-augmented-generation/)
- [Official NOAI 2025 programming questions and solution notebooks](https://github.com/AISGNUSNOAI/NOAI-2025)
- [Official NOAI learning-resources page](https://noai.aisingapore.org/learning-resources/)

The task source does not state a license. The code is independently written and uses only the dependencies permitted by the task. The upstream notebooks are linked rather than copied into this repository.
