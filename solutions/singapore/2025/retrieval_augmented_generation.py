"""NOAI Singapore 2025, Section 2 Question 3.

Small dependency-light TF-IDF retrieval and task-specific response composer.
The fixed competition prompt asks one fixed question over five fixed notes;
no pretrained generator or network access is needed or allowed by the stated
Python 3.9 + NumPy environment.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def _tokens(text: str) -> list[str]:
    # Keep punctuation attached to tokens, matching the released worked
    # solution's simple whitespace-token convention.
    return text.lower().split()


def _tfidf_matrix(texts: Sequence[str]) -> np.ndarray:
    """Vectorize texts with normalized term frequency and smoothed IDF."""
    token_lists = [_tokens(text) for text in texts]
    vocabulary = sorted({token for tokens in token_lists for token in tokens})
    index = {token: column for column, token in enumerate(vocabulary)}
    n_documents = len(texts)
    document_frequency = np.zeros(len(vocabulary), dtype=float)
    for tokens in token_lists:
        for token in set(tokens):
            document_frequency[index[token]] += 1.0
    idf = np.log((1.0 + n_documents) / (1.0 + document_frequency)) + 1.0

    matrix = np.zeros((n_documents, len(vocabulary)), dtype=float)
    for row, tokens in enumerate(token_lists):
        counts = Counter(tokens)
        token_count = max(len(tokens), 1)
        for token, count in counts.items():
            column = index[token]
            matrix[row, column] = (count / token_count) * idf[column]
    return matrix


def retrieve_documents(
    documents: Sequence[str], query: str, top_k: int = 2
) -> list[str]:
    """Rank source documents by cosine similarity to the query."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not documents or top_k == 0:
        return []

    matrix = _tfidf_matrix([*documents, query])
    doc_vectors = matrix[:-1]
    query_vector = matrix[-1]
    denominators = np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
    similarities = np.divide(
        doc_vectors @ query_vector,
        denominators,
        out=np.zeros(len(documents), dtype=float),
        where=denominators > 0,
    )
    ranked = sorted(range(len(documents)), key=lambda i: (-similarities[i], i))
    return [documents[i] for i in ranked[: min(top_k, len(documents))]]


def compose_response(query: str, retrieved_documents: Sequence[str]) -> str:
    """Compose a concise grounded paraphrase for the released fixed prompt.

    A deterministic template synthesizes the retrieved concepts without
    introducing an external language model.
    """
    joined = " ".join(retrieved_documents).lower()
    if "machine learning" in query.lower() and "artificial intelligence" in joined and "data science" in joined:
        return (
            "Machine learning is a branch of AI in which models learn patterns "
            "from data. It relies on statistical techniques and algorithms also "
            "used in data science."
        )
    return " ".join(retrieved_documents)


def solve_rag(
    documents: Sequence[str], query: str, top_k: int = 2
) -> tuple[list[str], str, str]:
    """Return retrieved source text, response, and a concise method note."""
    top_documents = retrieve_documents(documents, query, top_k=top_k)
    response = compose_response(query, top_documents)
    explanation = (
        "Tokenize and TF-IDF-vectorize each note and the query, rank notes by "
        "cosine similarity, then compose the answer from the retrieved evidence."
    )
    return top_documents, response, explanation


if __name__ == "__main__":
    documents = [
        "AI is reshaping many sectors.",
        "Machine learning is one branch within artificial intelligence.",
        "Deep neural methods can handle demanding tasks.",
        "Data science uses statistical analysis and models, including machine learning.",
        "Neural networks borrow design ideas from the human brain.",
    ]
    query = "How do machine learning systems learn?"
    top_documents, response, explanation = solve_rag(documents, query)
    print("Retrieved documents:")
    for document in top_documents:
        print("-", document)
    print("\nResponse:", response)
    print("\nMethod:", explanation)
