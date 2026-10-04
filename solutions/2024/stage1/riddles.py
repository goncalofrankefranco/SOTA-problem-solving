"""Definition-retrieval baseline for the 2024 Polish AI Olympiad Riddles task.

In the official notebook, wrap one RiddleRanker after loading the provided
definitions, morphology table, IDF values, and Word2Vec model, then expose its
`predict` method as `answer_riddle`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class RiddleRanker:
    """Rank dictionary headwords by definition overlap and embedding similarity."""

    def __init__(self, definitions, bases, idf, word2vec):
        vectors = word2vec.wv if hasattr(word2vec, "wv") else word2vec
        self.vectors = vectors
        self.words = list(definitions.keys())
        self.bases = bases
        self.idf = idf

        candidate_ids = []
        definition_tokens = []
        definition_vectors = []
        postings = defaultdict(list)
        headword_vectors = np.zeros((len(self.words), vectors.vector_size), dtype=np.float32)

        for candidate_id, word in enumerate(self.words):
            if word in vectors.key_to_index:
                headword_vectors[candidate_id] = vectors.get_vector(word)
            for raw_definition in definitions[word]:
                tokens = {self._lemma(token) for token in raw_definition
                          if self._valid_token(token)}
                if not tokens:
                    continue
                tokens_list = sorted(tokens)
                weights = np.asarray([self._weight(token) for token in tokens_list],
                                     dtype=np.float32)
                vectors_for_tokens = [vectors.get_vector(token)
                                      if token in vectors.key_to_index else None
                                      for token in tokens_list]
                present = [(weight, vector) for weight, vector in zip(weights, vectors_for_tokens)
                           if vector is not None]
                if present:
                    total = sum(weight for weight, _ in present)
                    if total <= 0:
                        total = float(len(present))
                        vector = np.mean([v for _, v in present], axis=0)
                    else:
                        vector = sum(weight * v for weight, v in present) / total
                    norm = float(np.linalg.norm(vector))
                    if norm:
                        vector = vector / norm
                    else:
                        vector = np.zeros(vectors.vector_size, dtype=np.float32)
                else:
                    vector = np.zeros(vectors.vector_size, dtype=np.float32)

                definition_id = len(definition_tokens)
                candidate_ids.append(candidate_id)
                definition_tokens.append(tokens)
                definition_vectors.append(np.asarray(vector, dtype=np.float32))
                for token in tokens:
                    postings[token].append(definition_id)

        self.definition_tokens = definition_tokens
        self.definition_candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
        self.definition_vectors = (np.stack(definition_vectors) if definition_vectors
                                   else np.zeros((0, vectors.vector_size), dtype=np.float32))
        self.postings = dict(postings)
        self.headword_vectors = headword_vectors

    def _lemma(self, token):
        token = str(token).lower()
        return self.bases.get(token, token).lower()

    @staticmethod
    def _valid_token(token):
        return any(character.isalpha() for character in str(token))

    def _weight(self, token):
        return max(0.0, float(self.idf.get(token, 0.0)))

    def _query_tokens(self, riddle):
        return {self._lemma(token) for token in riddle if self._valid_token(token)}

    def predict(self, riddle, K=20):
        """Return up to K distinct headwords, highest retrieval score first."""
        if not self.words or K <= 0:
            return []
        query = self._query_tokens(riddle)
        if not query:
            return self.words[:K]

        query_weights = {token: self._weight(token) for token in query}
        if sum(query_weights.values()) <= 0:
            query_weights = {token: 1.0 for token in query}
        normalizer = sum(query_weights.values()) or 1.0

        lexical = np.zeros(len(self.definition_tokens), dtype=np.float32)
        for token, weight in query_weights.items():
            for definition_id in self.postings.get(token, ()):
                lexical[definition_id] += weight / normalizer

        query_vectors = [(query_weights[token], self.vectors.get_vector(token))
                         for token in query if token in self.vectors.key_to_index]
        if query_vectors:
            total = sum(weight for weight, _ in query_vectors)
            query_vector = sum(weight * vector for weight, vector in query_vectors) / total
            norm = float(np.linalg.norm(query_vector))
            if norm:
                query_vector = query_vector / norm
            semantic = self.definition_vectors @ query_vector
            semantic = np.maximum(semantic, 0.0)
            headword = self.headword_vectors @ query_vector
            headword = np.maximum(headword, 0.0)
        else:
            semantic = np.zeros(len(self.definition_tokens), dtype=np.float32)
            headword = np.zeros(len(self.words), dtype=np.float32)

        definition_scores = 0.62 * lexical + 0.30 * semantic
        candidate_scores = np.full(len(self.words), -1.0, dtype=np.float32)
        if len(definition_scores):
            np.maximum.at(candidate_scores, self.definition_candidate_ids, definition_scores)
        candidate_scores += 0.08 * headword

        take = min(len(self.words), max(K, 1))
        top = np.argpartition(-candidate_scores, take - 1)[:take]
        top = top[np.argsort(-candidate_scores[top], kind="stable")]
        return [self.words[i] for i in top[:K]]


_DEFAULT_RANKER = None


def configure_riddle_ranker(definitions, bases, idf, word2vec):
    """Build and cache the search index once, before evaluating the queries."""
    global _DEFAULT_RANKER
    _DEFAULT_RANKER = RiddleRanker(definitions, bases, idf, word2vec)
    return _DEFAULT_RANKER


def answer_riddle(riddle, K, ranker=None):
    """Notebook-compatible function after calling ``configure_riddle_ranker``."""
    active_ranker = ranker or _DEFAULT_RANKER
    if active_ranker is None:
        raise RuntimeError("Call configure_riddle_ranker after loading the official corpus and model")
    return active_ranker.predict(riddle, K=K)
