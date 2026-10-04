"""Offline pictographic substitution-cipher solver for the 2024 OAI final.

Each plaintext character is represented by three emoji glyphs. The solver
groups glyphs by their contexts, aligns group and plaintext unigram frequencies,
then refines that alignment with a character trigram model learned from the
provided clear corpus. It does not read or require ground-truth ciphertext.
"""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


SEED = 42
CONTEXT_WINDOW = 4
EMBEDDING_DIM = 50
LM_SAMPLE_LINES = 2000
LM_SWAP_STEPS = 600


def _context_embeddings(ciphered_corpus: Sequence[str], tokens: list[str]) -> np.ndarray:
    """Positive-PMI spectral embeddings of glyph left/right contexts."""
    size = len(tokens)
    token_id = {token: i for i, token in enumerate(tokens)}
    sequences = [np.fromiter((token_id[c] for c in line), dtype=np.int32) for line in ciphered_corpus]
    cooc = np.zeros((size, size), dtype=np.float64)
    for distance in range(1, CONTEXT_WINDOW + 1):
        left_parts = [seq[:-distance] for seq in sequences if len(seq) > distance]
        right_parts = [seq[distance:] for seq in sequences if len(seq) > distance]
        if not left_parts:
            continue
        left = np.concatenate(left_parts)
        right = np.concatenate(right_parts)
        weights = np.full(left.shape[0], 1.0 / distance, dtype=np.float64)
        forward = np.bincount(left.astype(np.int64) * size + right, weights=weights, minlength=size * size)
        backward = np.bincount(right.astype(np.int64) * size + left, weights=weights, minlength=size * size)
        cooc += (forward + backward).reshape(size, size)

    total = cooc.sum()
    row = cooc.sum(axis=1, keepdims=True)
    col = cooc.sum(axis=0, keepdims=True)
    denominator = np.maximum(row @ col, 1e-12)
    ppmi = np.maximum(np.log((cooc * max(total, 1.0) + 1e-12) / denominator), 0.0)
    dim = min(EMBEDDING_DIM, size - 1)
    if dim <= 0 or not np.any(ppmi):
        return np.eye(size, dtype=np.float64)
    embedding = TruncatedSVD(n_components=dim, n_iter=10, random_state=SEED).fit_transform(ppmi)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    return embedding


def _balanced_clusters(embedding: np.ndarray, group_count: int) -> np.ndarray:
    """Assign exactly three glyphs to every cluster by repeated Hungarian match."""
    size = len(embedding)
    if size != 3 * group_count:
        raise ValueError("A valid cipher must contain exactly three glyphs per plaintext character.")
    labels = KMeans(n_clusters=group_count, n_init=8, max_iter=100, random_state=SEED).fit_predict(embedding)
    centers = np.vstack([embedding[labels == i].mean(axis=0) for i in range(group_count)])
    for _ in range(12):
        slots = np.repeat(centers, 3, axis=0)
        costs = cdist(embedding, slots, metric="sqeuclidean")
        glyph_rows, slot_cols = linear_sum_assignment(costs)
        updated = np.empty_like(centers)
        new_labels = np.empty(size, dtype=np.int32)
        new_labels[glyph_rows] = slot_cols // 3
        for i in range(group_count):
            updated[i] = embedding[new_labels == i].mean(axis=0)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels, centers = new_labels, updated
    return labels


class _TrigramLanguageModel:
    def __init__(self, clear_corpus: Sequence[str], alphabet: list[str], alpha: float = 0.15):
        self.alphabet = alphabet
        self.vocab_size = len(alphabet)
        self.char_id = {char: i for i, char in enumerate(alphabet)}
        self.alpha = alpha
        self.counts: Counter[tuple[int, int, int]] = Counter()
        self.contexts: Counter[tuple[int, int]] = Counter()
        unknown = self.vocab_size - 1
        for line in clear_corpus:
            ids = [self.char_id.get(char, unknown) for char in line]
            for i in range(2, len(ids)):
                context = (ids[i - 2], ids[i - 1])
                self.contexts[context] += 1
                self.counts[(context[0], context[1], ids[i])] += 1

        # Dense lookup keeps language-model refinement fast for the small
        # character alphabets used by this task.
        v = self.vocab_size
        self.log_probs = np.full(v**3, math.log(alpha / (alpha * v)), dtype=np.float64)
        for (a, b), count in self.contexts.items():
            start = (a * v + b) * v
            self.log_probs[start : start + v] = math.log(alpha / (count + alpha * v))
        for (a, b, c), count in self.counts.items():
            index = (a * v + b) * v + c
            context_count = self.contexts[(a, b)]
            self.log_probs[index] = math.log((count + alpha) / (context_count + alpha * v))

    def score(self, sequences: Sequence[np.ndarray], class_to_char: Sequence[str]) -> float:
        char_to_id = self.char_id
        unknown = self.vocab_size - 1
        lookup = np.fromiter((char_to_id.get(c, unknown) for c in class_to_char), dtype=np.int32)
        total = 0.0
        v = self.vocab_size
        for sequence in sequences:
            if len(sequence) < 3:
                continue
            decoded = lookup[sequence]
            indices = (decoded[:-2] * v + decoded[1:-1]) * v + decoded[2:]
            total += float(self.log_probs[indices].sum())
        return total


def _refine_mapping(
    clear_corpus: Sequence[str],
    ciphered_corpus: Sequence[str],
    initial: list[str],
    sequences: Sequence[np.ndarray],
) -> list[str]:
    """Hill-climb ciphertext-group to character assignments under clear text."""
    alphabet = list(dict.fromkeys(initial))
    lm_alphabet = alphabet + ["\u0000<unk>"]
    lm = _TrigramLanguageModel(clear_corpus, lm_alphabet)

    rng = random.Random(SEED)
    ids = list(range(len(ciphered_corpus)))
    rng.shuffle(ids)
    selected = ids[: min(LM_SAMPLE_LINES, len(ids))]
    sampled_sequences = [sequences[i] for i in selected]

    current = initial[:]
    current_score = lm.score(sampled_sequences, current)
    best, best_score = current[:], current_score
    n = len(current)
    if n < 2:
        return current

    # Frequency matching already gives a good start. A low temperature makes
    # the language model correct occasional ranking ties without random drift.
    for step in range(LM_SWAP_STEPS):
        a = rng.randrange(n)
        b = rng.randrange(n - 1)
        if b >= a:
            b += 1
        candidate = current[:]
        candidate[a], candidate[b] = candidate[b], candidate[a]
        candidate_score = lm.score(sampled_sequences, candidate)
        temperature = 0.02 * (1.0 - step / LM_SWAP_STEPS) + 1e-5
        if candidate_score >= current_score or rng.random() < math.exp((candidate_score - current_score) / temperature):
            current, current_score = candidate, candidate_score
        if current_score > best_score:
            best, best_score = current[:], current_score
    return best


def decipher_corpus(clear_corpus: list[str], ciphered_corpus: list[str]) -> list[str]:
    """Decipher pictographic lines using only clear text and ciphertext inputs."""
    if not ciphered_corpus:
        return []
    clear = [str(line).lower() for line in clear_corpus]
    ciphered = [str(line) for line in ciphered_corpus]
    glyphs = sorted({glyph for line in ciphered for glyph in line})
    if not glyphs:
        return [""] * len(ciphered)
    if len(glyphs) % 3:
        raise ValueError(f"Cipher alphabet has {len(glyphs)} glyphs; expected a multiple of three.")

    group_count = len(glyphs) // 3
    embedding = _context_embeddings(ciphered, glyphs)
    glyph_groups = _balanced_clusters(embedding, group_count)
    token_to_group = {token: int(group) for token, group in zip(glyphs, glyph_groups)}

    encoded = [np.fromiter((token_to_group[c] for c in line), dtype=np.int32) for line in ciphered]
    group_counts = np.zeros(group_count, dtype=np.float64)
    for sequence in encoded:
        group_counts += np.bincount(sequence, minlength=group_count)
    clear_counts = Counter(char for line in clear for char in line)
    alphabet = [char for char, _ in clear_counts.most_common(group_count)]
    fallback = " etaoinshrdlucmfwypvbgkqjxz0123456789.,;:'!?-()[]\"%*_/"
    for char in fallback:
        if len(alphabet) == group_count:
            break
        if char not in alphabet:
            alphabet.append(char)
    if len(alphabet) < group_count:
        alphabet.extend(chr(0xE000 + i) for i in range(group_count - len(alphabet)))

    group_order = np.argsort(-group_counts, kind="stable")
    initial = [""] * group_count
    for rank, group in enumerate(group_order):
        initial[int(group)] = alphabet[rank]
    mapping = _refine_mapping(clear, ciphered, initial, encoded)

    return ["".join(mapping[int(group)] for group in sequence) for sequence in encoded]
