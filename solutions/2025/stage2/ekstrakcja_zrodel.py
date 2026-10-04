"""768D corpus-fitted BM25-style embeddings for the source retrieval task.

If the official SGPT checkpoint is cached, the encoder uses it. Otherwise it
builds compact signed-hash vectors from a Porter-stemmed BM25 representation;
the fallback requires only the standard library, NumPy, and PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch


MODEL_ID = "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit"
_WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an the and or but if while of at by for with about against between into "
    "through during before after above below to from up down in out on off over "
    "under again further then once here there when where why how all any both "
    "each few more most other some such no nor not only own same so than too very "
    "can will just is are was were be been being do does did doing have has had "
    "having it its this that these those as we you your i he she they them their "
    "our us".split()
)


def _consonant(word: str, index: int) -> bool:
    char = word[index]
    if char in "aeiou":
        return False
    if char == "y":
        return index == 0 or not _consonant(word, index - 1)
    return True


def _measure(word: str) -> int:
    count = 0
    index = 0
    size = len(word)
    while index < size and _consonant(word, index):
        index += 1
    while index < size:
        while index < size and not _consonant(word, index):
            index += 1
        if index >= size:
            break
        count += 1
        while index < size and _consonant(word, index):
            index += 1
    return count


def _has_vowel(word: str) -> bool:
    return any(not _consonant(word, index) for index in range(len(word)))


def _cvc(word: str) -> bool:
    size = len(word)
    return (
        size >= 3
        and _consonant(word, size - 1)
        and not _consonant(word, size - 2)
        and _consonant(word, size - 3)
        and word[-1] not in "wxy"
    )


def _porter_stem(word: str) -> str:
    """Small self-contained implementation of the classic Porter suffix rules."""
    if len(word) <= 2:
        return word

    for suffix in ("'s'", "'s", "'"):
        if word.endswith(suffix):
            word = word[: -len(suffix)]
            break

    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif not word.endswith("ss") and word.endswith("s"):
        word = word[:-1]

    changed = False
    if word.endswith("eed"):
        stem = word[:-3]
        if _measure(stem) > 0:
            word = stem + "ee"
    elif word.endswith("ed") and _has_vowel(word[:-2]):
        word = word[:-2]
        changed = True
    elif word.endswith("ing") and _has_vowel(word[:-3]):
        word = word[:-3]
        changed = True

    if changed:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif (
            len(word) >= 2
            and word[-1] == word[-2]
            and _consonant(word, len(word) - 1)
            and word[-1] not in "lsz"
        ):
            word = word[:-1]
        elif _measure(word) == 1 and _cvc(word):
            word += "e"

    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    step_two = (
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
        ("alli", "al"), ("entli", "ent"), ("eli", "e"),
        ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
        ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
        ("iviti", "ive"), ("biliti", "ble"), ("logi", "log"),
    )
    for suffix, replacement in step_two:
        if word.endswith(suffix) and _measure(word[: -len(suffix)]) > 0:
            word = word[: -len(suffix)] + replacement
            break

    step_three = (
        ("icate", "ic"), ("ative", ""), ("alize", "al"),
        ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
    )
    for suffix, replacement in step_three:
        if word.endswith(suffix) and _measure(word[: -len(suffix)]) > 0:
            word = word[: -len(suffix)] + replacement
            break

    for suffix in (
        "ement", "ance", "ence", "able", "ible", "ment", "ant", "ent",
        "al", "er", "ic", "ou", "ism", "ate", "iti", "ous", "ive", "ize", "ful",
    ):
        if word.endswith(suffix) and _measure(word[: -len(suffix)]) > 1:
            word = word[: -len(suffix)]
            break
    else:
        if (
            word.endswith("ion")
            and len(word) > 3
            and word[-4] in "st"
            and _measure(word[:-3]) > 1
        ):
            word = word[:-3]

    if word.endswith("e"):
        stem = word[:-1]
        measure = _measure(stem)
        if measure > 1 or (measure == 1 and not _cvc(stem)):
            word = stem
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


class Embedder:
    """Encode queries and papers into a shared 768-dimensional cosine space."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._doc_count = 0
        self._title_avglen = 1.0
        self._body_avglen = 1.0
        self._title_idf: dict[str, float] = {}
        self._body_idf: dict[str, float] = {}
        self._fit_stats_from_corpus()
        self._try_load_sgpt()

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            _porter_stem(token)
            for token in _WORD_RE.findall(text.lower())
            if token not in _STOPWORDS
        ]

    @staticmethod
    def _idf(df: Counter[str], document_count: int) -> dict[str, float]:
        return {
            term: math.log1p((document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in df.items()
        }

    def _fit_stats_from_corpus(self) -> None:
        """Fit field-specific document frequencies from unlabeled papers only."""
        path = Path("corpus.jsonl")
        if not path.is_file():
            return

        title_df: Counter[str] = Counter()
        body_df: Counter[str] = Counter()
        title_lengths = []
        body_lengths = []
        with path.open(encoding="utf8") as handle:
            for line in handle:
                item = json.loads(line)
                title = self._tokens(str(item.get("title") or ""))
                body = self._tokens(str(item.get("text") or ""))
                title_lengths.append(len(title))
                body_lengths.append(len(body))
                title_df.update(set(title))
                body_df.update(set(body))
                self._doc_count += 1

        if self._doc_count:
            self._title_avglen = float(np.mean(title_lengths))
            self._body_avglen = float(np.mean(body_lengths))
            self._title_idf = self._idf(title_df, self._doc_count)
            self._body_idf = self._idf(body_df, self._doc_count)

    def _try_load_sgpt(self) -> None:
        """Use the official pretrained model if it is already cached offline."""
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, local_files_only=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.model = AutoModel.from_pretrained(
                MODEL_ID, local_files_only=True
            ).to(self.device)
            self.model.eval()
        except (OSError, ValueError, ImportError):
            self.model = None
            self.tokenizer = None

    @staticmethod
    def _hash_index(term: str) -> tuple[int, float]:
        digest = hashlib.blake2b(term.encode("utf8"), digest_size=8).digest()
        code = int.from_bytes(digest, "little", signed=False)
        return code % 768, (1.0 if (code >> 63) == 0 else -1.0)

    def _bm25_field_vector(
        self,
        tokens: list[str],
        idf: dict[str, float],
        average_length: float,
    ) -> np.ndarray:
        vector = np.zeros(768, dtype=np.float32)
        frequencies = Counter(tokens)
        length_norm = 2.0 * (1.0 - 0.75 + 0.75 * len(tokens) / max(average_length, 1e-8))
        for term, frequency in frequencies.items():
            weight = idf.get(term)
            if weight is None:
                continue
            value = weight * frequency * 3.0 / (frequency + length_norm)
            index, sign = self._hash_index(term)
            vector[index] += sign * value
        return vector

    def _encode_fallback(self, texts: list, corpus: bool) -> torch.Tensor:
        vectors = []
        for item in texts:
            if corpus:
                title = self._tokens(str(item.get("title") or ""))
                body = self._tokens(str(item.get("text") or ""))
                title_vec = self._bm25_field_vector(
                    title, self._title_idf, self._title_avglen
                )
                body_vec = self._bm25_field_vector(
                    body, self._body_idf, self._body_avglen
                )
                vectors.append(title_vec + body_vec)
            else:
                vector = np.zeros(768, dtype=np.float32)
                for term in set(self._tokens(str(item))):
                    idf = self._body_idf.get(term)
                    if idf is None:
                        continue
                    index, sign = self._hash_index(term)
                    vector[index] += sign * math.sqrt(idf)
                vectors.append(vector)

        if not vectors:
            return torch.empty((0, 768), dtype=torch.float32, device=self.device)
        return torch.as_tensor(np.stack(vectors), dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def _encode_sgpt(self, texts: list[str]) -> torch.Tensor:
        batches = []
        for start in range(0, len(texts), 32):
            encoded = self.tokenizer(
                texts[start : start + 32],
                max_length=150,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].to(hidden.dtype)
            weights = torch.arange(
                1, hidden.shape[1] + 1, dtype=hidden.dtype, device=hidden.device
            ).unsqueeze(0) * mask
            pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            batches.append(torch.nn.functional.normalize(pooled, dim=1))
        return torch.cat(batches, dim=0)

    def encode_queries(self, queries: list[str]) -> torch.Tensor:
        if self.model is None:
            return self._encode_fallback(queries, corpus=False)
        return self._encode_sgpt(queries)

    def encode_corpus(self, texts: list[dict]) -> torch.Tensor:
        if self.model is None:
            return self._encode_fallback(texts, corpus=True)
        joined = [
            f"{item.get('title', '')}. {item.get('text', '')}" for item in texts
        ]
        return self._encode_sgpt(joined)
