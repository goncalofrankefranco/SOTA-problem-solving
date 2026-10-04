"""Train-only implementation of Poland 2025 Stage I: hallucination detection.

The feature groups and XGBoost classifier follow the organizer's worked
notebook: token-probability statistics, TF-IDF similarities, answer agreement,
style, question type, and probabilities on extracted answer spans. TF-IDF is
fit only on the supplied training records. Validation labels are read only for
the final ROC-AUC report. The estimator uses the best parameters selected by
the organizer's three-fold search on training data.

Usage:
    python hallucination_detection.py train.json valid.json

Dependencies allowed by the task: numpy, scikit-learn, and xgboost. The script
does not download data, load pretrained encoders, or use external assets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


SUPPORT_COUNT = 4


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("JSON root must be a list of sample objects")
    return records


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def clean_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Trim malformed support token/probability tails at the first blank line."""
    cleaned = dict(sample)
    new_tokens: list[list] = []
    new_probabilities: list[list] = []
    tokens_by_answer = _as_list(sample.get("supporting_tokens"))
    probs_by_answer = _as_list(sample.get("supporting_probabilities"))
    for tokens, probabilities in zip(tokens_by_answer, probs_by_answer):
        tokens = _as_list(tokens)
        probabilities = _as_list(probabilities)
        cutoff = next((i for i, token in enumerate(tokens) if "\n\n" in str(token)), None)
        if cutoff is not None:
            tokens = tokens[: cutoff + 1]
            probabilities = probabilities[: cutoff + 1]
        common = min(len(tokens), len(probabilities))
        new_tokens.append(tokens[:common])
        new_probabilities.append(probabilities[:common])
    cleaned["supporting_tokens"] = new_tokens
    cleaned["supporting_probabilities"] = new_probabilities
    return cleaned


def _clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _answer_spans(text: str) -> list[str]:
    return re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)


class HallucinationFeatures:
    """All six organizer feature families in a single deterministic pipeline."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()
        self.feature_names_: list[str] | None = None
        self.is_fitted = False

    def fit(self, train_records: list[dict[str, Any]]) -> "HallucinationFeatures":
        corpus: list[str] = []
        for raw in train_records:
            sample = clean_sample(raw)
            corpus.append(_clean_text(_text(sample.get("question"))))
            corpus.append(_clean_text(_text(sample.get("answer"))))
            corpus.extend(
                _clean_text(_text(answer))
                for answer in _as_list(sample.get("supporting_answers"))
            )
        if not corpus or not any(corpus):
            raise ValueError("training data contains no usable text for TF-IDF")
        self.vectorizer.fit(corpus)
        self.is_fitted = True
        return self

    @staticmethod
    def _prob_stats(values: Any) -> tuple[float, float, float, float, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            float(arr.min()),
            float(arr.mean()),
            float(arr.max() - arr.min()),
            float(-np.log(arr + 1e-12).sum() / arr.size),
            float(arr.var()),
        )

    @staticmethod
    def _token_answer_probability(tokens: list, probabilities: list) -> tuple[float, str]:
        try:
            start = tokens.index("<answer>")
            end = tokens.index("</answer>")
        except ValueError:
            return 0.0, ""
        if end < start:
            start, end = end, start
        left, right = start + 1, end
        span_probs = probabilities[left:right]
        span_tokens = tokens[left:min(right, len(tokens))]
        if not span_probs or not span_tokens:
            return 0.0, "".join(map(str, span_tokens))
        return float(np.mean(np.asarray(span_probs, dtype=np.float64))), "".join(map(str, span_tokens))

    def _features(self, raw_sample: dict[str, Any]) -> dict[str, float]:
        sample = clean_sample(raw_sample)
        features: dict[str, float] = {}
        supporting_answers = [_text(x) for x in _as_list(sample.get("supporting_answers"))]
        supporting_tokens = _as_list(sample.get("supporting_tokens"))
        supporting_probs = _as_list(sample.get("supporting_probabilities"))

        # Token-probability statistics for each of the four alternate answers.
        for i in range(SUPPORT_COUNT):
            probs = supporting_probs[i] if i < len(supporting_probs) else []
            mtp, avg, mpd, gnll, var = self._prob_stats(probs)
            features[f"[stats] mtp_{i}"] = mtp
            features[f"[stats] avgtp_{i}"] = avg
            features[f"[stats] mpd_{i}"] = mpd
            features[f"[stats] g_nll_{i}"] = gnll
            features[f"[stats] var_prob_{i}"] = var

        # Semantic/text features: answer-span vocabulary diversity, TF-IDF, QA cosine.
        extracted = [span for answer in supporting_answers for span in _answer_spans(answer)]
        normalized_spans = [_clean_text(span) for span in extracted]
        words = " ".join(normalized_spans).split()
        features["[sem] bow_unique_ratio"] = len(set(words)) / len(words) if words else 0.0
        if normalized_spans:
            span_matrix = self.vectorizer.transform(normalized_spans)
            mean_vector = np.asarray(span_matrix.mean(axis=0)).ravel()
            features["[sem] tfidf_avg"] = float(mean_vector.mean())
        else:
            features["[sem] tfidf_avg"] = 0.0
        question = _clean_text(_text(sample.get("question")))
        answer = _clean_text(_text(sample.get("answer")))
        qa_matrix = self.vectorizer.transform([question, answer])
        features["[sem] qa_cosine_similarity"] = float(
            cosine_similarity(qa_matrix[0:1], qa_matrix[1:2])[0, 0]
        )

        # Agreement across extracted answers (exact text, as in the notebook).
        answer_counts = Counter(extracted)
        common_count = answer_counts.most_common(1)[0][1] if answer_counts else 0
        features["[cross] num_unique_answers"] = float(len(answer_counts))
        features["[cross] most_common_answer_count"] = float(common_count)
        features["[cross] agreement_ratio"] = (
            common_count / len(supporting_answers) if supporting_answers else 0.0
        )

        # Style and structure, including the notebook's longest matching block.
        def sentences(text: str) -> list[str]:
            return [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]

        answer_sentences = sentences(_text(sample.get("answer")))
        if not answer_sentences:
            answer_sentences = [_text(sample.get("answer"))]
        all_sentences = answer_sentences + [
            sentence for other in supporting_answers for sentence in sentences(other)
        ]
        lengths = [len(sentence.split()) for sentence in all_sentences]
        features["[style] sentence_length_variance"] = float(np.var(lengths)) if lengths else 0.0
        features["[style] average_sentence_length"] = float(np.mean(lengths)) if lengths else 0.0
        lcs = [
            SequenceMatcher(None, _text(sample.get("answer")), other).find_longest_match(
                0, len(_text(sample.get("answer"))), 0, len(other)
            ).size
            for other in supporting_answers
        ]
        features["[style] average_lcs"] = float(np.mean(lcs)) if lcs else 0.0

        # Question type one-hot flags.
        q_lower = question.strip()
        for word in ("who", "what", "where", "when", "why", "how", "which"):
            features[f"[type]  is_{word}"] = float(bool(re.match(rf"^{word}\b", q_lower)))

        # Mean answer-span probabilities, categorized by substring in the main answer.
        answer_probabilities: list[float] = []
        other_probabilities: list[float] = []
        for i, support_answer in enumerate(supporting_answers[:SUPPORT_COUNT]):
            token_list = supporting_tokens[i] if i < len(supporting_tokens) else []
            prob_list = supporting_probs[i] if i < len(supporting_probs) else []
            mean_probability, span_text = self._token_answer_probability(token_list, prob_list)
            if span_text and span_text in _text(sample.get("answer")):
                answer_probabilities.append(mean_probability)
            else:
                other_probabilities.append(mean_probability)

        for name, values in (("answer", answer_probabilities), ("other", other_probabilities)):
            features[f"[ansprob] {name}_min"] = float(np.min(values)) if values else 0.0
            features[f"[ansprob] {name}_mean"] = float(np.mean(values)) if values else 0.0
            features[f"[ansprob] {name}_max"] = float(np.max(values)) if values else 0.0
            features[f"[ansprob] {name}_std"] = float(np.std(values)) if values else 0.0
            features[f"[ansprob] {name}_len"] = float(len(values))

        for i in range(SUPPORT_COUNT):
            values = np.asarray(
                supporting_probs[i] if i < len(supporting_probs) else [], dtype=np.float64
            )
            features[f"[ansprob] supporting_proba_{i}_min"] = float(values.min()) if values.size else 0.0
            features[f"[ansprob] supporting_proba_{i}_mean"] = float(values.mean()) if values.size else 0.0
            features[f"[ansprob] supporting_proba_{i}_max"] = float(values.max()) if values.size else 0.0
            features[f"[ansprob] supporting_proba_{i}_std"] = float(values.std()) if values.size else 0.0
            features[f"[ansprob] supporting_proba_{i}_len"] = float(values.size)

        return features

    def transform(self, records: list[dict[str, Any]]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("fit the feature pipeline on training records first")
        rows = [self._features(record) for record in records]
        if not rows:
            return np.empty((0, len(self.feature_names_ or [])), dtype=np.float32)
        if self.feature_names_ is None:
            self.feature_names_ = list(rows[0])
        expected = self.feature_names_
        if any(list(row) != expected for row in rows):
            raise RuntimeError("feature schema changed between records")
        return np.asarray([[row[name] for name in expected] for row in rows], dtype=np.float32)

    def fit_transform(self, train_records: list[dict[str, Any]]) -> np.ndarray:
        self.fit(train_records)
        matrix = self.transform(train_records)
        self.feature_names_ = list(self._features(train_records[0])) if train_records else []
        return matrix


class HallucinationDetector:
    """XGBoost wrapper that predicts ``P(is_correct=True)``."""

    def __init__(self, n_estimators: int = 500):
        self.features = HallucinationFeatures()
        self.n_estimators = int(n_estimators)
        self.model = None

    def fit(self, train_records: list[dict[str, Any]]) -> "HallucinationDetector":
        if not train_records:
            raise ValueError("training data is empty")
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "Install the task-permitted xgboost package to fit the classifier"
            ) from exc

        x_train = self.features.fit_transform(train_records)
        y_train = np.asarray([int(bool(row.get("is_correct", False))) for row in train_records])
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=2,
            subsample=0.5,
            colsample_bytree=0.8,
            gamma=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        # Validation data is deliberately not passed to fit or early stopping.
        self.model.fit(x_train, y_train)
        return self

    def predict_correct_probability(self, records: list[dict[str, Any]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit(train_records) before prediction")
        x = self.features.transform(records)
        class_one_column = list(self.model.classes_).index(1)
        return self.model.predict_proba(x)[:, class_one_column]

    def predict_hallucinations(self, sample: dict[str, Any]) -> float:
        """Competition-compatible scalar; despite the name, returns P(correct)."""
        return float(self.predict_correct_probability([sample])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_json", help="released training JSON")
    parser.add_argument("valid_json", help="released validation JSON; used only for scoring")
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()

    train_records = load_records(args.train_json)
    valid_records = load_records(args.valid_json)
    detector = HallucinationDetector(n_estimators=args.n_estimators).fit(train_records)
    probabilities = detector.predict_correct_probability(valid_records)
    targets = np.asarray([int(bool(row.get("is_correct", False))) for row in valid_records])
    auc = roc_auc_score(targets, probabilities)
    print(f"train records: {len(train_records)}")
    print(f"validation records: {len(valid_records)}")
    print(f"validation ROC AUC: {auc:.4f}")
    score = 0 if auc <= 0.70 else (100 if auc >= 0.82 else round(100 * (auc - 0.70) / 0.12))
    print(f"estimated score: {score}/100")


if __name__ == "__main__":
    main()
