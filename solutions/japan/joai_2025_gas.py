#!/usr/bin/env python3
"""Multimodal JOAI 2025 gas-classification baseline and submission builder.

Expected input layout (download the competition data separately):
    data/
      train.csv
      test.csv
      sample_submission.csv  # optional
      images/*.png

The script uses only competition-provided data and scikit-learn/Pillow. It does
not download or redistribute the data or any pretrained weights.

Examples:
    python joai_2025_gas.py --data-dir /kaggle/input/playground-joai-competition-2025 \
        --mode validate --folds 5
    python joai_2025_gas.py --data-dir /kaggle/input/playground-joai-competition-2025 \
        --mode submit --output submission.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


LABELS = ["Mixture", "NoGas", "Perfume", "Smoke"]
IMAGE_GRID = 4
IMAGE_FEATURES = 3 * (8 + IMAGE_GRID * IMAGE_GRID) + 16 + 1

TEMP_RANGE_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?[-–]\s*"
    r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?([CF])\b",
    flags=re.IGNORECASE,
)
TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?([CF])\b", flags=re.IGNORECASE)
COORD_RE = re.compile(r"\b\d{1,3}\s*°\s*\d{1,2}", flags=re.IGNORECASE)
COLOR_WORDS = (
    "red", "orange", "yellow", "green", "blue", "purple", "white", "black",
    "pink", "cyan", "magenta", "brown", "gray", "grey",
)
SCENE_WORDS = (
    "hot", "warm", "cool", "cold", "uniform", "plume", "leak", "rising",
    "gas", "smoke", "perfume", "flame", "hand", "person", "door", "wall",
    "indoor", "outdoor", "obscured", "unavailable", "temperature",
)


def normalize_label(value: object) -> str:
    label = str(value).strip()
    if label == "Mixed":
        return "Mixture"
    return label


def temperature_values(caption: str) -> tuple[list[float], bool, bool]:
    """Extract Celsius-normalized temperature values, including compact ranges."""
    values: list[float] = []
    has_c = False
    has_f = False
    range_spans: list[tuple[int, int]] = []
    for match in TEMP_RANGE_RE.finditer(caption):
        lo, hi = float(match.group(1)), float(match.group(2))
        unit = match.group(3).upper()
        has_c |= unit == "C"
        has_f |= unit == "F"
        if unit == "F":
            lo = (lo - 32.0) * (5.0 / 9.0)
            hi = (hi - 32.0) * (5.0 / 9.0)
        values.extend((lo, hi))
        range_spans.append(match.span())

    for match in TEMP_RE.finditer(caption):
        if any(start <= match.start() and match.end() <= end for start, end in range_spans):
            continue
        value = float(match.group(1))
        unit = match.group(2).upper()
        has_c |= unit == "C"
        has_f |= unit == "F"
        if unit == "F":
            value = (value - 32.0) * (5.0 / 9.0)
        values.append(value)
    return values, has_c, has_f


def caption_numeric_features(caption_value: object) -> list[float]:
    caption = str(caption_value or "")
    lower = caption.lower()
    temps, has_c, has_f = temperature_values(caption)
    if temps:
        temp_stats = [min(temps), max(temps), float(np.mean(temps)), float(np.std(temps))]
    else:
        temp_stats = [0.0, 0.0, 0.0, 0.0]

    words = re.findall(r"[a-z]+", lower)
    feats = [
        *temp_stats,
        float(len(temps)),
        float(has_c),
        float(has_f),
        float(bool(COORD_RE.search(caption))),
        float(len(caption)),
        float(len(words)),
        float(sum(ch.isdigit() for ch in caption)),
    ]
    feats.extend(float(color in lower) for color in COLOR_WORDS)
    feats.extend(float(word in lower) for word in SCENE_WORDS)
    return feats


def sensor_features(frame: pd.DataFrame) -> np.ndarray:
    mq5 = pd.to_numeric(frame.get("MQ5", pd.Series(np.zeros(len(frame)))), errors="coerce").to_numpy(float)
    mq8 = pd.to_numeric(frame.get("MQ8", pd.Series(np.zeros(len(frame)))), errors="coerce").to_numpy(float)
    safe5 = np.maximum(mq5, 0.0)
    safe8 = np.maximum(mq8, 0.0)
    return np.column_stack(
        [
            mq5,
            mq8,
            np.log1p(safe5),
            np.log1p(safe8),
            mq5 + mq8,
            mq5 - mq8,
            mq5 * mq8,
            mq5 / (mq8 + 1.0),
            mq8 / (mq5 + 1.0),
            np.log1p(safe5) - np.log1p(safe8),
        ]
    )


def image_features(path: Path | None) -> np.ndarray:
    if path is None or not path.is_file():
        return np.r_[np.zeros(IMAGE_FEATURES - 1, dtype=np.float32), 1.0]

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as image:
            rgb = np.asarray(
                image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
    except Exception:
        return np.r_[np.zeros(IMAGE_FEATURES - 1, dtype=np.float32), 1.0]

    feats: list[float] = []
    for channel in range(3):
        plane = rgb[:, :, channel]
        feats.extend(np.quantile(plane, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).tolist())
        feats.append(float(np.std(plane)))
        pooled = plane.reshape(IMAGE_GRID, 32 // IMAGE_GRID, IMAGE_GRID, 32 // IMAGE_GRID).mean(axis=(1, 3))
        feats.extend(pooled.ravel().tolist())
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=False)
    feats.extend((hist / max(1, hist.sum())).tolist())
    feats.append(0.0)
    return np.asarray(feats, dtype=np.float32)


def build_image_lookup(data_dir: Path) -> dict[str, Path]:
    image_dir = data_dir / "images"
    if not image_dir.exists():
        image_dir = data_dir
    lookup: dict[str, Path] = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            lookup.setdefault(path.name.lower(), path)
    return lookup


def extract_image_matrix(frame: pd.DataFrame, lookup: dict[str, Path]) -> tuple[np.ndarray, int]:
    if "image_path_uuid" not in frame.columns:
        raise ValueError("CSV is missing the required image_path_uuid column")
    cache: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    missing = 0
    for raw_name in frame["image_path_uuid"].fillna("").astype(str):
        name = Path(raw_name).name
        key = name.lower()
        if key not in cache:
            path = lookup.get(key)
            cache[key] = image_features(path)
            missing += int(path is None)
        rows.append(cache[key])
    return np.vstack(rows), missing


def dense_features(frame: pd.DataFrame, image_matrix: np.ndarray) -> np.ndarray:
    captions = frame.get("Caption", pd.Series([""] * len(frame))).fillna("").astype(str)
    caption = np.asarray([caption_numeric_features(value) for value in captions], dtype=np.float32)
    sensors = sensor_features(frame).astype(np.float32)
    dense = np.column_stack((sensors, caption, image_matrix)).astype(np.float32)
    return np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)


def make_text_matrices(
    train_text: Iterable[str],
    other_text: Iterable[str],
    train_dense: np.ndarray,
    other_dense: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    train_text = [str(x) for x in train_text]
    other_text = [str(x) for x in other_text]
    blocks_train: list[sparse.csr_matrix] = []
    blocks_other: list[sparse.csr_matrix] = []
    specs = [
        dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=12000, token_pattern=r"(?u)\b\w+\b"),
        dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=16000),
    ]
    for spec in specs:
        vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", **spec)
        try:
            blocks_train.append(vectorizer.fit_transform(train_text).tocsr())
            blocks_other.append(vectorizer.transform(other_text).tocsr())
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).lower():
                raise

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_dense).astype(np.float32)
    scaled_other = scaler.transform(other_dense).astype(np.float32)
    blocks_train.append(sparse.csr_matrix(scaled_train))
    blocks_other.append(sparse.csr_matrix(scaled_other))
    return sparse.hstack(blocks_train, format="csr"), sparse.hstack(blocks_other, format="csr")


def fit_models(x_dense: np.ndarray, x_sparse: sparse.csr_matrix) -> tuple[ExtraTreesClassifier, LogisticRegression]:
    tree = ExtraTreesClassifier(
        n_estimators=500,
        max_features=0.9,
        min_samples_leaf=1,
        class_weight=None,
        n_jobs=-1,
        random_state=42,
    )
    linear = LogisticRegression(C=4.0, max_iter=1800, solver="lbfgs", random_state=42)
    return tree, linear


def aligned_proba(model: object, x: object, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)  # type: ignore[attr-defined]
    aligned = np.zeros((len(raw), len(LABELS)), dtype=np.float64)
    for source_index, label in enumerate(model.classes_):  # type: ignore[attr-defined]
        aligned[:, LABELS.index(str(label))] = raw[:, source_index]
    return aligned


def validate(train: pd.DataFrame, dense: np.ndarray, folds: int, seed: int) -> None:
    y = train["Gas"].map(normalize_label).to_numpy()
    oof_tree = np.zeros((len(y), len(LABELS)), dtype=np.float64)
    oof_linear = np.zeros_like(oof_tree)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    texts = train["Caption"].fillna("").astype(str).to_numpy()
    start = time.time()
    for fold, (tr_idx, va_idx) in enumerate(cv.split(dense, y), 1):
        x_tr_sparse, x_va_sparse = make_text_matrices(texts[tr_idx], texts[va_idx], dense[tr_idx], dense[va_idx])
        tree, linear = fit_models(dense[tr_idx], x_tr_sparse)
        tree.fit(dense[tr_idx], y[tr_idx])
        linear.fit(x_tr_sparse, y[tr_idx])
        oof_tree[va_idx] = aligned_proba(tree, dense[va_idx], np.asarray(LABELS))
        oof_linear[va_idx] = aligned_proba(linear, x_va_sparse, np.asarray(LABELS))
        p_tree = np.asarray(LABELS)[oof_tree[va_idx].argmax(axis=1)]
        p_linear = np.asarray(LABELS)[oof_linear[va_idx].argmax(axis=1)]
        print(
            f"fold={fold}/{folds}  tree_weighted_f1={f1_score(y[va_idx], p_tree, average='weighted'):.6f}"
            f"  text+numeric_weighted_f1={f1_score(y[va_idx], p_linear, average='weighted'):.6f}"
        )

    label_array = np.asarray(LABELS)
    print(f"\nCV weighted F1, ExtraTrees dense: {f1_score(y, label_array[np.argmax(oof_tree, axis=1)], average='weighted'):.6f}")
    print(f"CV weighted F1, TF-IDF + numeric logistic: {f1_score(y, label_array[np.argmax(oof_linear, axis=1)], average='weighted'):.6f}")
    blended = 0.5 * oof_tree + 0.5 * oof_linear
    blend_pred = np.asarray(LABELS)[np.argmax(blended, axis=1)]
    print(f"CV weighted F1, 50:50 probability blend: {f1_score(y, blend_pred, average='weighted'):.6f}")
    print("\nBlended classification report:")
    print(classification_report(y, blend_pred, labels=LABELS, zero_division=0, digits=4))
    print("Blended confusion matrix (rows=true, columns=predicted; label order:", LABELS, ")")
    print(confusion_matrix(y, blend_pred, labels=LABELS))
    print(f"Elapsed seconds (feature matrix already extracted): {time.time() - start:.1f}")


def train_submission(train: pd.DataFrame, test: pd.DataFrame, dense_train: np.ndarray, dense_test: np.ndarray, output: Path, data_dir: Path) -> None:
    y = train["Gas"].map(normalize_label).to_numpy()
    train_text = train["Caption"].fillna("").astype(str).to_numpy()
    test_text = test["Caption"].fillna("").astype(str).to_numpy()
    x_train_sparse, x_test_sparse = make_text_matrices(train_text, test_text, dense_train, dense_test)
    tree, linear = fit_models(dense_train, x_train_sparse)
    tree.fit(dense_train, y)
    linear.fit(x_train_sparse, y)
    probabilities = 0.5 * aligned_proba(tree, dense_test, np.asarray(LABELS))
    probabilities += 0.5 * aligned_proba(linear, x_test_sparse, np.asarray(LABELS))
    prediction = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]

    sample_path = data_dir / "sample_submission.csv"
    if sample_path.is_file():
        submission = pd.read_csv(sample_path)
        if len(submission) != len(test):
            raise ValueError("sample_submission.csv row count differs from test.csv")
        submission["Gas"] = prediction
    else:
        submission = pd.DataFrame({"index": np.arange(len(test)), "Gas": prediction})
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Wrote {len(submission)} predictions to {output}")


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv in {data_dir}. Download competition data separately; "
            "this script does not fetch licensed task files."
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required = {"MQ5", "MQ8", "Caption", "image_path_uuid"}
    missing = sorted(required - set(train.columns))
    if missing or "Gas" not in train.columns:
        raise ValueError(f"train.csv is missing required columns: {missing + ([] if 'Gas' in train else ['Gas'])}")
    missing_test = sorted(required - set(test.columns))
    if missing_test:
        raise ValueError(f"test.csv is missing required columns: {missing_test}")
    unknown = sorted(set(train["Gas"].map(normalize_label)) - set(LABELS))
    if unknown:
        raise ValueError(f"Unexpected label values: {unknown}")
    return train, test


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/input/playground-joai-competition-2025"))
    parser.add_argument("--mode", choices=("validate", "submit"), default="validate")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    train, test = load_data(args.data_dir)
    print(f"train={len(train):,} test={len(test):,}")
    print("class counts:", train["Gas"].map(normalize_label).value_counts().to_dict())

    t0 = time.time()
    lookup = build_image_lookup(args.data_dir)
    image_train, missing_train = extract_image_matrix(train, lookup)
    dense_train = dense_features(train, image_train)
    print(f"train images missing/unreadable: {missing_train}/{len(train)}")

    if args.mode == "validate":
        validate(train, dense_train, args.folds, args.seed)
    else:
        image_test, missing_test = extract_image_matrix(test, lookup)
        dense_test = dense_features(test, image_test)
        print(f"test images missing/unreadable: {missing_test}/{len(test)}")
        train_submission(train, test, dense_train, dense_test, args.output, args.data_dir)
    print(f"Total elapsed seconds: {time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
