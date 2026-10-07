"""CPU-friendly PyTorch text classifier for NOAI China 2024 task 3.

Word and character TF-IDF are used as sparse input features; a trainable
multiclass linear model is optimized in PyTorch. Run ``--mode validate`` for
duplicate-aware stratified cross-validation, or ``--mode submit`` with both
official CSVs available to write predictions in the requested format.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn


class SparseLinearClassifier(nn.Module):
    """A multiclass linear model whose sparse input multiplication uses PyTorch."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(x, self.weight) + self.bias


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_training(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path)
    required = {"text", "category"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    texts = frame["text"].fillna("").astype(str).tolist()
    labels = frame["category"].astype(str).to_numpy()
    return texts, labels


def make_features(
    train_texts: list[str], other_texts: list[str] | None = None
) -> tuple[sp.csr_matrix, sp.csr_matrix | None]:
    """Fit compact word/character TF-IDF on training text only."""
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = sp.hstack(
        [word.fit_transform(train_texts), char.fit_transform(train_texts)],
        format="csr",
        dtype=np.float32,
    )
    if other_texts is None:
        return x_train, None
    x_other = sp.hstack(
        [word.transform(other_texts), char.transform(other_texts)],
        format="csr",
        dtype=np.float32,
    )
    return x_train, x_other


def to_torch_sparse(matrix: sp.csr_matrix) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def fit_predict(
    x_train: sp.csr_matrix,
    y_train: np.ndarray,
    x_other: sp.csr_matrix,
    *,
    seed: int,
    epochs: int,
    learning_rate: float = 0.08,
    weight_decay: float = 0.02,
) -> tuple[np.ndarray, SparseLinearClassifier]:
    seed_everything(seed)
    model = SparseLinearClassifier(x_train.shape[1], int(y_train.max()) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    x_tensor = to_torch_sparse(x_train)
    x_other_tensor = to_torch_sparse(x_other)
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        predictions = model(x_other_tensor).argmax(dim=1).cpu().numpy()
    return predictions, model


def validate(path: Path, folds: int, seed: int, epochs: int) -> None:
    texts, labels = load_training(path)
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    groups = np.asarray([hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.empty_like(y)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(texts, y, groups), start=1):
        x_train, x_valid = make_features(
            [texts[i] for i in train_idx], [texts[i] for i in valid_idx]
        )
        assert x_valid is not None
        # Encode fold labels against the global sorted class list.
        prediction, _ = fit_predict(
            x_train,
            y[train_idx],
            x_valid,
            seed=seed + fold,
            epochs=epochs,
        )
        oof[valid_idx] = prediction
        print(
            f"fold={fold} macro_f1="
            f"{f1_score(y[valid_idx], prediction, average='macro', zero_division=0):.4f} "
            f"features={x_train.shape[1]} nnz={x_train.nnz}"
        )
    score = f1_score(y, oof, average="macro", zero_division=0)
    print(f"oof_macro_f1={score:.4f} rows={len(y)} classes={list(encoder.classes_)}")


def submit(train_path: Path, test_path: Path, output: Path, seed: int, epochs: int) -> None:
    texts, labels = load_training(train_path)
    test_frame = pd.read_csv(test_path)
    if "text" not in test_frame:
        raise ValueError(f"Missing text column in {test_path}")
    test_texts = test_frame["text"].fillna("").astype(str).tolist()
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    x_train, x_test = make_features(texts, test_texts)
    assert x_test is not None
    predictions, _ = fit_predict(x_train, y, x_test, seed=seed, epochs=epochs)
    test_frame["category"] = encoder.inverse_transform(predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    test_frame.to_csv(output, index=False)
    print(f"saved {output} rows={len(test_frame)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["validate", "submit"], default="validate")
    parser.add_argument("--train", type=Path, required=True, help="CSV with text and category columns")
    parser.add_argument("--test", type=Path, help="Unlabeled CSV with a text column")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=160)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "validate":
        validate(args.train, args.folds, args.seed, args.epochs)
    else:
        if args.test is None:
            parser.error("--test is required in submit mode")
        submit(args.train, args.test, args.output, args.seed, args.epochs)


if __name__ == "__main__":
    main()
