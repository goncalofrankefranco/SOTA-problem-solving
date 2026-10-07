"""Character-level German compound segmentation with a lexicon-aware BiLSTM.

Expected data: the organizer's train.json plus optional unlabeled val.json and
test.json. The script never downloads or writes copies of those assets.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD, UNK = 0, 1


def load_json(path: Path) -> dict[str, list[int]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {str(w): list(map(int, y)) for w, y in data.items()}


def make_vocab(words: Iterable[str]) -> dict[str, int]:
    chars = sorted({ch for word in words for ch in word})
    return {ch: i + 2 for i, ch in enumerate(chars)}


class WordDataset(Dataset):
    def __init__(self, data: dict[str, list[int]], vocab: dict[str, int]):
        self.items = [(w, y) for w, y in data.items()]
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        word, labels = self.items[i]
        x = torch.tensor([self.vocab.get(ch, UNK) for ch in word], dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.float32)
        return word, x, y


def collate(batch):
    words, xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    return words, pad_sequence(xs, batch_first=True, padding_value=PAD), pad_sequence(ys, batch_first=True, padding_value=-1), lengths


class Segmenter(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 48, hidden: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.encoder = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True, num_layers=2, dropout=0.15)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.embedding(x))[0]).squeeze(-1)


def constituents(data: dict[str, list[int]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for word, labels in data.items():
        start = 0
        for i, flag in enumerate(labels):
            if flag:
                counts[word[start:i + 1].lower()] += 1
                start = i + 1
    return counts


def decode_word(word: str, prob: np.ndarray, lex: Counter[str], threshold: float, lex_weight: float) -> list[int]:
    """Find a segmentation by dynamic programming over token boundaries."""
    n = len(word)
    if not n:
        return []
    p = np.clip(prob, 1e-5, 1 - 1e-5)
    # Shift the local log-odds so threshold has its usual meaning.
    offset = math.log(threshold / (1.0 - threshold))
    log0 = -np.logaddexp(0.0, np.log(p / (1.0 - p)) - offset)
    log1 = -np.logaddexp(0.0, -np.log(p / (1.0 - p)) + offset)
    prefix0 = np.concatenate(([0.0], np.cumsum(log0)))
    best = np.full(n + 1, -1e30, dtype=np.float64)
    prev = np.full(n + 1, -1, dtype=np.int32)
    best[0] = 0.0
    max_len = min(max((len(k) for k in lex), default=0), n)
    for end in range(1, n + 1):
        lower = max(0, end - max_len)
        for start in range(lower, end):
            token = word[start:end].lower()
            count = lex.get(token, 0)
            # Unknown pieces remain legal; known training constituents receive
            # a small frequency bonus. Every completed word ends in label 1.
            local = prefix0[end - 1] - prefix0[start] + log1[end - 1]
            if count:
                local += lex_weight * math.log1p(count)
            score = best[start] + local
            if score > best[end]:
                best[end], prev[end] = score, start
    result = [0] * n
    end = n
    while end > 0:
        start = int(prev[end])
        if start < 0:
            start = end - 1
        result[end - 1] = 1
        end = start
    return result


def predict_prob(model: Segmenter, data: dict[str, list[int]], vocab: dict[str, int], device: torch.device, batch_size: int = 512) -> dict[str, np.ndarray]:
    ds = WordDataset(data, vocab)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    out: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for words, x, _, lengths in loader:
            logits = torch.sigmoid(model(x.to(device))).cpu().numpy()
            for i, word in enumerate(words):
                out[word] = logits[i, :int(lengths[i])]
    return out


def train_model(data: dict[str, list[int]], vocab: dict[str, int], epochs: int, batch_size: int, seed: int, device: torch.device) -> Segmenter:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds = WordDataset(data, vocab)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = Segmenter(len(vocab) + 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(1.6, device=device))
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for _, x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            mask = y >= 0
            loss = (criterion(logits, y.clamp_min(0)) * mask).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            count += 1
        print(f"epoch={epoch + 1} loss={total / max(count, 1):.5f}")
    return model


def word_f1(gold: list[int], pred: list[int]) -> float:
    def spans(labels):
        result = set()
        start = 0
        for i, value in enumerate(labels):
            if value:
                result.add((start, i))
                start = i + 1
        if start < len(labels):
            result.add((start, len(labels) - 1))
        return result
    g, p = spans(gold), spans(pred)
    tp = len(g & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def mean_word_f1(gold: dict[str, list[int]], pred: dict[str, list[int]]) -> float:
    return float(np.mean([word_f1(y, pred[w]) for w, y in gold.items()]))


def predict_labels(model, data, vocab, lex, threshold, lex_weight, device):
    probs = predict_prob(model, data, vocab, device)
    return {word: decode_word(word, probs[word], lex, threshold, lex_weight) for word in data}


def write_submission(val_pred: dict[str, list[int]], test_pred: dict[str, list[int]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    val_path, test_path = out / "submissionval.json", out / "submissiontest.json"
    for path, data in ((val_path, val_pred), (test_path, test_pred)):
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(out / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(val_path, val_path.name)
        zf.write(test_path, test_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "submit"), default="validate")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path)
    parser.add_argument("--test-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("submission"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.42)
    parser.add_argument("--lex-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    data = load_json(args.train_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "validate":
        items = list(data.items())
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(items))
        n_val = max(1, int(0.1 * len(items)))
        val_items = [items[i] for i in order[:n_val]]
        train = dict(items[i] for i in order[n_val:])
        heldout = dict(val_items)
        vocab = make_vocab(train)
        model = train_model(train, vocab, args.epochs, args.batch_size, args.seed, device)
        lex = constituents(train)
        pred = predict_labels(model, heldout, vocab, lex, args.threshold, args.lex_weight, device)
        print(f"heldout_words={len(heldout)} mean_word_f1={mean_word_f1(heldout, pred):.6f} threshold={args.threshold} lex_weight={args.lex_weight}")
        return
    if args.val_json is None or args.test_json is None:
        raise SystemExit("submit mode requires --val-json and --test-json")
    val, test = load_json(args.val_json), load_json(args.test_json)
    vocab = make_vocab(data)
    model = train_model(data, vocab, args.epochs, args.batch_size, args.seed, device)
    lex = constituents(data)
    write_submission(predict_labels(model, val, vocab, lex, args.threshold, args.lex_weight, device),
                     predict_labels(model, test, vocab, lex, args.threshold, args.lex_weight, device), args.output_dir)
    print(f"wrote {args.output_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
