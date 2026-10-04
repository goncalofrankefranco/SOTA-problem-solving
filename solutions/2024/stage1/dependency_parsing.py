"""HerBERT-based distance/depth parser for Polish AI Olympiad 2024, Stage I.

This standalone implementation needs ``torch``, ``transformers`` and NumPy,
plus the organizer's ``train.conll`` / ``valid.conll`` files. It only trains the
small task heads on ``train.conll``; HerBERT is frozen. The official HerBERT
checkpoint is downloaded by ``from_pretrained`` when it is not already cached.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


HERBERT_MODEL = "allegro/herbert-base-cased"
DISTANCE_MODEL_PATH = "distance_model.pth"
DEPTH_MODEL_PATH = "depth_model.pth"


class Sentence:
    def __init__(self, words: Sequence[str]):
        self.words = list(words)

    def __len__(self) -> int:
        return len(self.words)

    def __repr__(self) -> str:
        return " ".join(self.words)


class ParsedSentence(Sentence):
    """Organizer-compatible sentence representation with 1-based heads."""

    def __init__(self, words: Sequence[str], heads: Sequence[int]):
        if len(words) != len(heads):
            raise ValueError("Different number of words and heads")
        if not all(0 <= head <= len(words) for head in heads):
            raise ValueError("Invalid head index")
        if list(heads).count(0) != 1:
            raise ValueError("A dependency tree must have exactly one root")
        super().__init__(words)
        self.heads = list(map(int, heads))
        self.root = self.heads.index(0)

    @classmethod
    def from_edges_and_root(
        cls, words: Sequence[str], edges: Iterable[tuple[int, int]], root: int
    ) -> "ParsedSentence":
        n = len(words)
        if not 0 <= root < n:
            raise ValueError("Root index is outside the sentence")
        adjacency = [[] for _ in range(n)]
        for i, j in edges:
            if i == j or not (0 <= i < n and 0 <= j < n):
                raise ValueError("Invalid tree edge")
            adjacency[i].append(j)
            adjacency[j].append(i)
        heads = [-1] * n
        heads[root] = 0
        stack = [(root, -1)]
        while stack:
            node, parent = stack.pop()
            for child in adjacency[node]:
                if child == parent:
                    continue
                if heads[child] != -1:
                    raise ValueError("Edges contain a cycle")
                heads[child] = node + 1
                stack.append((child, node))
        if any(head == -1 for head in heads):
            raise ValueError("Edges are not connected")
        return cls(words, heads)

    def get_sorted_edges(self) -> list[tuple[int, int]]:
        return [
            (min(i, head - 1), max(i, head - 1))
            for i, head in enumerate(self.heads)
            if head != 0
        ]


def read_conll(path: str | Path) -> list[ParsedSentence]:
    sentences: list[ParsedSentence] = []
    words: list[str] = []
    heads: list[int] = []

    def flush() -> None:
        nonlocal words, heads
        if words:
            sentences.append(ParsedSentence(words, heads))
            words, heads = [], []

    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                flush()
                continue
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 7:
                continue
            try:
                int(fields[0])
                head = int(fields[6])
            except ValueError:
                # Skip CoNLL range/empty-node rows, which are not tokens.
                continue
            words.append(fields[1])
            heads.append(head)
    flush()
    return sentences


def uuas_score(golden: ParsedSentence, predicted: ParsedSentence) -> float:
    gold_edges = set(golden.get_sorted_edges())
    pred_edges = set(predicted.get_sorted_edges())
    return len(gold_edges & pred_edges) / max(1, len(gold_edges))


def get_distances(sentence: ParsedSentence) -> np.ndarray:
    """Return the shortest-path distance between every pair of tree words."""
    n = len(sentence)
    adjacency = [[] for _ in range(n)]
    for child, head in enumerate(sentence.heads):
        if head:
            parent = head - 1
            adjacency[child].append(parent)
            adjacency[parent].append(child)
    distances = np.zeros((n, n), dtype=np.float32)
    for source in range(n):
        distances[source, :] = np.inf
        distances[source, source] = 0.0
        queue = [source]
        for node in queue:
            for child in adjacency[node]:
                if not np.isfinite(distances[source, child]):
                    distances[source, child] = distances[source, node] + 1.0
                    queue.append(child)
    return distances


def get_word_embeddings(
    sentences: Sequence[Sentence], tokenizer, model: nn.Module, batch_size: int = 24
) -> list[torch.Tensor]:
    """Extract frozen contextual HerBERT vectors and average word subwords."""
    model.eval()
    device = next(model.parameters()).device
    results: list[torch.Tensor] = []
    texts = [" ".join(sentence.words) for sentence in sentences]
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch_sentences = sentences[start : start + batch_size]
            encoded = tokenizer(
                texts[start : start + batch_size],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            for row, sentence in enumerate(batch_sentences):
                length = int(encoded["attention_mask"][row].sum().item())
                sentence_ids = encoded["input_ids"][row, :length].tolist()
                word_ids: list[list[int]] = [
                    tokenizer(word, add_special_tokens=False)["input_ids"]
                    for word in sentence.words
                ]
                joined_ids = [token_id for group in word_ids for token_id in group]
                if joined_ids != sentence_ids:
                    raise ValueError(
                        "HerBERT word tokenization did not match sentence tokenization; "
                        f"sentence={sentence!r}"
                    )
                vectors = []
                cursor = 0
                for ids in word_ids:
                    end = cursor + len(ids)
                    if end == cursor:
                        vectors.append(hidden.new_zeros(hidden.shape[-1]))
                    else:
                        vectors.append(hidden[row, cursor:end].mean(dim=0))
                    cursor = end
                results.append(torch.stack(vectors).cpu())
    return results


class DistanceModel(nn.Module):
    """Symmetric regression head for pairwise tree distances."""

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.GELU())
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        left = projected.unsqueeze(2)
        right = projected.unsqueeze(1)
        pair = torch.cat((torch.abs(left - right), left * right), dim=-1)
        distances = F.softplus(self.pair_head(pair).squeeze(-1))
        length = distances.shape[1]
        return distances.masked_fill(torch.eye(length, device=x.device, dtype=torch.bool)[None], 0.0)


class DepthModel(nn.Module):
    """Predict each word's distance from the syntactic root."""

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.network(x))


def _collate(examples, target_kind: str):
    embeddings, targets = zip(*examples)
    batch = len(examples)
    max_len = max(len(item) for item in embeddings)
    dim = embeddings[0].shape[-1]
    x = torch.zeros(batch, max_len, dim, dtype=torch.float32)
    shape = (batch, max_len, max_len) if target_kind == "distance" else (batch, max_len, 1)
    y = torch.full(shape, float("inf"), dtype=torch.float32)
    for i, (embedding, target) in enumerate(zip(embeddings, targets)):
        length = len(embedding)
        x[i, :length] = embedding
        if target_kind == "distance":
            y[i, :length, :length] = torch.as_tensor(target, dtype=torch.float32)
        else:
            y[i, :length, :] = torch.as_tensor(target, dtype=torch.float32)
    mask = torch.isfinite(y)
    return x, y, mask


def _fit_regressor(
    model: nn.Module,
    embeddings: Sequence[torch.Tensor],
    targets: Sequence[np.ndarray],
    target_kind: str,
    epochs: int,
    learning_rate: float,
    batch_size: int = 32,
) -> nn.Module:
    device = next(model.parameters()).device
    order = np.arange(len(embeddings))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        np.random.shuffle(order)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            x, y, mask = _collate(
                [(embeddings[i], targets[i]) for i in ids], target_kind
            )
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = ((prediction[mask] - y[mask]) ** 2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{target_kind} epoch {epoch + 1}/{epochs}: loss={epoch_loss / batches:.4f}")
    return model.eval()


def train_solution(
    train_path: str | Path,
    output_dir: str | Path = ".",
    model_name: str = HERBERT_MODEL,
    epochs: int = 40,
) -> tuple[DistanceModel, DepthModel, object, nn.Module]:
    """Train the task heads using only the organizer's labeled train set."""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name).to(device).eval()
    sentences = read_conll(train_path)
    embeddings = get_word_embeddings(sentences, tokenizer, bert)
    distance_targets = [get_distances(sentence) for sentence in sentences]
    depth_targets = [distance[sentence.root, :, None] for sentence, distance in zip(sentences, distance_targets)]

    distance_model = DistanceModel(bert.config.hidden_size).to(device)
    depth_model = DepthModel(bert.config.hidden_size).to(device)
    _fit_regressor(distance_model, embeddings, distance_targets, "distance", epochs, 1e-3)
    _fit_regressor(depth_model, embeddings, depth_targets, "depth", epochs, 1e-3)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(distance_model.cpu().state_dict(), output_dir / DISTANCE_MODEL_PATH)
    torch.save(depth_model.cpu().state_dict(), output_dir / DEPTH_MODEL_PATH)
    return distance_model, depth_model, tokenizer, bert.cpu()


def _minimum_spanning_tree(distances: np.ndarray) -> list[tuple[int, int]]:
    n = distances.shape[0]
    if n <= 1:
        return []
    weights = 0.5 * (distances + distances.T)
    np.fill_diagonal(weights, np.inf)
    selected = {0}
    edges: list[tuple[int, int]] = []
    while len(selected) < n:
        best = (float("inf"), -1, -1)
        for i in selected:
            candidates = np.flatnonzero(~np.isin(np.arange(n), list(selected)))
            if len(candidates) == 0:
                break
            j = int(candidates[np.argmin(weights[i, candidates])])
            candidate = (float(weights[i, j]), i, j)
            if candidate[0] < best[0]:
                best = candidate
        if best[1] < 0 or not np.isfinite(best[0]):
            raise ValueError("Could not construct a spanning tree")
        _, i, j = best
        edges.append((i, j))
        selected.add(j)
    return edges


def parse_sentence(
    sentence: Sentence,
    distance_model: DistanceModel,
    depth_model: DepthModel,
    tokenizer,
    bert: nn.Module,
) -> ParsedSentence:
    embeddings = get_word_embeddings([sentence], tokenizer, bert)[0]
    device = next(distance_model.parameters()).device
    x = embeddings.unsqueeze(0).to(device)
    with torch.no_grad():
        pair_distances = distance_model(x)[0].cpu().numpy()
        depths = depth_model(x)[0, :, 0].cpu().numpy()
    root = int(np.argmin(depths))
    edges = _minimum_spanning_tree(pair_distances)
    return ParsedSentence.from_edges_and_root(sentence.words, edges, root)


def points(root_placement: float, uuas: float) -> float:
    def scale(value: float) -> float:
        return (min(max(value, 0.5), 0.85) - 0.5) / 0.35

    return scale(root_placement) + scale(uuas)


def evaluate(
    sentences: Sequence[ParsedSentence],
    distance_model: DistanceModel,
    depth_model: DepthModel,
    tokenizer,
    bert: nn.Module,
) -> tuple[float, float, float]:
    root_correct = 0
    uuas_total = 0.0
    for sentence in sentences:
        predicted = parse_sentence(sentence, distance_model, depth_model, tokenizer, bert)
        if len(predicted.get_sorted_edges()) != len(sentence) - 1:
            raise ValueError("Parser did not return a tree")
        root_correct += int(predicted.root == sentence.root)
        uuas_total += uuas_score(sentence, predicted)
    root_accuracy = root_correct / len(sentences)
    uuas = uuas_total / len(sentences)
    score = points(root_accuracy, uuas)
    print(f"Root placement: {root_accuracy:.4f}; UUAS: {uuas:.4f}; score: {score:.4f}/2")
    return root_accuracy, uuas, score


def _load_head(model_class, path: Path, hidden_size: int, device):
    model = model_class(hidden_size).to(device)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--model-name", default=HERBERT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train", action="store_true", help="Fit heads on train.conll first")
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert = AutoModel.from_pretrained(args.model_name).to(device).eval()
    if args.train:
        train_solution(
            args.data_dir / "train.conll", args.output_dir, args.model_name, args.epochs
        )
    distance_model = _load_head(
        DistanceModel, args.output_dir / DISTANCE_MODEL_PATH, bert.config.hidden_size, device
    )
    depth_model = _load_head(
        DepthModel, args.output_dir / DEPTH_MODEL_PATH, bert.config.hidden_size, device
    )
    evaluate(read_conll(args.data_dir / "valid.conll"), distance_model, depth_model, tokenizer, bert)


if __name__ == "__main__":
    main()
