"""Train-only LSTM classifier for Poland 2025 Stage I: hidden subsequences.

The architecture and optimization follow the organizer's strongest notebook
variant (8-class, 2-layer LSTM using every time-step output). It has 49,512
trainable parameters and accepts at most 4,000 minibatch updates. The script
does not download data and never uses validation targets during training.

Usage:
    python hidden_subsequences.py train_dataset.csv val_dataset.csv

CSV rows contain 24 binary inputs followed by the target value. PyTorch is the
only non-standard dependency. The official task expects a GPU; CPU execution
is supported for smoke checks but may exceed the competition time limit.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEQUENCE_LENGTH = 24
MAX_STEPS = 4000
DEFAULT_BATCH_SIZE = 128


def seed_everything(seed: int = 12345) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_csv_split(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load 24 binary columns and one target column from an official CSV."""
    values = pd.read_csv(path).to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != SEQUENCE_LENGTH + 1:
        raise ValueError(
            f"expected {SEQUENCE_LENGTH} input columns plus one target, got {values.shape}"
        )
    x = torch.from_numpy(values[:, :-1])
    y = torch.from_numpy(values[:, -1])
    if not torch.isin(x, torch.tensor([0.0, 1.0])).all():
        raise ValueError("input sequence must contain only 0 and 1")
    return x, y


class HiddenSubsequenceLSTM(nn.Module):
    """Organizer v3 classifier; forward returns values in the task's target set."""

    def __init__(self, possible_values: torch.Tensor | np.ndarray | list[float]):
        super().__init__()
        values = torch.as_tensor(possible_values, dtype=torch.float32).flatten().sort().values
        if values.numel() == 0:
            raise ValueError("possible_values cannot be empty")
        self.register_buffer("possible_values", values)
        self.lstm = nn.LSTM(input_size=1, hidden_size=56, num_layers=2, batch_first=True)
        self.head = nn.Linear(56 * SEQUENCE_LENGTH, values.numel())

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().unsqueeze(-1)
        recurrent, _ = self.lstm(x)
        return self.head(recurrent.reshape(x.shape[0], -1))

    def encode_targets(self, y: torch.Tensor) -> torch.Tensor:
        values = self.possible_values.to(y.device)
        class_ids = torch.searchsorted(values, y.float().reshape(-1))
        if (class_ids >= values.numel()).any() or not torch.equal(
            values[class_ids], y.float().reshape(-1)
        ):
            raise ValueError("target value was not present in the training target set")
        return class_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_ids = self.logits(x).argmax(dim=1)
        return self.possible_values[class_ids]


def train_model(
    train_csv: str | Path,
    *,
    steps: int = MAX_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 0.01,
    seed: int = 12345,
    device: str | torch.device | None = None,
) -> tuple[HiddenSubsequenceLSTM, torch.device]:
    """Fit only on the supplied training CSV for no more than 4,000 steps."""
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be between 1 and {MAX_STEPS}")
    seed_everything(seed)
    x_train, y_train = read_csv_split(train_csv)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    if len(train_loader) == 0:
        raise ValueError("training CSV contains no examples")

    target_values = torch.unique(y_train, sorted=True)
    model = HiddenSubsequenceLSTM(target_values)
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    batches = iter(train_loader)
    model.train()
    for _ in range(steps):
        try:
            x_batch, y_batch = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            x_batch, y_batch = next(batches)
        x_batch = x_batch.to(target_device)
        y_batch = y_batch.to(target_device)
        class_targets = model.encode_targets(y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model.logits(x_batch), class_targets)
        loss.backward()
        optimizer.step()

    return model.eval(), target_device


@torch.no_grad()
def evaluate_mse(
    model: HiddenSubsequenceLSTM,
    csv_path: str | Path,
    device: str | torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> float:
    """Score a released split after fitting; its targets never enter training."""
    x, y = read_csv_split(csv_path)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    batch_mses = []
    model.eval()
    for x_batch, y_batch in loader:
        predictions = model(x_batch.to(device)).cpu()
        batch_mses.append(float(torch.mean((predictions - y_batch) ** 2)))
    return float(np.mean(batch_mses))


def estimate_points(mse: float) -> int:
    return int(round(max(100.0 * (64.0 - mse) / 64.0, 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", help="released training CSV")
    parser.add_argument("valid_csv", help="released validation CSV; used only for scoring")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=None, help="defaults to CUDA when available")
    args = parser.parse_args()

    model, device = train_model(
        args.train_csv, steps=args.steps, batch_size=args.batch_size, device=args.device
    )
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    mse = evaluate_mse(model, args.valid_csv, device, batch_size=args.batch_size)
    print(f"trainable parameters: {count}")
    print(f"validation MSE: {mse:.4f}")
    print(f"estimated score: {estimate_points(mse)}/100")


if __name__ == "__main__":
    main()

