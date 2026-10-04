"""Self-supervised HAR classifier for the 2024 Polish AI Olympiad final.

The included encoder checkpoint was trained on unlabeled ``train_x_big`` data.
At evaluation, the classifier is fitted only to the supplied ``train_x_small``
and ``train_y_small`` pair, as required by the task.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FP = Path(__file__).with_name("encoder.pt")
N_CHANNEL = 3
N_LENGTH = 206
PROJECTION_DIM = 64
LOGISTIC_BATCH_SIZE = 64
LOGISTIC_EPOCHS = 50
FINETUNE_SEED = 42
PRETRAIN_EPOCHS = 50


def setup_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class SimpleEncoder(nn.Module):
    """Three-stage temporal CNN; ``h`` is a 128-dimensional representation."""

    def __init__(self, projection_dim: int = PROJECTION_DIM, n_channel: int = N_CHANNEL, n_length: int = N_LENGTH):
        super().__init__()
        self.n_features = 128
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channel, 32, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.projector = nn.Sequential(nn.Linear(self.n_features, projection_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        h = self.encoder(x.float())
        z = self.projector(h)
        return h, z

    def save_model(self, path: str | Path = MODEL_FP) -> None:
        torch.save(self.state_dict(), path)


class MLPClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        hidden = max(n_classes, (n_features // max(n_classes, 1) // 2) * n_classes)
        self.model = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AddGaussianNoise(nn.Module):
    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class RandomScaling(nn.Module):
    def __init__(self, limits: tuple[float, float]):
        super().__init__()
        self.limits = limits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1, device=x.device).uniform_(*self.limits)
        return x * scale


class TimeMasking(nn.Module):
    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = int(x.shape[-1] * self.ratio)
        start = torch.randint(0, max(1, x.shape[-1] - length + 1), (1,)).item()
        out = x.clone()
        out[..., start : start + length] = 0
        return out


class RandomCrop(nn.Module):
    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = max(1, int(x.shape[-1] * self.ratio))
        start = torch.randint(0, max(1, x.shape[-1] - length + 1), (1,)).item()
        cropped = x[..., start : start + length]
        return F.interpolate(cropped, size=x.shape[-1], mode="linear", align_corners=False)


class UnlabelledViews(Dataset):
    def __init__(self, values: torch.Tensor):
        self.values = torch.as_tensor(values).float()
        self.view_a = nn.Sequential(AddGaussianNoise(0.05), RandomScaling((0.9, 1.1)))
        self.view_b = nn.Sequential(TimeMasking(0.1), RandomCrop(0.85))

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        x = self.values[index].clone()
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        return self.view_a(x), self.view_b(x), torch.tensor(0)


def contrastive_loss(z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
    """The original notebook's one-way batch InfoNCE objective."""
    targets = torch.arange(z_i.shape[0], device=z_i.device)
    logits = z_i @ z_j.T
    return F.cross_entropy(logits, targets)


def pretrain_encoder(
    X_train_big: torch.Tensor,
    model_path: str | Path = MODEL_FP,
    device: str | torch.device = DEVICE,
    epochs: int = PRETRAIN_EPOCHS,
) -> SimpleEncoder:
    """Learn temporal representations without reading class labels."""
    setup_seed(42)
    dataset = UnlabelledViews(X_train_big)
    loader = DataLoader(dataset, batch_size=LOGISTIC_BATCH_SIZE, shuffle=True, drop_last=True)
    encoder = SimpleEncoder().to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    encoder.train()
    for _ in range(epochs):
        for x_i, x_j, _ in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            _, z_i = encoder(x_i)
            _, z_j = encoder(x_j)
            loss = contrastive_loss(z_i, z_j)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    encoder.save_model(model_path)
    return encoder


def _fit_classifier(
    encoder: SimpleEncoder,
    X_train_small: torch.Tensor,
    y_train_small: torch.Tensor,
    device: torch.device,
) -> tuple[MLPClassifier, torch.Tensor, torch.Tensor]:
    labels = torch.as_tensor(y_train_small).reshape(-1).long()
    label_values, class_targets = torch.unique(labels, sorted=True, return_inverse=True)
    features = torch.as_tensor(X_train_small).float()
    if features.ndim == 4 and features.shape[1] == 1:
        features = features.squeeze(1)
    dataset = TensorDataset(features, class_targets)
    loader = DataLoader(dataset, batch_size=LOGISTIC_BATCH_SIZE, shuffle=True, drop_last=True)

    classifier = MLPClassifier(encoder.n_features, len(label_values)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    # This mirrors the released participant notebook: the encoder is frozen by
    # no_grad, while its BatchNorm running statistics adapt to labeled inputs.
    for _ in range(LOGISTIC_EPOCHS):
        encoder.train()
        classifier.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                h, _ = encoder(x)
            logits = classifier(h)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return classifier, label_values, features


def finetune_and_predict(
    X_train_small: torch.Tensor,
    y_train_small: torch.Tensor,
    X_test: torch.Tensor,
    model_path: str | Path = MODEL_FP,
) -> torch.Tensor:
    """Fit on labeled training examples and return one class index per test item."""
    setup_seed(FINETUNE_SEED)
    device = DEVICE
    path = Path(model_path)
    if not path.is_file():
        path = MODEL_FP
    encoder = SimpleEncoder()
    encoder.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    encoder = encoder.to(device)
    classifier, label_values, _ = _fit_classifier(encoder, X_train_small, y_train_small, device)

    x_test = torch.as_tensor(X_test).float()
    if x_test.ndim == 4 and x_test.shape[1] == 1:
        x_test = x_test.squeeze(1)
    # Retain training mode, matching the author notebook's BatchNorm behavior.
    encoder.train()
    classifier.eval()
    outputs = []
    with torch.no_grad():
        for batch in x_test.split(512):
            h, _ = encoder(batch.to(device))
            outputs.append(classifier(h).argmax(dim=1).cpu())
    class_ids = torch.cat(outputs).long()
    return label_values[class_ids]


def evaluate_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float((torch.as_tensor(y_true).reshape(-1).cpu() == torch.as_tensor(y_pred).reshape(-1).cpu()).float().mean())
