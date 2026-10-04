"""Multi-task denoising model for the 2025 Polish AI Olympiad noise task.

Training uses only the released training pairs, their noise-type labels, and
the pixel censoring caused by clipping. Validation/test parameter values are
never read by this module.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class _PairedImages(Dataset):
    def __init__(self, path: str | Path):
        with Path(path).open("rb") as handle:
            rows = pickle.load(handle)
        self.original = np.stack([row["original"] for row in rows]).astype(np.float32)
        self.noised = np.stack([row["noised"] for row in rows]).astype(np.float32)
        self.labels = np.asarray([row["label"] for row in rows], dtype=np.float32)
        if self.original.max() > 1.0:
            self.original /= 255.0
            self.noised /= 255.0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        clean = torch.from_numpy(self.original[index]).permute(2, 0, 1)
        noised = torch.from_numpy(self.noised[index]).permute(2, 0, 1)
        return {
            "original": clean,
            "noised": noised,
            "label": torch.tensor(self.labels[index]),
        }


class Model(nn.Module):
    """Predict the clean image, noise family, and Gaussian parameters."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.noise_residual = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.noise_residual.weight)
        nn.init.zeros_(self.noise_residual.bias)

        self.global_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )
        self.noise_type_head = nn.Linear(64, 1)
        self.parameter_head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor):
        features = self.features(x)
        residual = self.noise_residual(features)
        denoised = (x - residual).clamp(0.0, 1.0)

        pooled = self.global_features(features)
        noise_type = torch.sigmoid(self.noise_type_head(pooled))
        raw_parameters = self.parameter_head(pooled)
        mean = 0.5 * torch.sigmoid(raw_parameters[:, :1])
        std = 0.05 + 0.25 * torch.sigmoid(raw_parameters[:, 1:2])
        return denoised, noise_type, mean, std


def _gaussian_censored_nll(
    original: torch.Tensor,
    noised: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Gaussian likelihood with exact and left/right censored pixels."""
    clean = original.flatten(1)
    observed = noised.flatten(1)
    mu = mean
    sigma = std.clamp_min(1e-4)

    lower_censored = observed <= 0.0
    upper_censored = observed >= 1.0
    exact = ~(lower_censored | upper_censored)
    residual = observed - clean

    exact_log_likelihood = (
        -0.5 * ((residual - mu) / sigma).square()
        - torch.log(sigma)
        - 0.9189385332
    )
    lower_log_likelihood = torch.special.log_ndtr((-clean - mu) / sigma)
    upper_log_likelihood = torch.special.log_ndtr((mu - (1.0 - clean)) / sigma)
    log_likelihood = torch.where(
        lower_censored,
        lower_log_likelihood,
        torch.where(upper_censored, upper_log_likelihood, exact_log_likelihood),
    )
    return -log_likelihood.mean(dim=1)


def _refit_classifier_head(
    model: Model,
    dataset: Dataset,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 1024,
) -> None:
    """Tune the noise classifier on frozen CNN features from training rows."""
    model.eval()
    features = []
    labels = []
    extraction_loader = DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0
    )
    with torch.no_grad():
        for batch in extraction_loader:
            noised = batch["noised"].to(device=device, dtype=torch.float32)
            features.append(model.global_features(model.features(noised)))
            labels.append(batch["label"].to(device=device, dtype=torch.float32).reshape(-1))

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    torch.manual_seed(7)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)
    optimizer = torch.optim.Adam(
        model.noise_type_head.parameters(), lr=0.02, weight_decay=1e-4
    )
    for _ in range(epochs):
        permutation = torch.randperm(len(features), device=device)
        for indices in permutation.split(batch_size):
            logits = model.noise_type_head(features[indices]).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def train_model(
    train_loader=None,
    data_root: str | Path = ".",
    device: torch.device | str | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> Model:
    """Train from train.pkl or an official-style training DataLoader.

    No validation data or validation parameter targets are accessed. When a
    loader is provided, its training dataset is rebatched for throughput.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if train_loader is None:
        dataset = _PairedImages(Path(data_root) / "train.pkl")
    else:
        dataset = train_loader.dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        for batch in loader:
            original = batch["original"].to(device=device, dtype=torch.float32)
            noised = batch["noised"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.float32).reshape(-1)

            denoised, noise_type, mean, std = model(noised)
            denoising_loss = F.mse_loss(denoised, original)
            classification_loss = F.binary_cross_entropy(
                noise_type.reshape(-1), labels
            )

            gaussian_mask = labels == 0.0
            if gaussian_mask.any():
                parameter_loss = _gaussian_censored_nll(
                    original[gaussian_mask],
                    noised[gaussian_mask],
                    mean[gaussian_mask],
                    std[gaussian_mask],
                ).mean()
            else:
                parameter_loss = denoising_loss.new_zeros(())

            loss = denoising_loss + 0.15 * classification_loss + 0.10 * parameter_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

    _refit_classifier_head(model, dataset, device)
    model.eval()
    return model
