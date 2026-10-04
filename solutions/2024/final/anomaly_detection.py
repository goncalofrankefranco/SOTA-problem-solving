"""One-class image anomaly detection for the 2024 Polish AI Olympiad final.

The fit path sees only the official normal-only training split.  A small neural
autoencoder learns a representation of handcrafted, low-resolution image
statistics; a shrinkage Gaussian fitted to the normal training representations
and statistics supplies the anomaly score.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
from PIL import Image
from sklearn.covariance import LedoitWolf
from torch import nn
from torch.utils.data import DataLoader, Dataset


BATCH_SIZE = 128
IMAGE_SIZE = 32
AUTOENCODER_EPOCHS = 10
AUTOENCODER_BATCH_SIZE = 128
NORMAL_QUANTILE = 0.95
SEED = 0


def image_features(batch: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return 353 color, texture, and spatial statistics per RGB image.

    PIL resizing and the same feature definitions are used for training and
    prediction. Inputs may be uint8 images or float tensors in [0, 1].
    """
    if isinstance(batch, torch.Tensor):
        array = batch.detach().to(device="cpu").permute(0, 2, 3, 1).numpy()
    else:
        array = np.asarray(batch)
        if array.ndim == 3:
            array = array[None, ...]
        if array.shape[1] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 1, -1)

    rows = []
    for image in array:
        image = image[..., :3]
        if image.dtype != np.uint8:
            if float(np.nanmax(image)) <= 1.0:
                image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        rgb = np.asarray(Image.fromarray(image, mode="RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
        parts: list[np.ndarray | float] = []

        for channel in range(3):
            values = rgb[:, :, channel]
            mean = float(values.mean())
            std = float(values.std())
            centered = values - mean
            parts.extend(
                [
                    mean,
                    std,
                    *np.quantile(values, [0.1, 0.5, 0.9]).tolist(),
                    float(np.mean(centered**3) / (std**3 + 1e-6)),
                    float(np.mean(centered**4) / (std**4 + 1e-6)),
                ]
            )
            histogram, _ = np.histogram(values, bins=16, range=(0.0, 1.0), density=True)
            parts.extend(histogram.tolist())

        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        laplacian = scipy.ndimage.laplace(gray)
        for values in (gray, np.abs(np.diff(gray, axis=0)), np.abs(np.diff(gray, axis=1)), np.abs(laplacian)):
            parts.extend(
                [
                    float(values.mean()),
                    float(values.std()),
                    *np.quantile(values, [0.9, 0.99]).tolist(),
                    float(np.mean(values < 0.02)),
                    float(np.mean(values > 0.15)),
                ]
            )

        maximum = rgb.max(axis=2)
        minimum = rgb.min(axis=2)
        saturation = (maximum - minimum) / (maximum + 1e-3)
        parts.extend(
            [
                float(saturation.mean()),
                float(saturation.std()),
                float(np.quantile(saturation, 0.9)),
                float(np.mean(saturation > 0.5)),
            ]
        )

        # Coarse RGB layout and local gray-texture maps.
        parts.extend(rgb.reshape(8, 4, 8, 4, 3).mean(axis=(1, 3)).ravel().tolist())
        parts.extend(gray.reshape(8, 4, 8, 4).std(axis=(1, 3)).ravel().tolist())
        rows.append(parts)

    features = np.asarray(rows, dtype=np.float32)
    if features.shape[1] != 353:
        raise RuntimeError(f"Expected 353 features per image, got {features.shape[1]}.")
    return features


class ImageDataset(Dataset):
    """Dataset compatible with the official CSV and image-folder layout."""

    def __init__(self, image_dir: str | Path, csv_path: str | Path):
        self.image_dir = Path(image_dir)
        self.frame = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image = np.asarray(Image.open(self.image_dir / row.iloc[0]).convert("RGB"), dtype=np.uint8)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        return tensor, int(row.iloc[1])


class Model(nn.Module):
    """Neural one-class detector with a training-only normality threshold."""

    def __init__(self, feature_count: int = 353, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, feature_count),
        )
        self.register_buffer("feature_mean", torch.zeros(feature_count))
        self.register_buffer("feature_scale", torch.ones(feature_count))
        combined_size = feature_count + latent_dim
        self.register_buffer("density_mean", torch.zeros(combined_size))
        self.register_buffer("density_precision", torch.eye(combined_size))
        self.register_buffer("threshold", torch.tensor(float("inf")))

    def _encode(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.as_tensor(image_features(batch), device=self.feature_mean.device)
        scaled = (features - self.feature_mean) / self.feature_scale
        latent = self.encoder(scaled)
        return scaled, latent

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image batch's standardized feature vectors."""
        _, latent = self._encode(batch)
        return self.decoder(latent)

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        scaled, latent = self._encode(batch)
        combined = torch.cat((scaled, latent), dim=1)
        delta = (combined - self.density_mean).double()
        precision = self.density_precision.double()
        return torch.einsum("bi,ij,bj->b", delta, precision, delta).to(dtype=torch.float32)

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Return 1 for anomalies and 0 for normal samples."""
        scores = self.anomaly_score(batch)
        return (scores > self.threshold).to(dtype=torch.int64)


def train_model(
    dataloader: Iterable,
    device: str | torch.device | None = None,
    epochs: int = AUTOENCODER_EPOCHS,
) -> Model:
    """Fit exclusively on the official normal-only training loader."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    feature_batches = []
    for images, _ in dataloader:
        feature_batches.append(image_features(images))
    features = np.concatenate(feature_batches, axis=0)

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = ((features - mean) / scale).astype(np.float32)

    model = Model(feature_count=features.shape[1]).to(device)
    model.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32, device=device))
    model.feature_scale.copy_(torch.as_tensor(scale, dtype=torch.float32, device=device))

    train_tensor = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_tensor), generator=generator)
        for indices in order.split(AUTOENCODER_BATCH_SIZE):
            values = train_tensor[indices.to(device)]
            reconstruction = model.decoder(model.encoder(values))
            loss = nn.functional.mse_loss(reconstruction, values)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        latent = model.encoder(train_tensor).cpu().numpy()
    combined = np.concatenate((standardized, latent), axis=1)

    density = LedoitWolf().fit(combined)
    training_scores = density.mahalanobis(combined)
    model.density_mean.copy_(torch.as_tensor(density.location_, dtype=torch.float32, device=device))
    model.density_precision.copy_(torch.as_tensor(density.precision_, dtype=torch.float32, device=device))
    model.threshold.copy_(torch.as_tensor(np.quantile(training_scores, NORMAL_QUANTILE), dtype=torch.float32, device=device))
    return model


def train_from_paths(
    train_dir: str | Path = "train",
    train_csv: str | Path = "train.csv",
    device: str | torch.device | None = None,
) -> Model:
    dataset = ImageDataset(train_dir, train_csv)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_model(loader, device=device)


if __name__ == "__main__":
    model = train_from_paths()
    print(f"Fitted one-class model on normal data; threshold={float(model.threshold):.4f}")
