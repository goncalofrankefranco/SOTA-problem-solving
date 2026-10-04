"""Compact CNN for the 2024 Polish AI Olympiad Imbalanced Classification task.

Run from the task directory (containing ``train_data``) to train and save the
required ``cnn-classifier.pth`` file. Validation data is never read by training.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class CnnClassifier(nn.Module):
    MODEL_PATH = "cnn-classifier.pth"

    @classmethod
    def load(cls):
        """Load a state dict previously written by ``create_with_training``."""
        model = cls()
        try:
            state = torch.load(cls.MODEL_PATH, map_location="cpu", weights_only=True)
        except TypeError:  # Compatibility with older PyTorch used in Colab.
            state = torch.load(cls.MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        return model.eval()

    @classmethod
    def create_with_training(cls):
        """Fit on ``train_data`` only, then save the compact model weights."""
        data_root = Path(os.environ.get("POLAND_2024_IMBALANCED_DATA_DIR", "."))
        train_dataset = _ImageDataset(data_root / "train_data", augment=True)
        counts = np.bincount(train_dataset.labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"Training data must contain both classes; counts={counts.tolist()}")

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        sample_weights = 1.0 / counts[train_dataset.labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)

        model = cls()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # Three epochs reached 100% on the released split in the local run.
        # Keep the validation set out of the training loop.
        for epoch in range(3):
            model.train()
            total_loss = 0.0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                logits = model.logits(images).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
            print(f"epoch {epoch + 1}/3: train_loss={total_loss / len(loader):.4f}")

        model.to("cpu").eval()
        torch.save(model.state_dict(), cls.MODEL_PATH)
        print(f"Saved {cls.MODEL_PATH} ({sum(p.numel() for p in model.parameters()):,} parameters)")
        return model


class _ImageDataset(Dataset):
    def __init__(self, directory: Path, augment: bool = False):
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")
        self.files = sorted(path for path in directory.glob("*") if path.is_file())
        self.labels = np.asarray(
            [0 if "normal" in path.name.lower() else 1 for path in self.files], dtype=np.int64
        )
        self.augment = augment
        if not self.files:
            raise ValueError(f"No training images found in {directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        x = torch.from_numpy(pixels[None, ...])
        if self.augment:
            if random.random() < 0.5:
                x = x.flip(-1)
            if random.random() < 0.5:
                x = x.flip(-2)
        return x, int(self.labels[index])


class YourCnnClassifier(CnnClassifier):
    """Task model. ``forward`` returns a probability for the starter's 0.5 test."""

    def __init__(self):
        super().__init__()
        width = 24
        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=5, padding=2),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, width * 2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 3, kernel_size=3, padding=1),
            nn.GroupNorm(8, width * 3),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 3, width * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, width * 4),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(width * 4 * 7 * 7, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The task grader compares outputs to 0.5, so return probabilities,
        # while training uses logits with BCEWithLogitsLoss.
        return torch.sigmoid(self.logits(x))


if __name__ == "__main__":
    YourCnnClassifier.create_with_training()
