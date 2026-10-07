#!/usr/bin/env python3
"""Train a CNN and write a Kaggle submission for Colombia's Five Artists task.

The script downloads the official Hugging Face datasets. It first uses a
stratified holdout to estimate accuracy and select the epoch count, then trains
on all labeled examples before predicting the 2,000 test images.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

TRAIN_DATASET = "eleon360/five-artists-dataset"
TEST_DATASET = "eleon360/five-artists-test-dataset"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ArtistImages(Dataset):
    def __init__(self, source, indices, size: int, augment: bool):
        self.source = source
        self.indices = list(map(int, indices))
        base = [transforms.Resize((size, size), antialias=True)]
        if augment:
            base.extend(
                [
                    transforms.RandomRotation(degrees=6),
                    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02),
                ]
            )
        base.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
            ]
        )
        self.transform = transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        row = self.source[self.indices[item]]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        tensor = self.transform(image)
        if "artist_id" in row:
            return tensor, int(row["artist_id"])
        return tensor, int(row["image_id"])


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class ArtistCNN(nn.Module):
    def __init__(self, classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(48, 64, 2),
            ResidualBlock(64, 96, 2),
            ResidualBlock(96, 160, 2),
            ResidualBlock(160, 256, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.30), nn.Linear(256, classes))

    def forward(self, x):
        return self.head(self.pool(self.features(self.stem(x))))


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader | None,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, int, dict[str, float] | None]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = max_epochs
    best_accuracy = -1.0
    best_metrics = None
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
        scheduler.step()

        mean_loss = total_loss / max(len(train_loader.dataset), 1)
        if valid_loader is None:
            print(f"epoch={epoch:02d}/{max_epochs} train_loss={mean_loss:.4f}")
            best_state = copy.deepcopy(model.state_dict())
            continue

        y_true, y_pred = predict(model, valid_loader, device)
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"epoch={epoch:02d} train_loss={mean_loss:.4f} valid_accuracy={accuracy:.5f} macro_f1={macro_f1:.5f}")
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {"accuracy": float(accuracy), "macro_f1": float(macro_f1)}
            stale = 0
        else:
            stale += 1
            if epoch >= 8 and stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_metrics


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(torch.as_tensor(labels).cpu().tolist())
    return targets, predictions


@torch.no_grad()
def predict_test(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    image_ids: list[int] = []
    predictions: list[int] = []
    for images, ids in loader:
        logits = model(images.to(device, non_blocking=True))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        image_ids.extend(torch.as_tensor(ids).cpu().tolist())
    return image_ids, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", default=TRAIN_DATASET)
    parser.add_argument("--test-dataset", default=TEST_DATASET)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artist_cnn.pt"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-column", default="artist_id", help="Use the Kaggle data-page target column")
    args = parser.parse_args()

    from datasets import load_dataset

    seed_everything(args.seed)
    train_data = load_dataset(args.train_dataset, split="train")
    test_data = load_dataset(args.test_dataset, split="train")
    labels = np.asarray(train_data["artist_id"], dtype=np.int64)
    if not set(np.unique(labels)).issubset(set(range(5))):
        raise ValueError("Expected artist_id labels 0 through 4")
    counts = np.bincount(labels, minlength=5)
    print(f"train={len(train_data):,}; test={len(test_data):,}; image=256x256 per Hugging Face viewer")
    print(f"artist_id class counts={counts.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    indices = np.arange(len(train_data))
    train_idx, valid_idx = train_test_split(
        indices, test_size=0.15, random_state=args.seed, stratify=labels
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(
        ArtistImages(train_data, train_idx, args.size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
    )
    valid_loader = DataLoader(
        ArtistImages(train_data, valid_idx, args.size, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
    )

    model = ArtistCNN().to(device)
    model, best_epoch, metrics = run_training(
        model, train_loader, valid_loader, device, args.epochs, args.patience
    )
    y_valid, y_valid_hat = predict(model, valid_loader, device)
    print(f"best_epoch={best_epoch}; holdout_accuracy={accuracy_score(y_valid, y_valid_hat):.6f}")
    print(f"holdout_macro_f1={f1_score(y_valid, y_valid_hat, average='macro', zero_division=0):.6f}")
    print(classification_report(y_valid, y_valid_hat, labels=list(range(5)), zero_division=0))

    # Refit on all labeled images for the number of epochs selected by holdout accuracy.
    seed_everything(args.seed)
    full_loader = DataLoader(
        ArtistImages(train_data, indices, args.size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
    )
    final_model = ArtistCNN().to(device)
    final_model, _, _ = run_training(final_model, full_loader, None, device, best_epoch, args.patience)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": final_model.state_dict(), "size": args.size, "best_epoch": best_epoch, "seed": args.seed},
        args.checkpoint,
    )

    test_loader = DataLoader(
        ArtistImages(test_data, np.arange(len(test_data)), args.size, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
    )
    image_ids, test_predictions = predict_test(final_model, test_loader, device)
    output = pd.DataFrame({"image_id": image_ids, args.prediction_column: test_predictions})
    output.to_csv(args.output, index=False)
    print(f"wrote {len(output):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
