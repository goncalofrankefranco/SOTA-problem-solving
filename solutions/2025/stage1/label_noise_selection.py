"""Standalone training solution for Poland 2025 Stage I: label noise.

This is the organizer's Co-Teaching-style rule: model 1 uses model 2's losses
to keep the easiest examples, and vice versa. Selection is performed within
each observed class, with rates 50% for label 0 and 93% for label 1.

Competition callback:
    selected = your_select_indices(targets, losses)
    # selected[i] indexes the examples used to update model i.

The fixed SmallMobileNet architecture is copied from the official starter;
the train loop applies the organizer's selection callback. The script accepts
already extracted official ``train`` and ``val`` folders and never downloads
data. Validation labels are used only for final BAC scoring.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd


KEEP_RATE_BY_CLASS = (0.50, 0.93)
NUM_CLASSES = 2
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 6


def your_select_indices(
    targets: torch.Tensor,
    losses: Sequence[torch.Tensor],
    keep_rates: Sequence[float] = KEEP_RATE_BY_CLASS,
) -> list[list[int]]:
    """Return cross-selected batch indices for the two fixed classifiers.

    Args:
        targets: Noisy batch labels with shape ``(batch,)``.
        losses: Per-example losses ``[model_1_losses, model_2_losses]``.
        keep_rates: Fraction to keep for each class, ordered by labels 0 and 1.

    Returns:
        Two Python lists of indices, compatible with ``inputs[indices]`` in the
        official training loop. Selection only uses the batch labels and losses.
    """
    if len(losses) != 2:
        raise ValueError("losses must contain one per-example loss tensor per model")
    if len(keep_rates) != 2:
        raise ValueError("keep_rates must contain rates for classes 0 and 1")

    labels = targets.reshape(-1)
    loss_vectors = [loss.detach().reshape(-1) for loss in losses]
    if any(loss.numel() != labels.numel() for loss in loss_vectors):
        raise ValueError("each loss tensor must have one value per target")

    selected: list[list[int]] = [[], []]
    for class_tensor in torch.unique(labels):
        class_id = int(class_tensor.item())
        if class_id not in (0, 1):
            raise ValueError(f"expected binary labels 0 and 1, got {class_id}")
        class_indices = torch.nonzero(labels == class_id, as_tuple=True)[0]
        keep_count = int(class_indices.numel() * float(keep_rates[class_id]))
        if keep_count == 0:
            continue

        class_mask = labels == class_id
        for model_id in range(2):
            # Each network chooses examples for the other network.
            opposite_losses = loss_vectors[1 - model_id].to(labels.device)
            masked = opposite_losses.masked_fill(~class_mask, float("inf"))
            chosen = torch.topk(masked, k=keep_count, largest=False).indices
            selected[model_id].extend(chosen.detach().cpu().tolist())

    return selected


class SmallMobileNet(nn.Module):
    """The fixed grayscale classifier architecture supplied by the organizer."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            nn.Conv2d(32, 64, 1, stride=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.Conv2d(64, 128, 1, stride=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, 1, stride=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU6(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(torch.flatten(x, 1))


class TaskImageDataset(Dataset):
    """Read an extracted organizer split with ``dataset_labels.csv`` and ``data/``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.labels = pd.read_csv(self.root / "dataset_labels.csv")
        required = {"file_name", "label"}
        if not required.issubset(self.labels.columns):
            raise ValueError(f"{self.root}/dataset_labels.csv must contain {sorted(required)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.labels.iloc[index]
        image_path = self.root / "data" / str(row["file_name"])
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).unsqueeze(0)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tensor, label


def seed_everything(seed: int = 123) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fit_two_models(
    train_root: str | Path,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-3,
    device: str | torch.device | None = None,
) -> tuple[SmallMobileNet, SmallMobileNet, torch.device]:
    """Train both fixed networks; validation data is not accepted by this function."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    seed_everything()
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    loader = DataLoader(
        TaskImageDataset(train_root), batch_size=batch_size, shuffle=False, num_workers=0
    )
    models = [SmallMobileNet().to(target_device), SmallMobileNet().to(target_device)]
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for model in models
    ]
    per_example_loss = nn.CrossEntropyLoss(reduction="none")

    for _ in range(epochs):
        for model in models:
            model.train()
        for inputs, targets in loader:
            inputs = inputs.to(target_device)
            targets = targets.to(target_device).reshape(-1)
            losses = [per_example_loss(model(inputs), targets) for model in models]
            selected = your_select_indices(targets, losses)
            for i, (model, optimizer) in enumerate(zip(models, optimizers)):
                indices = torch.as_tensor(selected[i], dtype=torch.long, device=target_device)
                if indices.numel() == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(inputs[indices]), targets[indices])
                loss.backward()
                optimizer.step()
    return models[0].eval(), models[1].eval(), target_device


@torch.no_grad()
def validation_bac(
    model: SmallMobileNet,
    val_root: str | Path,
    device: str | torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> float:
    """Measure a trained model on the clean released validation split."""
    loader = DataLoader(TaskImageDataset(val_root), batch_size=batch_size, shuffle=False)
    predictions: list[int] = []
    targets_all: list[int] = []
    model.eval()
    for images, targets in loader:
        logits = model(images.to(device))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets_all.extend(targets.reshape(-1).cpu().tolist())
    return float(balanced_accuracy_score(targets_all, predictions))


def estimate_points(bac1: float, bac2: float) -> int:
    mean_bac = (bac1 + bac2) / 2.0
    return 0 if mean_bac <= 0.5 else (100 if mean_bac >= 0.8 else round(100 * (mean_bac - 0.5) / 0.3))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_root", help="extracted noisy training split")
    parser.add_argument("val_root", help="extracted clean validation split; used only for scoring")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=None, help="defaults to CUDA when available")
    args = parser.parse_args()

    model1, model2, device = fit_two_models(
        args.train_root, epochs=args.epochs, batch_size=args.batch_size, device=args.device
    )
    bac1 = validation_bac(model1, args.val_root, device, batch_size=args.batch_size)
    bac2 = validation_bac(model2, args.val_root, device, batch_size=args.batch_size)
    print(f"validation BAC: model 1={bac1:.6f}, model 2={bac2:.6f}")
    print(f"mean BAC: {(bac1 + bac2) / 2:.6f}; estimated score: {estimate_points(bac1, bac2)}/100")


if __name__ == "__main__":
    main()
