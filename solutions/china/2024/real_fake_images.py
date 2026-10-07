"""Constraint-compliant CNN for NOAI China 2024 task 2.

Expected folders are ``train_v1/cifar`` (real, label 0) and
``train_v1/uvit`` (generated, label 1). Run ``--mode validate`` for a local
stratified holdout estimate or ``--mode train`` to save a trained checkpoint.
The script never downloads or copies competition assets.
"""

from __future__ import annotations

import argparse
import random
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


class MyModel(nn.Module):
    """Two convolution and two max-pool layers with a selectable head."""

    def __init__(self, head: str = "global") -> None:
        super().__init__()
        if head not in {"dense", "global"}:
            raise ValueError("head must be 'dense' or 'global'")
        self.head = head
        channels1, channels2 = (24, 48) if head == "global" else (32, 64)
        self.conv1 = nn.Conv2d(3, channels1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        if head == "global":
            self.output = nn.Linear(channels2, 1)
        else:
            self.hidden = nn.Linear(channels2 * 8 * 8, 64)
            self.output = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        if self.head == "global":
            x = x.mean(dim=(2, 3))
        else:
            x = x.flatten(1)
            x = self.relu2(self.hidden(x))
        return self.output(x).squeeze(-1)


class ImageFolders(Dataset):
    def __init__(self, root: Path) -> None:
        self.samples: list[tuple[Path, int]] = []
        for folder, label in (("cifar", 0), ("uvit", 1)):
            image_dir = root / folder
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Expected image directory: {image_dir}")
            self.samples.extend((path, label) for path in sorted(image_dir.glob("*.png")))
        if not self.samples:
            raise ValueError(f"No PNG images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if array.shape != (32, 32, 3):
            raise ValueError(f"Expected a 32x32 RGB image, got {array.shape} from {path}")
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor, label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: MyModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.float32)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(all_logits), np.concatenate(all_labels)


def validate(root: Path, seed: int, epochs: int, batch_size: int, head: str) -> None:
    seed_everything(seed)
    dataset = ImageFolders(root)
    labels = np.asarray([label for _, label in dataset.samples], dtype=np.int64)
    train_idx, valid_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        train_loss, train_logits, train_y = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_logits, val_y = run_epoch(model, valid_loader, None, device)
        val_accuracy = accuracy_score(val_y, val_logits >= 0)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_accuracy={accuracy_score(train_y, train_logits >= 0):.4f} "
            f"valid_loss={val_loss:.4f} valid_accuracy={val_accuracy:.4f}"
        )
    num_linear = 1 if head == "global" else 2
    simplicity = 1.0 / (2 + num_linear + 1)
    score = (simplicity + float(val_accuracy)) * 0.75
    print(
        f"local_valid_accuracy={val_accuracy:.4f} "
        f"implied_task_score={score:.4f} simplicity={simplicity:.4f} "
        f"images={len(dataset)} train={len(train_idx)} valid={len(valid_idx)}"
    )


def submission_model_source(head: str) -> str:
    """Standalone architecture matching the weights placed in submission_dic.pth."""
    if head == "global":
        return '''import torch\nfrom torch import nn\n\n\nclass MyModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, padding=1)\n        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, padding=1)\n        self.relu1 = nn.ReLU()\n        self.relu2 = nn.ReLU()\n        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)\n        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)\n        self.output = nn.Linear(48, 1)\n        self.sigmoid = nn.Sigmoid()\n\n    def forward(self, x):\n        x = self.pool1(self.relu1(self.conv1(x)))\n        x = self.pool2(self.relu2(self.conv2(x)))\n        x = x.mean(dim=(2, 3))\n        return self.sigmoid(self.output(x))\n'''
    return '''import torch\nfrom torch import nn\n\n\nclass MyModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)\n        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)\n        self.relu1 = nn.ReLU()\n        self.relu2 = nn.ReLU()\n        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)\n        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)\n        self.hidden = nn.Linear(64 * 8 * 8, 64)\n        self.output = nn.Linear(64, 1)\n        self.sigmoid = nn.Sigmoid()\n\n    def forward(self, x):\n        x = self.pool1(self.relu1(self.conv1(x)))\n        x = self.pool2(self.relu2(self.conv2(x)))\n        x = x.flatten(1)\n        x = self.relu2(self.hidden(x))\n        return self.sigmoid(self.output(x))\n'''


def write_submission_package(model: MyModel, head: str, output: Path) -> None:
    """Write the required standalone model source and raw state dict into a zip."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real-fake-submission-") as temp_dir:
        temp = Path(temp_dir)
        model_file = temp / "submission_model.py"
        weights_file = temp / "submission_dic.pth"
        model_file.write_text(submission_model_source(head), encoding="utf-8")
        torch.save(model.state_dict(), weights_file)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(model_file, arcname="submission_model.py")
            archive.write(weights_file, arcname="submission_dic.pth")


def train_all(root: Path, output: Path, seed: int, epochs: int, batch_size: int, head: str) -> None:
    seed_everything(seed)
    dataset = ImageFolders(root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        loss, logits, labels = run_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch} loss={loss:.4f} accuracy={accuracy_score(labels, logits >= 0):.4f}")
    write_submission_package(model.cpu(), head, output)
    print(f"saved {output} (submission_model.py, submission_dic.pth; head={head})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["validate", "train"], default="validate")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing cifar/ and uvit/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--head", choices=["dense", "global"], default="global")
    parser.add_argument("--output", type=Path, default=Path("submission.zip"), help="Output submission.zip")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "validate":
        validate(args.data_dir, args.seed, args.epochs, args.batch_size, args.head)
    else:
        train_all(args.data_dir, args.output, args.seed, args.epochs, args.batch_size, args.head)


if __name__ == "__main__":
    main()
