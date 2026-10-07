"""Weakly supervised digit model for NOAI China 2026 Task 3.

The model learns from the released aggregate labels (sum and product) by
marginalising over every four-digit sequence consistent with those labels.
The organizer's images and labels are required at run time and are not stored
in this repository.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def make_tuple_lookup() -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Return every ordered digit tuple and its corresponding output pair."""
    tuples = np.indices((10, 10, 10, 10), dtype=np.int64).reshape(4, -1).T
    sums = tuples.sum(axis=1)
    products = tuples.prod(axis=1)
    pairs = np.column_stack([sums, products])
    unique_pairs, pair_ids = np.unique(pairs, axis=0, return_inverse=True)
    return (
        torch.as_tensor(tuples, dtype=torch.long),
        torch.as_tensor(pair_ids, dtype=torch.long),
        unique_pairs[:, 0].astype(np.int64),
        unique_pairs[:, 1].astype(np.int64),
    )


TUPLES, TUPLE_PAIR_IDS, PAIR_SUMS, PAIR_PRODUCTS = make_tuple_lookup()
PAIR_TO_ID = {(int(s), int(p)): i for i, (s, p) in enumerate(zip(PAIR_SUMS, PAIR_PRODUCTS))}


class ProductSumDataset(Dataset):
    def __init__(self, records: list[dict], image_dir: str | Path, labeled: bool = True):
        self.records = records
        self.image_dir = Path(image_dir)
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.image_dir / f"{record['id']}.png"
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
        if image.shape != (28, 112):
            raise ValueError(f"Expected a 28x112 grayscale image, got {image.shape}: {image_path}")
        image = (image - 0.1307) / 0.3081
        tensor = torch.from_numpy(image).unsqueeze(0)
        if self.labeled:
            return tensor, int(record["sum"]), int(record["product"])
        return tensor, str(record["id"])


class DigitEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 10))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Shared digit recognition across the four known 28-pixel segments.
        batch = images.shape[0]
        crops = torch.stack([images[..., i * 28:(i + 1) * 28] for i in range(4)], dim=1)
        crops = crops.reshape(batch * 4, 1, 28, 28)
        return self.classifier(self.features(crops)).reshape(batch, 4, 10)


def pair_probabilities(logits: torch.Tensor) -> torch.Tensor:
    log_digit_p = torch.log_softmax(logits, dim=-1)
    tuple_ids = TUPLES.to(logits.device)
    joint_log = torch.zeros(logits.shape[0], tuple_ids.shape[0], device=logits.device)
    for position in range(4):
        joint_log += log_digit_p[:, position, :][:, tuple_ids[:, position]]
    probabilities = torch.exp(joint_log)
    pair_ids = TUPLE_PAIR_IDS.to(logits.device).expand(logits.shape[0], -1)
    output = torch.zeros(logits.shape[0], len(PAIR_SUMS), device=logits.device)
    return output.scatter_add_(1, pair_ids, probabilities)


def marginal_pair_loss(logits: torch.Tensor, sums: torch.Tensor, products: torch.Tensor) -> torch.Tensor:
    pair_p = pair_probabilities(logits)
    pair_ids = [PAIR_TO_ID.get((int(s), int(p))) for s, p in zip(sums.tolist(), products.tolist())]
    if any(i is None for i in pair_ids):
        raise ValueError("A training target is not attainable by four decimal digits")
    target = torch.as_tensor(pair_ids, dtype=torch.long, device=logits.device)
    selected = pair_p.gather(1, target[:, None]).squeeze(1).clamp_min(1e-12)
    return -torch.log(selected).mean()


def read_labels(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        row["sum"] = int(row["sum"])
        row["product"] = int(row["product"])
    if not records:
        raise ValueError(f"No labeled examples in {path}")
    return records


def fit_model(records: list[dict], image_dir: str | Path, device: torch.device, epochs: int = 25, seed: int = 42) -> DigitEncoder:
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(ProductSumDataset(records, image_dir), batch_size=96, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    model = DigitEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        total, seen = 0.0, 0
        for images, sums, products in loader:
            images = images.to(device, non_blocking=True)
            sums, products = sums.to(device), products.to(device)
            loss = marginal_pair_loss(model(images), sums, products)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
            seen += len(images)
        scheduler.step()
        print(f"epoch {epoch + 1:02d}/{epochs}: aggregate negative log likelihood={total / seen:.4f}")
    return model


@torch.no_grad()
def predict(
    model: DigitEncoder,
    image_dir: str | Path,
    device: torch.device,
    batch_size: int = 128,
    sample_ids: set[str] | None = None,
) -> list[dict]:
    paths = sorted(Path(image_dir).glob("*.png"))
    if sample_ids is not None:
        paths = [p for p in paths if p.stem in sample_ids]
    if not paths:
        raise FileNotFoundError(f"No PNG images in {image_dir}")
    records = [{"id": p.stem} for p in paths]
    loader = DataLoader(ProductSumDataset(records, image_dir, labeled=False), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    model.eval()
    rows: list[dict] = []
    for images, sample_ids in loader:
        image_logits = model(images.to(device, non_blocking=True))
        pair_p = pair_probabilities(image_logits).cpu().numpy()
        best = pair_p.argmax(axis=1)
        for sample_id, pair_id in zip(sample_ids, best):
            rows.append({"id": str(sample_id), "sum": int(PAIR_SUMS[pair_id]), "product": int(PAIR_PRODUCTS[pair_id])})
    return rows


def write_submission(records: list[dict], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "sum", "product"])
        writer.writeheader()
        writer.writerows(records)


def accuracy(records: list[dict], expected: list[dict]) -> tuple[float, float, float]:
    pred = {str(r["id"]): r for r in records}
    gold = {str(r["id"]): r for r in expected}
    common = sorted(set(pred) & set(gold))
    if len(common) != len(gold):
        raise ValueError("Validation predictions do not cover every held-out id")
    sum_acc = np.mean([pred[i]["sum"] == gold[i]["sum"] for i in common])
    product_acc = np.mean([pred[i]["product"] == gold[i]["product"] for i in common])
    return float((sum_acc + product_acc) / 2), float(sum_acc), float(product_acc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "submit"), default="submit")
    parser.add_argument("--train-dir", default="/bohr/train-rppd/v1")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_PATH", "/bohr"))
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(args.train_dir)
    labels_path = next((p for p in (train_dir / "train_labels.csv", train_dir / "train.csv") if p.is_file()), None)
    if labels_path is None:
        raise FileNotFoundError(f"Expected train_labels.csv or train.csv under {train_dir}")
    labels = read_labels(labels_path)
    if args.mode == "validate":
        rng = np.random.default_rng(42)
        order = rng.permutation(len(labels))
        boundary = int(0.9 * len(order))
        train_records = [labels[i] for i in order[:boundary]]
        val_records = [labels[i] for i in order[boundary:]]
        model = fit_model(train_records, train_dir / "train_images", device, args.epochs)
        val_ids = {str(r["id"]) for r in val_records}
        val_preds = predict(model, train_dir / "train_images", device, sample_ids=val_ids)
        _, sum_acc, product_acc = accuracy(val_preds, val_records)
        print(f"holdout exact accuracy: sum={sum_acc:.6f}, product={product_acc:.6f}")
        return

    model = fit_model(labels, train_dir / "train_images", device, args.epochs)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        rows = predict(model, data_dir / split, device)
        write_submission(rows, out_dir / f"submission_{split}.csv")
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(out_dir / "submission_val.csv", "submission_val.csv")
        archive.write(out_dir / "submission_test.csv", "submission_test.csv")
    print(f"Wrote {out_dir / 'submission.zip'} on {device}")


if __name__ == "__main__":
    main()
