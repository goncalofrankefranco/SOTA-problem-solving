"""Weakly supervised detection of product-photo grid collages.

The organizer supplies category labels but no collage labels. This candidate
learns from each real image as a weak negative and synthetic mosaics made from
the same category as positives. A manual-label CSV can be supplied instead
when an annotator has labeled the official training images.
"""
from __future__ import annotations

import argparse
import csv
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_lookup(root: Path) -> dict[str, Path]:
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    result = {}
    for p in paths:
        result[p.name] = p
        result[p.stem] = p
    return result


def load_records(root: Path, csv_path: Path):
    frame = pd.read_csv(csv_path)
    if len(frame.columns) < 2:
        raise ValueError(f"Expected id and category columns in {csv_path}")
    id_col = next((c for c in frame.columns if str(c).lower() in {"id", "image", "filename", "file"}), frame.columns[0])
    cat_col = next((c for c in frame.columns if c != id_col), frame.columns[1])
    lookup = image_lookup(root)
    records = []
    for _, row in frame.iterrows():
        key = str(row[id_col])
        path = lookup.get(key) or lookup.get(Path(key).name) or lookup.get(Path(key).stem)
        if path is None:
            raise FileNotFoundError(f"Could not match image id {key!r} under {root}")
        records.append((path, str(row[cat_col])))
    return records


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def varied_tile(im: Image.Image, size: tuple[int, int], rng: random.Random) -> Image.Image:
    im = im.copy()
    w, h = im.size
    # Random crop simulates grid cells containing either full products or a
    # cropped part of a source image.
    scale = rng.uniform(0.55, 1.0)
    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
    left, top = rng.randint(0, max(0, w - cw)), rng.randint(0, max(0, h - ch))
    im = im.crop((left, top, left + cw, top + ch)).resize(size, Image.Resampling.BILINEAR)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.82, 1.18))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.12:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))
    return im


def make_collage(pool: list[Path], category: str, rng: random.Random, size: int = 256) -> Image.Image:
    same = [p for p, c in pool if c == category]
    candidates = same if len(same) >= 4 else [p for p, _ in pool]
    layout = rng.choice([(2, 2), (2, 3), (3, 2), (3, 3), (1, 2), (2, 1)])
    rows, cols = layout
    # Nonuniform cell edges cover free-form and irregular grid layouts.
    usable = size - rng.randint(0, 8) * (cols - 1)
    cuts_x = sorted([0, usable] + [rng.randint(usable // cols - 12, usable // cols + 12) * i for i in range(1, cols)])
    usable_y = size - rng.randint(0, 8) * (rows - 1)
    cuts_y = sorted([0, usable_y] + [rng.randint(usable_y // rows - 12, usable_y // rows + 12) * i for i in range(1, rows)])
    gap = rng.randint(0, 8)
    background = tuple(rng.randint(210, 255) for _ in range(3))
    canvas = Image.new("RGB", (size, size), background)
    for r in range(rows):
        for c in range(cols):
            x0, x1 = cuts_x[c], cuts_x[c + 1]
            y0, y1 = cuts_y[r], cuts_y[r + 1]
            x0 += gap // 2
            x1 -= gap - gap // 2
            y0 += gap // 2
            y1 -= gap - gap // 2
            if x1 <= x0 or y1 <= y0:
                continue
            path = rng.choice(candidates)
            canvas.paste(varied_tile(open_rgb(path), (x1 - x0, y1 - y0), rng), (x0, y0))
    # Apply plausible recompression to reduce reliance on sharp synthetic seams.
    if rng.random() < 0.45:
        from io import BytesIO
        buffer = BytesIO()
        canvas.save(buffer, format="JPEG", quality=rng.randint(45, 90))
        buffer.seek(0)
        canvas = Image.open(buffer).convert("RGB")
    return canvas


def to_tensor(im: Image.Image, train: bool = False) -> torch.Tensor:
    if train:
        if random.random() < 0.5:
            im = ImageOps.mirror(im)
        if random.random() < 0.6:
            im = ImageEnhance.Brightness(im).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.6:
            im = ImageEnhance.Color(im).enhance(random.uniform(0.8, 1.2))
    arr = np.asarray(im.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class WeakCollageDataset(Dataset):
    def __init__(self, records, seed: int, training: bool = True):
        self.records, self.seed, self.training = records, seed, training
        self.paths = [p for p, _ in records]
        self.cats = [c for _, c in records]

    def __len__(self):
        return len(self.records) * 2

    def __getitem__(self, i):
        row, is_collage = i // 2, i % 2
        path, category = self.records[row]
        if not is_collage:
            image = open_rgb(path)
        else:
            rng = random.Random(self.seed + i * 104729 + random.randint(0, 2**30))
            image = make_collage(self.records, category, rng)
        return to_tensor(image, train=self.training), torch.tensor(is_collage, dtype=torch.long)


class LabeledImages(Dataset):
    def __init__(self, items, training: bool):
        self.items, self.training = items, training
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label = self.items[i]
        return to_tensor(open_rgb(path), self.training), torch.tensor(label, dtype=torch.long)


def make_model(pretrained: bool = False):
    from torchvision.models import ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train_model(dataset: Dataset, epochs: int, batch_size: int, seed: int, device: torch.device, pretrained: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                        pin_memory=device.type == "cuda", drop_last=False)
    model = make_model(pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4 if pretrained else 1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch={epoch + 1} loss={total / max(1, len(loader)):.5f}")
    return model


def predict(model, records, device, batch_size: int):
    ds = LabeledImages([(p, 0) for p, _ in records], training=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    pred = []
    with torch.no_grad():
        for x, _ in loader:
            pred.extend(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy().tolist())
    return np.asarray(pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "submit"), default="submit")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--manual-labels", type=Path, help="Optional CSV with id,label for a small supervised validation/train experiment")
    parser.add_argument("--output-dir", type=Path, default=Path("submission"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pretrained", action="store_true", help="Use cached ResNet18 weights; this script does not download weights")
    args = parser.parse_args()
    records = load_records(args.train_dir, args.train_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "validate":
        if args.manual_labels is None:
            raise SystemExit("A task score cannot be measured without manual target labels; use --manual-labels for supervised validation")
        labels_frame = pd.read_csv(args.manual_labels)
        lookup = {str(row.iloc[0]): int(row.iloc[1]) for _, row in labels_frame.iterrows()}
        items = [(p, lookup.get(str(p.name), lookup.get(str(p.stem)))) for p, _ in records]
        items = [(p, y) for p, y in items if y in (0, 1)]
        train_items, val_items = train_test_split(items, test_size=0.2, random_state=args.seed,
                                                  stratify=[y for _, y in items])
        train = LabeledImages(train_items, training=True)
        model = train_model(train, args.epochs, args.batch_size, args.seed, device, args.pretrained)
        val = LabeledImages(val_items, training=False)
        loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=2)
        model.eval()
        gold, pred = [], []
        with torch.no_grad():
            for x, y in loader:
                pred.extend(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy().tolist())
                gold.extend(y.numpy().tolist())
        hard = (np.asarray(pred) >= args.threshold).astype(int)
        print(f"heldout={len(gold)} accuracy={accuracy_score(gold, hard):.6f} threshold={args.threshold}")
        return
    if not all((args.val_dir, args.val_csv, args.test_dir, args.test_csv)):
        raise SystemExit("submit mode requires --val-dir, --val-csv, --test-dir, and --test-csv")
    train_data = WeakCollageDataset(records, seed=args.seed, training=True)
    model = train_model(train_data, args.epochs, args.batch_size, args.seed, device, args.pretrained)
    val_records = load_records(args.val_dir, args.val_csv)
    test_records = load_records(args.test_dir, args.test_csv)
    val_probs, test_probs = predict(model, val_records, device, args.batch_size), predict(model, test_records, device, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_csv, test_csv = args.output_dir / "submissionA.csv", args.output_dir / "submissionB.csv"
    pd.DataFrame((val_probs >= args.threshold).astype(int)).to_csv(val_csv, index=False, header=False)
    pd.DataFrame((test_probs >= args.threshold).astype(int)).to_csv(test_csv, index=False, header=False)
    with zipfile.ZipFile(args.output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(val_csv, val_csv.name)
        zf.write(test_csv, test_csv.name)
    print(f"wrote {args.output_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
