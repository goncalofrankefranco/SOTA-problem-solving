"""Train a one-channel ResNet detector for bona-fide vs spoofed speech.

Input directories follow the organizer layout: training_set/{bonafide,spoof}
for labeled data, and flat validation_set/testing_set directories for inference.
No organizer data or model weights are downloaded or copied by this script.
"""
from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18


EXTENSIONS = {".pt", ".pth"}


def numeric_key(path: Path):
    import re
    nums = re.findall(r"\d+", path.name)
    return (int(nums[-1]) if nums else -1, path.name)


def files_in(path: Path) -> list[Path]:
    return sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=numeric_key)


class Spectrograms(Dataset):
    def __init__(self, root: Path, labeled: bool):
        self.root, self.labeled = root, labeled
        if labeled:
            self.items = []
            for label, folder in ((0, "bonafide"), (1, "spoof")):
                self.items.extend((p, label) for p in files_in(root / folder))
            self.items.sort(key=lambda z: (z[1], numeric_key(z[0])))
        else:
            self.items = [(p, -1) for p in files_in(root)]
        if not self.items:
            raise ValueError(f"No .pt spectrograms found under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        try:
            x = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            x = torch.load(path, map_location="cpu")
        if isinstance(x, dict):
            x = x.get("spectrogram", x.get("x"))
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected [1, mel, time] tensor in {path}, got {tuple(x.shape)}")
        x = torch.nan_to_num(x).clamp(-120.0, 60.0) / 50.0
        return x, torch.tensor(label, dtype=torch.long), path.name


def make_model(pretrained: bool = False) -> nn.Module:
    # If the official machine already has torchvision weights cached, they can
    # be enabled with --pretrained. Default is offline and self-contained.
    from torchvision.models import ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    conv = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                     stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            conv.weight.copy_(model.conv1.weight.mean(dim=1, keepdim=True))
    model.conv1 = conv
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def spec_augment(x: torch.Tensor, max_freq: int = 12, max_time: int = 18) -> torch.Tensor:
    """Small SpecAugment masks; applied on the training tensor only."""
    out = x.clone()
    _, _, f, t = out.shape
    for i in range(len(out)):
        if random.random() < 0.7:
            width = random.randint(1, max_freq)
            start = random.randint(0, max(0, f - width))
            out[i, :, start:start + width, :] = 0
        if random.random() < 0.7:
            width = random.randint(1, max_time)
            start = random.randint(0, max(0, t - width))
            out[i, :, :, start:start + width] = 0
    return out


def predict(model, loader, device):
    model.eval()
    probs, labels, names = [], [], []
    with torch.no_grad():
        for x, y, name in loader:
            logits = model(x.to(device))
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
            names.extend(name)
    return np.asarray(probs), np.asarray(labels), names


def train_model(dataset: Spectrograms, train_indices: list[int], val_indices: list[int], epochs: int,
                batch_size: int, seed: int, pretrained: bool, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=device.type == "cuda")
    model = make_model(pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_state, best_f1, best_epoch = None, -1.0, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            x = spec_augment(x)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach())
        probs, gold, _ = predict(model, val_loader, device)
        pred = (probs >= 0.5).astype(np.int64)
        score = f1_score(gold, pred, average="macro", zero_division=0)
        print(f"epoch={epoch + 1} loss={total_loss / max(1, len(train_loader)):.5f} val_macro_f1={score:.5f}")
        if score > best_f1:
            best_f1, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def write_predictions(probs: np.ndarray, path: Path, threshold: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame((probs >= threshold).astype(np.int64)).to_csv(path, index=False, header=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "submit"), default="validate")
    parser.add_argument("--train-dir", type=Path, required=True, help="training_set with bonafide/ and spoof/")
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("submission"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pretrained", action="store_true", help="Use cached torchvision ResNet18 weights; no download is attempted intentionally")
    args = parser.parse_args()
    dataset = Spectrograms(args.train_dir, labeled=True)
    y = np.asarray([label for _, label in dataset.items], dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "validate":
        rng = np.random.default_rng(args.seed)
        train_ix, val_ix = [], []
        for label in (0, 1):
            ix = np.flatnonzero(y == label)
            rng.shuffle(ix)
            cut = max(1, int(0.2 * len(ix)))
            val_ix.extend(ix[:cut].tolist())
            train_ix.extend(ix[cut:].tolist())
        model, best_epoch = train_model(dataset, train_ix, val_ix, args.epochs, args.batch_size, args.seed, args.pretrained, device)
        loader = DataLoader(Subset(dataset, val_ix), batch_size=args.batch_size, shuffle=False, num_workers=2)
        probs, gold, _ = predict(model, loader, device)
        pred = (probs >= args.threshold).astype(np.int64)
        print(f"heldout={len(gold)} best_epoch={best_epoch} macro_f1={f1_score(gold, pred, average='macro', zero_division=0):.6f} positive_f1={f1_score(gold, pred, average='binary', zero_division=0):.6f} threshold={args.threshold}")
        return
    if args.val_dir is None or args.test_dir is None:
        raise SystemExit("submit mode requires --val-dir and --test-dir")
    # Keep 10% for early stopping, then use the full labeled set for the final
    # fit with the best epoch count. This avoids using hidden labels.
    rng = np.random.default_rng(args.seed)
    train_ix, early_ix = [], []
    for label in (0, 1):
        ix = np.flatnonzero(y == label)
        rng.shuffle(ix)
        cut = max(1, int(0.1 * len(ix)))
        early_ix.extend(ix[:cut].tolist())
        train_ix.extend(ix[cut:].tolist())
    _, best_epoch = train_model(dataset, train_ix, early_ix, args.epochs, args.batch_size, args.seed, args.pretrained, device)
    # Refit with all labeled files for that number of epochs.
    all_ix = list(range(len(dataset)))
    final, _ = train_model(dataset, all_ix, early_ix, best_epoch, args.batch_size, args.seed + 1, args.pretrained, device)
    val_set, test_set = Spectrograms(args.val_dir, labeled=False), Spectrograms(args.test_dir, labeled=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_probs, _, _ = predict(final, val_loader, device)
    test_probs, _, _ = predict(final, test_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_csv, test_csv = args.output_dir / "submissionA.csv", args.output_dir / "submissionB.csv"
    write_predictions(val_probs, val_csv, args.threshold)
    write_predictions(test_probs, test_csv, args.threshold)
    with zipfile.ZipFile(args.output_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(val_csv, val_csv.name)
        zf.write(test_csv, test_csv.name)
    print(f"wrote {args.output_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
