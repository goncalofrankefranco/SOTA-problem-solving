"""Train a constraint-compliant classifier for the NOAI China 2024 task 1.

The network receives only ``loc_x`` and ``loc_y``. Run with ``--mode validate``
to measure stratified cross-validation accuracy, or ``--mode train`` to fit all
available labeled rows and save a checkpoint for inference.
"""

from __future__ import annotations

import argparse
import random
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn


class MyModel(nn.Module):
    """Three linear layers, no more than eight neurons per layer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"loc_x", "loc_y", "shot_made_flag"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    frame = frame.dropna(subset=["loc_x", "loc_y", "shot_made_flag"])
    x = frame[["loc_x", "loc_y"]].to_numpy(dtype=np.float32)
    y = frame["shot_made_flag"].to_numpy(dtype=np.float32)
    return x, y


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 250,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-3,
) -> MyModel:
    seed_everything(seed)
    scaler = StandardScaler().fit(x)
    x_tensor = torch.as_tensor(scaler.transform(x), dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    model = MyModel()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    # Fold the fitted standardization into the first affine layer so the saved
    # submission model itself still accepts only the two raw allowed features.
    with torch.no_grad():
        original_weight = model.fc1.weight.detach().clone()
        scale = torch.as_tensor(scaler.scale_, dtype=original_weight.dtype)
        mean = torch.as_tensor(scaler.mean_, dtype=original_weight.dtype)
        model.fc1.weight.copy_(original_weight / scale.unsqueeze(0))
        model.fc1.bias.sub_(original_weight @ (mean / scale))
    return model.eval()


def validate(path: Path, folds: int, seed: int, epochs: int) -> None:
    x, y = load_xy(path)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_logits = np.empty(len(y), dtype=np.float32)
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        model = fit_model(
            x[train_idx], y[train_idx], seed=seed + fold, epochs=epochs
        )
        valid_x = torch.as_tensor(x[valid_idx], dtype=torch.float32)
        with torch.inference_mode():
            logits = model(valid_x).cpu().numpy()
        oof_logits[valid_idx] = logits
        fold_rows.append(
            (fold, accuracy_score(y[valid_idx], logits >= 0), roc_auc_score(y[valid_idx], logits))
        )
    for fold, accuracy, auc in fold_rows:
        print(f"fold={fold} accuracy={accuracy:.4f} roc_auc={auc:.4f}")
    print(f"oof_accuracy={accuracy_score(y, oof_logits >= 0):.4f}")
    print(f"oof_roc_auc={roc_auc_score(y, oof_logits):.4f}")
    print(f"rows={len(y)} positive_rate={y.mean():.4f}")


def submission_model_source() -> str:
    """Return the standalone model definition expected inside submission.zip."""
    return '''import torch\nfrom torch import nn\n\n\nclass MyModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(2, 8)\n        self.fc2 = nn.Linear(8, 8)\n        self.fc3 = nn.Linear(8, 1)\n        self.activation = nn.Tanh()\n\n    def forward(self, x):\n        x = self.activation(self.fc1(x))\n        x = self.activation(self.fc2(x))\n        return self.fc3(x).squeeze(-1)\n'''


def write_submission_package(model: MyModel, output: Path) -> None:
    """Write the two required submission files at the root of a zip archive."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="basketball-submission-") as temp_dir:
        temp = Path(temp_dir)
        model_file = temp / "submission_model.py"
        weights_file = temp / "submission_dic.pth"
        model_file.write_text(submission_model_source(), encoding="utf-8")
        torch.save(model.state_dict(), weights_file)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(model_file, arcname="submission_model.py")
            archive.write(weights_file, arcname="submission_dic.pth")


def train_all(path: Path, output: Path, seed: int, epochs: int, threshold: float) -> None:
    x, y = load_xy(path)
    model = fit_model(x, y, seed=seed, epochs=epochs)
    # The task grader applies a 0.5 sigmoid threshold, equivalent to logit 0.
    # Shift the output bias to implement a threshold measured from OOF logits.
    with torch.no_grad():
        model.fc3.bias.sub_(threshold)
    write_submission_package(model, output)
    print(f"saved {output} (submission_model.py, submission_dic.pth; threshold={threshold:g})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["validate", "train"], default="validate")
    parser.add_argument("--data", type=Path, required=True, help="CSV with loc_x, loc_y, shot_made_flag")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.27)
    parser.add_argument("--output", type=Path, default=Path("submission.zip"), help="Output submission.zip")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "validate":
        validate(args.data, args.folds, args.seed, args.epochs)
    else:
        train_all(args.data, args.output, args.seed, args.epochs, args.threshold)


if __name__ == "__main__":
    main()
