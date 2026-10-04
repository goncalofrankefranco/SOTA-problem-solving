"""PGD attack for the 2024 Polish AI Olympiad Adversarial Attacks task.

The official notebook supplies ``trained_model.pth`` and calls a one-argument
``perturbe_dataset`` function.  The standalone helper also accepts an explicit
model so it can be measured without changing the notebook.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """The exact classifier architecture defined by the organizers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.log_softmax(self.fc2(x), dim=1)


def _load_official_model(
    checkpoint: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> tuple[Net, torch.device]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint) if checkpoint is not None else Path("trained_model.pth")
    model = Net()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, device


def pgd_perturb_dataset(
    original_dataset: np.ndarray,
    model: nn.Module,
    device: Optional[torch.device | str] = None,
    *,
    epsilon: float = 0.299,
    steps: int = 3,
    step_size: float = 0.1,
    batch_size: int = 512,
) -> np.ndarray:
    """Maximize untargeted loss with a projected sign-gradient attack.

    The organizer normalizes inputs independently to [-1, 1].  Output floats
    remain in that same range and are not passed through another normalization.
    ``epsilon`` is kept just below the strict 0.3 pixel-distance limit to avoid
    floating point boundary overshoot in the evaluator.
    """
    images = np.asarray(original_dataset, dtype=np.float32)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Expected (N, 28, 28) images, got {images.shape}")
    if not np.isfinite(images).all():
        raise ValueError("Input images must contain only finite values")

    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.enable_grad():
        for start in range(0, len(images), batch_size):
            clean = torch.from_numpy(images[start : start + batch_size, None]).to(target_device)
            with torch.no_grad():
                # The task interface does not provide ground-truth labels.
                pseudo_labels = model(clean).argmax(dim=1)
            adversarial = clean.detach().clone()
            for _ in range(steps):
                adversarial.requires_grad_(True)
                log_probs = model(adversarial)
                loss = F.nll_loss(log_probs, pseudo_labels)
                gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
                adversarial = adversarial.detach() + step_size * gradient.sign()
                delta = torch.clamp(adversarial - clean, min=-epsilon, max=epsilon)
                adversarial = torch.clamp(clean + delta, min=-1.0, max=1.0).detach()
            outputs.append(adversarial[:, 0].cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(outputs, axis=0) if outputs else images.copy()


def perturbe_dataset(
    original_dataset: np.ndarray,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device | str] = None,
    checkpoint: Optional[str | Path] = None,
) -> np.ndarray:
    """Notebook-compatible entry point; loads the official checkpoint if needed."""
    if model is None:
        model, loaded_device = _load_official_model(checkpoint, device)
        device = loaded_device
    return pgd_perturb_dataset(original_dataset, model, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/contest_validation_samples.npy")
    parser.add_argument("--checkpoint", default="trained_model.pth")
    parser.add_argument("--output", default="adversarial_validation.npy")
    args = parser.parse_args()
    raw = np.load(args.data).astype(np.float32) / 255.0
    flat = raw.reshape(len(raw), -1)
    lo = flat.min(axis=1, keepdims=True)
    hi = flat.max(axis=1, keepdims=True)
    normalized = ((2.0 * (flat - lo) / (hi - lo)) - 1.0).reshape(raw.shape)
    model, device = _load_official_model(args.checkpoint)
    adversarial = pgd_perturb_dataset(normalized, model, device)
    np.save(args.output, adversarial)
    print(f"Saved {len(adversarial)} adversarial images to {args.output}")
