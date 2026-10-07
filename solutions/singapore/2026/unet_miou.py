"""Reference implementation for NOAI Singapore 2026 Programming Task 2."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BrokenUNet(nn.Module):
    """Released padding-free U-Net repaired with center-cropped skip features."""

    def __init__(self, in_channels: int = 3, out_classes: int = 2) -> None:
        super().__init__()
        self.enc1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=0)
        self.enc1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.enc2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(2)

        self.bot1 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.bot2 = nn.Conv2d(256, 256, kernel_size=3, padding=0)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1_1 = nn.Conv2d(256, 128, kernel_size=3, padding=0)
        self.dec1_2 = nn.Conv2d(128, 64, kernel_size=3, padding=0)
        self.final = nn.Conv2d(64, out_classes, kernel_size=1)

    @staticmethod
    def center_crop(layer: torch.Tensor, target_shape) -> torch.Tensor:
        """Center-crop a BCHW skip tensor to the target's spatial dimensions."""
        target_h, target_w = int(target_shape[-2]), int(target_shape[-1])
        height, width = layer.shape[-2:]
        if target_h > height or target_w > width:
            raise ValueError(
                f"target spatial shape {(target_h, target_w)} exceeds source {(height, width)}"
            )
        top = (height - target_h) // 2
        left = (width - target_w) // 2
        return layer[..., top : top + target_h, left : left + target_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.enc1_1(x))
        x1 = F.relu(self.enc1_2(x1))
        x1_p = self.pool1(x1)

        x2 = F.relu(self.enc2_1(x1_p))
        x2 = F.relu(self.enc2_2(x2))
        x2_p = self.pool2(x2)

        bot = F.relu(self.bot1(x2_p))
        bot = F.relu(self.bot2(bot))

        d1 = self.upconv1(bot)
        x2_cropped = self.center_crop(x2, d1.shape)
        cat1 = torch.cat((x2_cropped, d1), dim=1)

        dec1 = F.relu(self.dec1_1(cat1))
        dec1 = F.relu(self.dec1_2(dec1))
        return self.final(dec1)


def calculate_miou(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    num_classes: int = 2,
    *,
    skip_classes_absent_from_truth: bool = False,
) -> float:
    """Mean class IoU, skipping empty unions, per task definition.

    ``skip_classes_absent_from_truth=True`` reproduces the released notebook's
    contradictory self-check, which skips a class whenever it is absent from
    the target even when it has false-positive predictions. That convention is
    not the written union-based IoU, so the default remains the formal metric.
    """
    if pred_mask.shape != true_mask.shape:
        raise ValueError("pred_mask and true_mask must have identical shapes")
    pred_flat = pred_mask.reshape(-1)
    true_flat = true_mask.reshape(-1)
    ious: list[float] = []

    for cls in range(num_classes):
        pred_cls = pred_flat == cls
        true_cls = true_flat == cls
        if skip_classes_absent_from_truth and not bool(true_cls.any()):
            continue
        intersection = (pred_cls & true_cls).sum().item()
        union = (pred_cls | true_cls).sum().item()
        if union:
            ious.append(intersection / union)

    return float(sum(ious) / len(ious)) if ious else 0.0
