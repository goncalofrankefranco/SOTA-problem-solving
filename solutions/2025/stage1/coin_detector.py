"""Train-from-scratch multi-scale coin detector for Poland 2025 Stage I.

The official notebook provides a 64px sliding-window classifier but does not
include a worked solution or a measured validation result.  This candidate
adds box-offset regression and multi-scale proposals.  Its validation score is
unverified because the official train/validation pickles are hosted on Google
Drive, which is blocked in this environment.

Notebook integration after the official ``setup_data`` call::

    from coin_detector import CoinDetector
    your_model = CoinDetector().fit(train_ds, device=DEVICE)

Only the official training dataset is used by ``fit``.  No pretrained weights
or external image data are used.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.ops import nms


_BACKGROUND = 9
_PATCH_SIZE = 64


def _clip_window(cx: float, cy: float, size: int, width: int, height: int) -> tuple[int, int, int, int]:
    size = max(8, min(int(size), width, height))
    x1 = int(round(cx - size / 2.0))
    y1 = int(round(cy - size / 2.0))
    x1 = min(max(0, x1), max(0, width - size))
    y1 = min(max(0, y1), max(0, height - size))
    return x1, y1, x1 + size, y1 + size


def _box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=box.dtype, device=box.device)
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_b = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter).clamp(min=1e-6)


def _crop_tensor(image: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    x1, y1, x2, y2 = box
    crop = image[:, y1:y2, x1:x2]
    return F.interpolate(
        crop.unsqueeze(0).float(),
        size=(_PATCH_SIZE, _PATCH_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0]


class _PatchSampler(Dataset):
    """Generate jittered positive crops and random background crops on demand."""

    def __init__(self, source: Dataset, base_size: float, crops_per_image: int = 64):
        self.source = source
        self.base_size = float(base_size)
        self.crops_per_image = int(crops_per_image)

    def __len__(self) -> int:
        return len(self.source) * self.crops_per_image

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.source[index // self.crops_per_image]
        image = torch.as_tensor(sample["image"]).float()
        boxes = torch.as_tensor(sample["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(sample["labels"], dtype=torch.long)
        _, height, width = image.shape

        use_background = len(boxes) == 0 or random.random() < 0.15
        if use_background:
            size = max(16, round(self.base_size * random.uniform(0.75, 1.25)))
            cx = random.uniform(size / 2.0, max(size / 2.0, width - size / 2.0))
            cy = random.uniform(size / 2.0, max(size / 2.0, height - size / 2.0))
        else:
            target_index = random.randrange(len(boxes))
            target = boxes[target_index]
            target_w = max(8.0, float(target[2] - target[0]))
            target_h = max(8.0, float(target[3] - target[1]))
            size = max(16, round(0.5 * (target_w + target_h) * random.uniform(0.78, 1.22)))
            jitter = 0.22 * size
            cx = float((target[0] + target[2]) / 2.0) + random.uniform(-jitter, jitter)
            cy = float((target[1] + target[3]) / 2.0) + random.uniform(-jitter, jitter)

        window = _clip_window(cx, cy, size, width, height)
        x1, y1, x2, y2 = window
        anchor = torch.tensor(window, dtype=torch.float32)
        ious = _box_iou_one_to_many(anchor, boxes)
        if ious.numel() and float(ious.max()) >= 0.35:
            matched = int(ious.argmax())
            gt = boxes[matched]
            label = labels[matched]
            aw, ah = x2 - x1, y2 - y1
            acx, acy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            gcx, gcy = float((gt[0] + gt[2]) / 2.0), float((gt[1] + gt[3]) / 2.0)
            target_delta = torch.tensor(
                [
                    (gcx - acx) / aw,
                    (gcy - acy) / ah,
                    math.log(max(1.0, float(gt[2] - gt[0])) / aw),
                    math.log(max(1.0, float(gt[3] - gt[1])) / ah),
                ],
                dtype=torch.float32,
            )
        else:
            label = torch.tensor(_BACKGROUND, dtype=torch.long)
            target_delta = torch.zeros(4, dtype=torch.float32)

        return {
            "crop": _crop_tensor(image, window),
            "label": torch.as_tensor(label, dtype=torch.long),
            "delta": target_delta,
            "positive": torch.as_tensor(int(label) != _BACKGROUND, dtype=torch.bool),
        }


class _PatchNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(512, 10)
        self.box_regressor = nn.Linear(512, 4)

    def forward(self, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(crops)
        return self.classifier(features), self.box_regressor(features)


class CoinDetector(nn.Module):
    """ResNet-18 patch classifier plus box regression and class-wise NMS."""

    def __init__(self, anchor_size: float = 64.0, score_threshold: float = 0.20):
        super().__init__()
        self.network = _PatchNetwork()
        self.anchor_size = float(anchor_size)
        self.score_threshold = float(score_threshold)
        self.max_detections = 100

    def fit(
        self,
        train_dataset: Dataset,
        *,
        device: torch.device | str | None = None,
        epochs: int = 15,
        crops_per_image: int = 64,
        batch_size: int = 128,
    ) -> "CoinDetector":
        """Train using only the supplied official training dataset."""
        widths: list[float] = []
        for index in range(len(train_dataset)):
            sample = train_dataset[index]
            boxes = torch.as_tensor(sample["boxes"], dtype=torch.float32)
            if boxes.numel():
                widths.extend(((boxes[:, 2] - boxes[:, 0]).clamp(min=8)).tolist())
        if widths:
            self.anchor_size = float(np.median(widths))

        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        self.to(target_device)
        patch_dataset = _PatchSampler(train_dataset, self.anchor_size, crops_per_image)
        loader = DataLoader(
            patch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=target_device.type == "cuda",
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        class_loss = nn.CrossEntropyLoss()
        box_loss = nn.SmoothL1Loss(reduction="none")

        for _ in range(int(epochs)):
            self.train()
            for batch in loader:
                crops = batch["crop"].to(target_device, non_blocking=True)
                labels = batch["label"].to(target_device, non_blocking=True)
                deltas = batch["delta"].to(target_device, non_blocking=True)
                positive = batch["positive"].to(target_device, non_blocking=True)
                logits, predicted_deltas = self.network(crops)
                loss = class_loss(logits, labels)
                if positive.any():
                    regression = box_loss(predicted_deltas[positive], deltas[positive]).mean()
                    loss = loss + 2.0 * regression
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self.eval()
        return self

    @staticmethod
    def _grid_starts(length: int, size: int, stride: int) -> list[int]:
        if length <= size:
            return [0]
        starts = list(range(0, length - size + 1, stride))
        if starts[-1] != length - size:
            starts.append(length - size)
        return starts

    def _proposals(self, height: int, width: int) -> list[tuple[int, int, int, int]]:
        sizes = sorted(
            {
                max(16, round(self.anchor_size * scale))
                for scale in (0.75, 0.85, 0.95, 1.05, 1.15, 1.25)
            }
        )
        proposals: list[tuple[int, int, int, int]] = []
        for size in sizes:
            size = min(size, width, height)
            stride = max(8, round(size / 4.0))
            xs = self._grid_starts(width, size, stride)
            ys = self._grid_starts(height, size, stride)
            proposals.extend((x, y, x + size, y + size) for y in ys for x in xs)
        return proposals

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> list[tuple[float, float, float, float, int, float]]:
        device = next(self.parameters()).device
        image = torch.as_tensor(image, dtype=torch.float32, device=device)
        if image.ndim != 3:
            raise ValueError(f"expected image with shape (channels, height, width), got {tuple(image.shape)}")
        _, height, width = image.shape
        proposals = self._proposals(height, width)
        if not proposals:
            return []

        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for start in range(0, len(proposals), 256):
            batch_boxes = proposals[start : start + 256]
            crops = torch.stack([_crop_tensor(image, box) for box in batch_boxes])
            logits, deltas = self.network(crops)
            probabilities = logits.softmax(dim=1)[:, :_BACKGROUND]
            scores, labels = probabilities.max(dim=1)
            keep = scores >= self.score_threshold
            if not keep.any():
                continue

            anchors = torch.tensor(batch_boxes, dtype=torch.float32, device=device)[keep]
            delta = deltas[keep].clamp(min=-0.5, max=0.5)
            aw = anchors[:, 2] - anchors[:, 0]
            ah = anchors[:, 3] - anchors[:, 1]
            acx = (anchors[:, 0] + anchors[:, 2]) / 2.0
            acy = (anchors[:, 1] + anchors[:, 3]) / 2.0
            pcx = acx + delta[:, 0] * aw
            pcy = acy + delta[:, 1] * ah
            pw = aw * delta[:, 2].exp()
            ph = ah * delta[:, 3].exp()
            boxes = torch.stack((pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2), dim=1)
            boxes[:, 0::2].clamp_(0, width)
            boxes[:, 1::2].clamp_(0, height)
            all_boxes.append(boxes)
            all_scores.append(scores[keep])
            all_labels.append(labels[keep])

        if not all_boxes:
            return []
        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)

        kept: list[torch.Tensor] = []
        for label in labels.unique():
            mask = labels == label
            class_indices = torch.where(mask)[0]
            class_keep = nms(boxes[mask], scores[mask], iou_threshold=0.45)
            kept.append(class_indices[class_keep])
        selected = torch.cat(kept)
        selected = selected[scores[selected].argsort(descending=True)[: self.max_detections]]

        result = []
        for i in selected.tolist():
            x1, y1, x2, y2 = boxes[i].tolist()
            result.append((x1, y1, x2, y2, int(labels[i]), float(scores[i])))
        return result


__all__ = ["CoinDetector"]
