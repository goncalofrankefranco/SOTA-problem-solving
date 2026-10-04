"""Feature-extractor-only class unlearning for the 2025 OAI task.

This implements the 4-epoch method in the released local reference solution
notebook. The official notebook supplies LeNet, data loaders, and the scorer.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn


def unlearn(model: nn.Module, data: dict, target_class: int) -> nn.Module:
    """Unlearn one Fashion-MNIST class without changing the final layer.

    The full training loader is split in each minibatch: retain examples use
    cross-entropy, while forget examples are pushed toward the uniform
    distribution over the other nine classes. An L2 anchor keeps the feature
    extractor close to its initial weights.
    """
    updated = deepcopy(model)
    device = next(updated.parameters()).device
    for parameter in updated.fc2.parameters():
        parameter.requires_grad_(False)

    updated.train()
    initial = {
        name: tensor.detach().clone()
        for name, tensor in updated.state_dict().items()
    }
    loader = data["fashion"]["loader"]
    optimizer = torch.optim.AdamW(
        (p for p in updated.parameters() if p.requires_grad), lr=1e-4
    )
    cross_entropy = nn.CrossEntropyLoss()

    for _ in range(4):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = updated(images)
            forget = labels == int(target_class)
            retain = ~forget

            loss = logits.new_zeros(())
            if retain.any():
                loss = loss + cross_entropy(logits[retain], labels[retain])
            if forget.any():
                target_distribution = torch.full_like(
                    logits[forget], 1.0 / (logits.shape[1] - 1)
                )
                target_distribution[:, int(target_class)] = 0.0
                loss = loss + F.kl_div(
                    F.log_softmax(logits[forget], dim=1),
                    target_distribution,
                    reduction="batchmean",
                )

            l2_penalty = logits.new_zeros(())
            for name, parameter in updated.named_parameters():
                if parameter.requires_grad:
                    l2_penalty = l2_penalty + torch.norm(
                        parameter - initial[name].to(device)
                    ).square()
            loss = loss + l2_penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return updated
