"""Counterfactual explanations for the 2025 Polish AI Olympiad credit task.

The function deliberately uses only the supplied classifier, supplied class-1
KDE, and X_explain. It does not read X_train or y_train.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _class_one_log_density(x: torch.Tensor, gen_model) -> torch.Tensor:
    """Vectorized equivalent of the notebook's class-conditional KDE call."""
    kde = gen_model.models["1"]
    train = kde.train_Xs.to(device=x.device, dtype=x.dtype)
    bandwidth = float(kde.kernel.bandwidth)
    n, dimensions = train.shape

    squared_distance = (x[:, None, :] - train[None, :, :]).square().sum(dim=-1)
    log_normalizer = (
        0.5 * dimensions * torch.log(x.new_tensor(2.0 * np.pi))
        + dimensions * torch.log(x.new_tensor(bandwidth))
        + torch.log(x.new_tensor(float(n)))
    )
    log_kernel = -0.5 * squared_distance / (bandwidth * bandwidth)
    return torch.logsumexp(log_kernel - log_normalizer, dim=1)


def your_generate_explanations(
    X_explain: np.ndarray,
    y_explain: np.ndarray,
    disc_model,
    gen_model,
    log_prob_threshold: float,
) -> np.ndarray:
    """Move each rejected point to a nearby class-1 point in a dense region."""
    try:
        device = next(disc_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    x_orig = torch.as_tensor(X_explain, dtype=torch.float32, device=device)
    targets = torch.as_tensor(y_explain, dtype=torch.float32, device=device).reshape(-1)
    x_new = x_orig.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([x_new], lr=0.01)
    threshold = float(log_prob_threshold)

    # The three terms optimize a successful decision, short changes, and
    # density above the supplied realism threshold.
    for _ in range(1000):
        optimizer.zero_grad(set_to_none=True)

        logits = disc_model(x_new).reshape(-1)
        validity_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        distance_loss = (x_new - x_orig).square().sum(dim=1)
        log_probability = _class_one_log_density(x_new, gen_model)
        plausibility_loss = F.relu(threshold - log_probability)

        loss = (
            25.0 * validity_loss
            + 300.0 * distance_loss
            + 500.0 * plausibility_loss
        ).mean()
        loss.backward()
        optimizer.step()

    return x_new.detach().cpu().numpy().astype(X_explain.dtype, copy=False)
