"""Train-only low-rank sigmoid-pair solution for the 2024 pruning task."""

from __future__ import annotations

import pickle

import numpy as np


def apply_rank9_sigmoid_pruning(model, X_train, y_train, slope=1e-4):
    """Fit a rank-9 linear map and encode it in the fixed sigmoid MLP.

    The first layer must be Linear(128, 1024), followed by Sigmoid and
    Linear(1024, 10). Only training arrays are read. Call the notebook's
    `save_parameters(model, "model_parameters.pkl")` after this function, or
    pass `save_path` to write the same name/value mapping directly.
    """
    import torch

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    design = np.column_stack((X, np.ones(len(X), dtype=np.float64)))
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    coefficient = solution[:-1]  # [128, 10]
    intercept = solution[-1]     # [10]
    left, singular, right = np.linalg.svd(coefficient, full_matrices=False)
    rank = 9

    first = model.layers[0]
    last = model.layers[2]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for component in range(rank):
            plus = 2 * component
            minus = plus + 1
            direction = torch.as_tensor(left[:, component] * slope,
                                         dtype=first.weight.dtype,
                                         device=first.weight.device)
            first.weight[plus].copy_(direction)
            first.weight[minus].copy_(-direction)
            output = torch.as_tensor((2.0 / slope) * singular[component]
                                     * right[component],
                                     dtype=last.weight.dtype,
                                     device=last.weight.device)
            last.weight[:, plus].copy_(output)
            last.weight[:, minus].copy_(-output)
        last.bias.copy_(torch.as_tensor(intercept, dtype=last.bias.dtype,
                                        device=last.bias.device))
    return model


def save_parameters(model, file_name="model_parameters.pkl"):
    """Write the parameter mapping expected by the official validator."""
    import torch

    values = {name: value.detach().to("cpu") for name, value in model.named_parameters()}
    with open(file_name, "wb") as stream:
        pickle.dump(values, stream)


def your_pruning_algorithm(model):
    """Notebook-compatible entry point; expects notebook globals X_train/y_train."""
    namespace = globals()
    if "X_train" not in namespace or "y_train" not in namespace:
        raise RuntimeError("X_train and y_train must be loaded from the official train_data arrays")
    model = apply_rank9_sigmoid_pruning(model, namespace["X_train"], namespace["y_train"])
    saver = namespace.get("save_parameters")
    if callable(saver):
        saver(model, "model_parameters.pkl")
    else:
        save_parameters(model, "model_parameters.pkl")
    return model
