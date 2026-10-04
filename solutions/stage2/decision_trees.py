"""Decision-tree solution for the Polish 2026 Stage 2 task.

The input consists of many independent two-feature binary classification
problems.  We use the unlabeled test points to estimate a stable whitening
transform, rotate the axes to reduce dependence between the training
projections, then choose the tree pruning strength by repeated stratified CV.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier


def preprocess_data(X_train: np.ndarray, X_test: np.ndarray):
    """Whiten pooled features and choose a low-mutual-information rotation.

    The histogram objective is estimated from the pooled, unlabeled point
    cloud. Both train and test points share the resulting transform. The function keeps
    the required two-column shape and has a safe identity fallback for invalid
    or unexpected inputs.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    if (
        X_train.ndim != 2
        or X_test.ndim != 2
        or X_train.shape[1] != 2
        or X_test.shape[1] != 2
        or len(X_train) < 2
        or len(X_test) == 0
    ):
        return X_train.copy(), X_test.copy()

    pooled = np.concatenate((X_train, X_test), axis=0)
    if not np.isfinite(pooled).all():
        return X_train.copy(), X_test.copy()

    center = pooled.mean(axis=0)
    covariance = np.cov(pooled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max()), 1.0) * 1e-12

    # Symmetric whitening preserves the original coordinate frame while
    # correcting for scale and correlation.
    whitening = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, floor)))
        @ eigenvectors.T
    )
    Z = (pooled - center) @ whitening

    n_train = len(X_train)
    angles = np.deg2rad(np.arange(180, dtype=np.float64))
    cosines = np.cos(angles)
    sines = np.sin(angles)
    first = Z[:, 0, None] * cosines + Z[:, 1, None] * sines
    second = -Z[:, 0, None] * sines + Z[:, 1, None] * cosines

    # Estimate mutual information from 24 equal-frequency bins on each axis.
    # The small pseudocount makes the estimate finite for sparse histograms.
    mutual_information = np.empty(len(angles), dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, 25)
    for j in range(len(angles)):
        edges_1 = np.unique(np.quantile(first[:, j], quantiles))
        edges_2 = np.unique(np.quantile(second[:, j], quantiles))
        if len(edges_1) < 2 or len(edges_2) < 2:
            mutual_information[j] = np.inf
            continue

        counts = np.histogram2d(
            first[:, j], second[:, j], bins=(edges_1, edges_2)
        )[0]
        counts += 1e-6
        joint = counts / counts.sum()
        marginal_1 = joint.sum(axis=1, keepdims=True)
        marginal_2 = joint.sum(axis=0, keepdims=True)
        mutual_information[j] = np.sum(
            joint * np.log(joint / (marginal_1 @ marginal_2))
        )

    best = int(np.argmin(mutual_information))
    c, s = cosines[best], sines[best]
    rotation = np.array([[c, -s], [s, c]])
    Z = Z @ rotation
    return Z[:n_train], Z[n_train:]


def get_decision_tree_hyperparameters(X_train: np.ndarray, y_train: np.ndarray):
    """Select cost-complexity pruning strength by repeated stratified CV."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).ravel()
    classes, counts = np.unique(y_train, return_counts=True)
    fallback_alpha = 0.0015

    if len(classes) < 2:
        return {"ccp_alpha": fallback_alpha, "random_state": 42}

    n_splits = min(4, int(counts.min()))
    if n_splits < 2:
        return {"ccp_alpha": fallback_alpha, "random_state": 42}

    alphas = np.round(0.0015 + 0.00025 * np.arange(13), 5)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=2, random_state=1741
    )
    folds = list(cv.split(X_train, y_train))
    scores = np.zeros((len(alphas), len(folds)), dtype=np.float64)

    for alpha_index, alpha in enumerate(alphas):
        for fold_index, (train_index, valid_index) in enumerate(folds):
            tree = DecisionTreeClassifier(
                ccp_alpha=float(alpha), random_state=42
            )
            tree.fit(X_train[train_index], y_train[train_index])
            scores[alpha_index, fold_index] = tree.score(
                X_train[valid_index], y_train[valid_index]
            )

    mean_scores = scores.mean(axis=1)
    best_score = float(mean_scores.max())

    # The selected global policy uses a prior conditioned on the best CV alpha.
    # Both values are fixed across tasks; the condition depends only on this
    # dataset's training labels and repeated-CV scores.
    best_index = int(np.argmax(mean_scores))
    cv_best_alpha = float(alphas[best_index])
    prior_alpha = 0.0015 if cv_best_alpha <= 0.003 else 0.00375

    # Retain near-best candidates because individual CV estimates are noisy.
    eligible = np.flatnonzero(mean_scores >= best_score - 0.001)
    selected = min(
        eligible, key=lambda index: abs(float(alphas[index]) - prior_alpha)
    )
    return {"ccp_alpha": float(alphas[selected]), "random_state": 42}


def fit_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    """Convenience helper for local use of the two task functions above."""
    params = get_decision_tree_hyperparameters(X_train, y_train)
    return DecisionTreeClassifier(**params).fit(X_train, y_train)
