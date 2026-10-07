"""NOAI Singapore 2025, Section 2 Question 2.

K-means with caller-supplied initial centres, deterministic nearest-centre
assignment, empty-cluster retention, and four-decimal output.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean distance between two coordinate vectors."""
    a_array = np.asarray(a, dtype=float)
    b_array = np.asarray(b, dtype=float)
    if a_array.shape != b_array.shape:
        raise ValueError("a and b must have the same shape")
    return float(np.linalg.norm(a_array - b_array))


def k_means_clustering(
    points: Iterable[Sequence[float]],
    k: int,
    initial_centroids: Iterable[Sequence[float]],
    max_iterations: int,
) -> list[tuple[float, ...]]:
    """Return k-means centres, rounded to four decimals.

    Empty clusters keep their previous centre. Ties go to the first centre,
    matching ``argmin``'s deterministic behavior and the released example.
    Centres remain full precision during iteration; only the returned answer
    is rounded.
    """
    data = np.asarray(list(points), dtype=float)
    centres = np.asarray(list(initial_centroids), dtype=float)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("points must be a non-empty collection of vectors")
    if centres.ndim != 2 or centres.shape[0] != k or k <= 0:
        raise ValueError("initial_centroids must contain exactly k vectors")
    if data.shape[1] != centres.shape[1]:
        raise ValueError("points and centroids must have matching dimensions")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")

    for _ in range(max_iterations):
        distances_squared = np.sum(
            (data[:, np.newaxis, :] - centres[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        assignments = np.argmin(distances_squared, axis=1)

        updated = centres.copy()
        for cluster in range(k):
            members = data[assignments == cluster]
            if members.size:
                updated[cluster] = members.mean(axis=0)

        if np.allclose(updated, centres, rtol=0.0, atol=1e-12):
            centres = updated
            break
        centres = updated

    rounded = np.round(centres, 4)
    return [tuple(float(value) for value in row) for row in rounded]


if __name__ == "__main__":
    points = [(2, 4), (2, 8), (2, 10), (10, 2), (10, 4), (10, 0)]
    initial_centroids = [(2, 2), (10, 1)]
    print(k_means_clustering(points, 2, initial_centroids, 10))
