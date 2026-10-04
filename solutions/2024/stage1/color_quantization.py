"""Constrained color quantization for the Polish AI Olympiad 2024 task.

The function uses only the current image as its per-image training set. It
returns exactly 37 RGB colors, as required by the official scorer.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans


N_COLORS = 37
SIMPLE_COLORS = np.asarray(
    [
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [0, 255, 255],
        [255, 0, 0],
        [255, 0, 255],
        [255, 255, 0],
        [255, 255, 255],
    ],
    dtype=np.float32,
)


def _simple_color_cost(colors: np.ndarray) -> np.ndarray:
    """Distance of each RGB color to the nearest of the task's eight corners."""
    return cdist(np.asarray(colors, dtype=np.float32), SIMPLE_COLORS).min(axis=1)


def your_quantization_algorithm(
    img: np.ndarray,
    *,
    seed: int = 0,
    max_cost: float = 100.0,
    proximal_shift: float = 10.5,
    iterations: int = 6,
) -> np.ndarray:
    """Return a 37-color image optimized for the task's weighted objective.

    The objective's per-pixel terms are ``2 * squared_error`` and ``42 *
    color_cost``. For a fixed cluster, their proximal update moves the color
    centroid 42/4 = 10.5 RGB units toward its closest simple color. A maximum
    palette-cost cap further trades some MSE for a much smaller ``21 *
    max_color_cost`` term.
    """
    source = np.asarray(img)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape (H, W, 3), got {source.shape}")
    if source.dtype != np.uint8:
        raise TypeError("The official evaluator requires uint8 RGB images")
    if source.size == 0:
        raise ValueError("Image must contain at least one pixel")

    pixels = source.reshape(-1, 3).astype(np.float32)
    pixel_sq = np.sum(pixels * pixels, axis=1, keepdims=True)
    rng = np.random.RandomState(seed)
    sample = pixels[rng.choice(len(pixels), min(25_000, len(pixels)), replace=False)]
    kmeans = MiniBatchKMeans(
        n_clusters=N_COLORS,
        random_state=seed,
        batch_size=4096,
        max_iter=80,
        n_init=1,
        init_size=15_000,
        reassignment_ratio=0.01,
    ).fit(sample)
    centers = kmeans.cluster_centers_.astype(np.float32)

    for _ in range(iterations):
        center_sq = np.sum(centers * centers, axis=1, keepdims=True).T
        squared_distances = np.maximum(0.0, pixel_sq + center_sq - 2.0 * pixels @ centers.T)
        center_cost = _simple_color_cost(centers)
        labels = np.argmin(2.0 * squared_distances + 42.0 * center_cost[None, :], axis=1)

        counts = np.bincount(labels, minlength=N_COLORS).astype(np.float32)
        sums = np.stack(
            [np.bincount(labels, weights=pixels[:, channel], minlength=N_COLORS) for channel in range(3)],
            axis=1,
        )
        nonempty = counts > 0
        means = centers.copy()
        means[nonempty] = sums[nonempty] / counts[nonempty, None]

        nearest = np.argmin(cdist(means, SIMPLE_COLORS), axis=1)
        vertices = SIMPLE_COLORS[nearest]
        displacement = means - vertices
        distance = np.linalg.norm(displacement, axis=1)
        target_distance = np.minimum(np.maximum(distance - proximal_shift, 0.0), max_cost)
        centers = vertices + displacement * np.divide(
            target_distance,
            distance,
            out=np.zeros_like(target_distance),
            where=distance > 0,
        )[:, None]

        # Keep every palette entry active when the vividness prior empties a
        # cluster. The final one-pixel reassignment is only a cardinality fix.
        if not np.all(nonempty):
            current_error = squared_distances[np.arange(len(pixels)), labels]
            reserved: set[int] = set()
            for cluster in np.flatnonzero(~nonempty):
                candidate = np.sum((pixels - centers[cluster]) ** 2, axis=1) - current_error
                if reserved:
                    candidate[np.fromiter(reserved, dtype=np.int64)] = np.inf
                pixel_idx = int(np.argmin(candidate))
                reserved.add(pixel_idx)
                centers[cluster] = pixels[pixel_idx]

    # Compute the final task-specific assignment with the stabilized palette.
    center_sq = np.sum(centers * centers, axis=1, keepdims=True).T
    squared_distances = np.maximum(0.0, pixel_sq + center_sq - 2.0 * pixels @ centers.T)
    center_cost = _simple_color_cost(centers)
    labels = np.argmin(2.0 * squared_distances + 42.0 * center_cost[None, :], axis=1)

    palette = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    seen: set[tuple[int, int, int]] = set()
    for i in range(N_COLORS):
        color = [int(channel) for channel in palette[i]]
        while tuple(color) in seen:
            for channel in range(3):
                if color[channel] < 255:
                    color[channel] += 1
                    break
                color[channel] = 0
        palette[i] = color
        seen.add(tuple(color))

    quantized = palette[labels].reshape(source.shape)
    reserved_pixels: set[int] = set()
    current_error = squared_distances[np.arange(len(pixels)), labels]
    for cluster in range(N_COLORS):
        if np.any(labels == cluster):
            continue
        candidate = np.sum((pixels - palette[cluster].astype(np.float32)) ** 2, axis=1) - current_error
        if reserved_pixels:
            candidate[np.fromiter(reserved_pixels, dtype=np.int64)] = np.inf
        pixel_idx = int(np.argmin(candidate))
        reserved_pixels.add(pixel_idx)
        quantized.reshape(-1, 3)[pixel_idx] = palette[cluster]

    return quantized.astype(np.uint8, copy=False)


def evaluate_directory(data_dir: str | Path) -> tuple[float, list[float]]:
    """Evaluate the task score on a directory of JPEGs without copying data."""
    scores: list[float] = []
    for path in sorted(Path(data_dir).glob("*.jpg")):
        with Image.open(path) as image:
            original = np.asarray(image.convert("RGB").resize((512, 512)), dtype=np.uint8)
        quantized = your_quantization_algorithm(original)
        if len(np.unique(quantized.reshape(-1, 3), axis=0)) != N_COLORS:
            raise RuntimeError(f"{path.name}: expected exactly {N_COLORS} output colors")
        mse = np.mean((original.astype(np.float32) - quantized.astype(np.float32)) ** 2)
        unique_colors, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
        color_cost = _simple_color_cost(unique_colors)
        mean_cost = float(np.sum(color_cost * counts) / counts.sum())
        maximum_cost = float(color_cost.max())
        score = float(2.0 * mse + 21.0 * maximum_cost + 42.0 * mean_cost)
        scores.append(score)
        print(
            f"{path.name}: MSE={mse:.3f}, max_color_cost={maximum_cost:.3f}, "
            f"mean_color_cost={mean_cost:.3f}, score={score:.3f}"
        )
    if not scores:
        raise ValueError(f"No .jpg images found in {data_dir}")
    mean_score = float(np.mean(scores))
    print(f"Mean score: {mean_score:.3f}")
    return mean_score, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Task train_data or valid_data folder")
    args = parser.parse_args()
    evaluate_directory(args.data_dir)
