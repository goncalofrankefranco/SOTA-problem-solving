"""Coordinate-only tracking and color-based tracking for the 2024 Polish AI Olympiad."""

from __future__ import annotations

from itertools import permutations
import re

import numpy as np


def _center_detections(coordinates):
    """Return a chronological list of (center, box_size) detections per frame."""
    if not coordinates:
        return []

    def frame_number(item):
        match = re.search(r"(\d+)", str(item[0]))
        return int(match.group(1)) if match else 0

    frames = []
    for _, boxes in sorted(coordinates.items(), key=frame_number):
        detections = []
        for box in boxes or []:
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = map(float, box[:4])
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append((np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                               np.array([x2 - x1, y2 - y1])))
        frames.append(detections)
    return frames


def _robust_velocity(history, window=4):
    """Coordinate-wise median of recent observed velocity vectors."""
    recent = history[-(window + 1):]
    velocities = []
    for (t0, p0, _), (t1, p1, _) in zip(recent[:-1], recent[1:]):
        if t1 > t0:
            velocities.append((p1 - p0) / (t1 - t0))
    if not velocities:
        return np.zeros(2, dtype=float)
    velocity = np.median(np.stack(velocities), axis=0)
    # Suppress detector jitter when a cup is stationary.
    if np.linalg.norm(velocity) < 2.0:
        return np.zeros(2, dtype=float)
    speed = np.linalg.norm(velocity)
    if speed > 40.0:
        velocity = velocity * (40.0 / speed)
    return velocity


def _track_centers(frames):
    """Track up to three centers using a global assignment at each frame."""
    start = next((i for i, detections in enumerate(frames) if len(detections) >= 3), None)
    if start is None:
        return [0, 1, 2]

    initial = sorted(frames[start], key=lambda detection: detection[0][0])[:3]
    tracks = [[(start, center.copy(), size.copy())] for center, size in initial]

    for t in range(start + 1, len(frames)):
        detections = frames[t]
        # The task contains at most three objects. Ignore extra components/boxes
        # by retaining the three largest, if a detector ever returns more.
        if len(detections) > 3:
            detections = sorted(detections,
                                key=lambda detection: detection[1][0] * detection[1][1],
                                reverse=True)[:3]
        m = len(detections)
        if m == 0:
            continue

        predictions = []
        for history in tracks:
            last_t, last_center, _ = history[-1]
            velocity = _robust_velocity(history)
            gap = min(t - last_t, 8)
            predictions.append(last_center + velocity * gap)

        best_mapping = None
        best_cost = float("inf")
        # The tuple maps detections (in input order) to distinct track IDs.
        for mapping in permutations(range(3), m):
            cost = sum(float(np.sum((detections[j][0] - predictions[track_id]) ** 2))
                       for j, track_id in enumerate(mapping))
            if cost < best_cost:
                best_cost = cost
                best_mapping = mapping
        for j, track_id in enumerate(best_mapping):
            center, size = detections[j]
            tracks[track_id].append((t, center.copy(), size.copy()))

    final_x = [history[-1][1][0] for history in tracks]
    # The problem's answer lists the original cup IDs occupying final slots
    # from left to right.
    return [int(i) for i in np.argsort(final_x)]


def your_algorithm_task_1(coordinates):
    """Return final cup order from complete level-1 bounding-box detections."""
    return _track_centers(_center_detections(coordinates))


def your_algorithm_task_2(coordinates):
    """Return final cup order, bridging level-2 missing detections robustly."""
    return _track_centers(_center_detections(coordinates))


def _image_components(images):
    """Extract red-cup connected components and filter merged regions."""
    from scipy.ndimage import label

    raw_frames = []
    for image in images:
        array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        if array.ndim != 3 or array.shape[-1] < 3:
            raw_frames.append([])
            continue
        red, green, blue = array[..., 0], array[..., 1], array[..., 2]
        mask = ((red > 90) & (red > 1.35 * green) & (red > 1.35 * blue)
                & (green < 100) & (blue < 100))
        labels, count = label(mask)
        sizes = np.bincount(labels.ravel())
        detections = []
        for label_id in range(1, count + 1):
            area = int(sizes[label_id])
            if area < 1000:
                continue
            ys, xs = np.where(labels == label_id)
            center = np.array([xs.mean(), ys.mean()], dtype=float)
            box_size = np.array([xs.max() - xs.min() + 1,
                                 ys.max() - ys.min() + 1], dtype=float)
            detections.append((center, box_size, area))
        raw_frames.append(detections)

    complete_areas = [d[2] for frame in raw_frames if len(frame) == 3 for d in frame]
    complete_y = [d[0][1] for frame in raw_frames if len(frame) == 3 for d in frame]
    if len(complete_areas) >= 3:
        y_mean = float(np.mean(complete_y))
        y_scale = float(np.std(complete_y)) or 1.0
        coefficients = np.polyfit((np.asarray(complete_y) - y_mean) / y_scale,
                                  np.log(np.asarray(complete_areas)), 2)

        def expected_area(y):
            z = (y - y_mean) / y_scale
            return float(np.exp(np.polyval(coefficients, z)))
    else:
        reference = float(np.median(complete_areas)) if complete_areas else 0.0

        def expected_area(_y):
            return reference

    frames = []
    for frame in raw_frames:
        detections = []
        for center, box_size, area in frame:
            if len(frame) < 3:
                expected = expected_area(center[1])
                # An oversized connected region is likely two or more cups
                # joined by occlusion; leave it out rather than relabeling it.
                if expected > 0 and area / expected > 1.3:
                    continue
            detections.append((center, box_size))
        frames.append(detections)
    return frames


def your_algorithm_task_3(images):
    """Detect red cup components, track their centers, and return final order."""
    return _track_centers(_image_components(images))
