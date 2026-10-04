# Object Tracking 1 — Three Cups

## Task and metric

- **Domain:** computer vision, object tracking from detections.
- **Metric:** exact permutation accuracy; this subtask is worth up to 0.5 points.
- **Abridged statement:** given bounding boxes for three cups in each frame of a shuffle animation, return the initial cup IDs in final left-to-right order. The task explicitly forbids using the images in this subtask; only box coordinates may be used.

## EDA

The official helper loads frames in JSON insertion order. Each example has 50 frames and exactly three boxes per frame. Box-list order changes between frames, so list indices are not object identities. Box centers are sufficient for tracking; image appearance is outside the permitted input.

## Experiments and result

I tested the official `example_data` videos (two labeled videos, not the hidden validation set). A nearest-assignment tracker with constant-velocity prediction returned both expected final permutations correctly: **2/2 example videos**. Using the median of recent observed per-frame velocities also returned **2/2**. The returned permutation is sorted by each track's final x-coordinate, which matches the notebook's stated convention.

The official validation script expects `valid_data/level_1/...`, but the available repository clone contains only two examples per level. Therefore a full released-validation accuracy is **unmeasured**; 2/2 is sample performance, not a validation score.

## Method and runtime

Initialize IDs by sorting the first complete frame by x. For each later frame, enumerate the six possible box-to-track assignments and select the assignment minimizing squared distance to recent constant-velocity predictions. Sort the final track centers by x to form the answer. This is deterministic and needs no training. Measured inference on the two loaded examples took 0.050 seconds total on the local CPU (0.038 and 0.012 seconds per clip); JSON loading is excluded.

## Alternatives considered

Using image pixels is disallowed here. A fixed box-list index is invalid because detections are reordered. Appearance embeddings and learned detectors add cost without helping this coordinate-only toy task.

## Progressive hints

1. Convert each box `[xmin, ymin, xmax, ymax]` to its center.
2. Give the three cups stable IDs from their initial x-order.
3. Match every later frame globally across all six assignments; extrapolate recent motion to avoid switching IDs when paths approach.
4. The answer is final left-to-right track order, not the inverse start-to-end mapping.

**One-line summary:** velocity-guided box assignment gets 2/2 official examples; the hidden validation split is absent locally.
