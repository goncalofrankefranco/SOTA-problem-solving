# Object Tracking 2 — Occlusion and Blur

## Task and metric

- **Domain:** computer vision, robust object tracking from noisy detections.
- **Metric:** exact permutation accuracy; this subtask awards up to 0.5 points and expects strong performance.
- **Abridged statement:** infer the final left-to-right order of the three cups from box coordinates when boxes can be imprecise, objects occlude each other, and detections disappear in some frames. Images remain for visualization only; the algorithm must consume coordinates.

## EDA

The two official example videos contain 150 frames each. One has 84/150 frames with three detections and the other 67/150; the rest have one or two boxes, with some fully missing frames. Box positions vary in both x and y, and the list order is arbitrary. Occlusion gaps make one-step nearest-neighbor assignment ambiguous.

## Experiments and result

On the two labeled examples, nearest assignment with only the latest displacement as velocity got **0/2**. Replacing that slope with the coordinate-wise median of recent observed velocities, while enumerating assignments to any subset of the three tracks, got **2/2**. This damps box jitter while preserving sustained motion through short gaps.

The official validator uses `valid_data/level_2/...`; that released validation directory is not present in the available clone. Full validation accuracy is **unmeasured**; the reported 2/2 is only on labeled examples.

## Method and runtime

Initialize IDs from the first complete frame. At each frame, compute a prediction for each track from the median of recent observed displacement-per-frame vectors. Enumerate injective mappings from the available boxes to tracks (there are at most six) and minimize the squared center-to-prediction distance. Missing tracks remain unmodified until observed again. The function uses coordinates only and took 0.052 seconds total on the local CPU for the two 150-frame examples (0.027 and 0.024 seconds per clip); JSON loading is excluded.

## Alternatives considered

Images are explicitly excluded for this level. Raw last-step velocity is too sensitive to jitter and missing detections. A full learned tracker is unnecessary with only three trajectories; a larger motion model can be considered if full validation later reveals harder gaps.

## Progressive hints

1. Treat a missing box as a missing observation, not a new cup.
2. Keep a stable ID for each initial cup and infer final order by x-position.
3. Use recent motion to predict through one- or two-frame gaps.
4. Use a robust velocity estimate across several observations; enumerate assignments instead of relying on the box-list order.

**One-line summary:** robust median-velocity assignment gets 2/2 official examples; released validation files are unavailable locally.
