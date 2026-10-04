# Object Tracking 3 — Build from Scratch

## Task and metric

- **Domain:** computer vision, color segmentation and multi-object tracking.
- **Metric:** exact permutation accuracy; this subtask awards up to 0.5 points.
- **Abridged statement:** from raw frames, detect and track the three cups without supplied boxes, then return their final left-to-right order. Inference must complete within five minutes on CPU.

## EDA

The two local examples use 640×480 RGB frames (150 and 120 frames). The cups are saturated red against a pale blue/white checkerboard, making a simple red-pixel mask a useful detector. Connected components produce three detections in many frames, but occlusion merges multiple cups into a single component. There are no training annotations beyond the one example track file, and no official `valid_data` directory is present.

## Experiments and result

1. Red thresholding plus connected-component centroids and one-step velocity tracking did not recover the full permutation.
2. Using a median recent-velocity estimate improved coordinate-level tracking but still confused identities at long merged-component intervals.
3. Filtering oversized merged components using a per-video component-size reference and then tracking with median velocity gets **1/2 labeled examples** (the second example is correct; the first remains ambiguous). This is sample-only performance; official validation accuracy is **unmeasured** because the validation videos and targets are absent.
4. A 100-state beam search over recent identity assignments produced the same two predictions and still scored **1/2**; it took about 16 seconds across the two samples, compared with 5.2 seconds for the simpler framewise method. It did not justify the extra cost.

Component extraction and tracking took 3.07 seconds total for the two examples on the local CPU (1.84 seconds for 150 frames and 1.23 seconds for 120 frames), excluding image loading. The earlier end-to-end sample harness, including image loading, took 5.24 seconds.

## Method and alternatives

Threshold red pixels, remove tiny regions, and use connected-component centroids and extents as detections. Estimate typical component size from frames with three separate components; discard a component that is clearly a union during partial-visibility frames. Track the remaining centers with the robust assignment method from Task 2. The approach is transparent and has no pretrained weights, but long intervals where cups merge remain an information bottleneck. Beam search did not resolve the difficult sample; watershed splitting or a temporal image model are plausible next steps, though neither can be measured without more labeled clips.

## Progressive hints

1. The cup color gives a strong foreground cue without a pretrained detector.
2. Use connected red regions to get candidate centers and sizes.
3. Treat an abnormally large merged region as uncertain instead of assigning it to one cup.
4. Preserve velocity hypotheses across occlusion and use later separated frames to resolve identity.

**One-line summary:** red-component tracking is runnable on examples and gets 1/2; full validation and the difficult first sample remain unresolved.
