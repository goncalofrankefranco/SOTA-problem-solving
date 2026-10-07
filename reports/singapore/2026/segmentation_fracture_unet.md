# NOAI Singapore 2026 — The Segmentation Fracture (Debugging U-Net)

**Problem domain:** Computer vision, semantic segmentation, metric implementation

**Evaluation metric:** 20 marks in the task statement: 10 for repairing the U-Net skip connection and explaining cropping trade-offs, and 10 for implementing mean IoU. A final integrity check has no marks. The released solution notebook labels the architecture answer cell “15 pts,” which conflicts with the task statement's 10-mark Part 1.

## Abridged statement

Repair the released padding-free U-Net so encoder skip features can be center-cropped to the decoder dimensions and concatenated. Implement mean intersection over union (mIoU) directly with per-class intersections and unions, skipping a class only when its union is empty. Explain the information loss and alternatives associated with center cropping.

## Data analysis

There are no satellite images or trained weights in the released task. The notebook uses a random input tensor with shape `(B, 3, 256, 256)` and two output classes, Urban and Forest. Its convolutions use valid padding, so each convolution shrinks the spatial dimensions. The local CPU forward check on `(1, 3, 256, 256)` produced logits of shape `(1, 2, 110, 110)`. The notebook's illustrative comment mentions outputs such as `196 × 196`, but the included architecture's actual sequence of valid convolutions yields `110 × 110`.

## Experiments

- The local CPU forward pass completed and returned `(1, 2, 110, 110)` in about 0.6 seconds.
- For the released “missing class” example, the formal union-based mIoU is **0.375**. Class 0 has intersection 3 and union 4, so IoU is 0.75. Class 1 has a false-positive prediction and no target pixels, so its intersection is 0 and its union is 1, giving IoU 0. The mean is `(0.75 + 0) / 2 = 0.375`.
- The notebook self-check instead expects **0.75** and says to skip class 1 because “Union = 0.” That statement is false for its provided tensors: the false positive makes class 1's union equal to 1. The expected 0.75 comes from skipping classes absent from the target even when they are predicted, which is not the stated union-based IoU.

## Solution

[The implementation](../../../solutions/singapore/2026/unet_miou.py) center-crops the encoder skip tensor using symmetric integer offsets before concatenation. Its default `calculate_miou` flattens the masks, loops over classes, computes intersections and unions from boolean masks, and skips only empty unions. The function also exposes `skip_classes_absent_from_truth=True` as an optional compatibility switch for reproducing the notebook's visible 0.75 self-check; the reported result and chosen metric use the default union-based definition.

## Score evidence and limits

**No mark score was measured.** The task statement gives a 20-mark maximum, but there is no organizer-grader result or released hidden test set. The recorded evidence is the local output shape and the explicit calculation above; the contradictory self-check cannot validate the formal metric. The official solution notebook's “15 pts” label for the architecture cell is another scoring inconsistency relative to the task statement's 10-mark allocation, so a full-mark claim is not supported.

## Implementation and diagnostics

Each valid 3-by-3 convolution reduces each spatial dimension by two pixels, and pooling and transposed convolution alter the spatial size further. Cropping the larger encoder feature map at its center aligns it with the upsampled decoder map without changing the convolution configuration. The metric computes, for each class, `intersection = (prediction_is_class & target_is_class).sum()` and `union = (prediction_is_class | target_is_class).sum()`.

**Rules and dependencies:** The checklist lists PyTorch (`torch`, `torch.nn`, `torch.nn.functional`) and NumPy as allowed, and prohibits scikit-learn, TorchMetrics, and other high-level metric wrappers. The implementation uses PyTorch and writes the metric directly.

## Compute and footprint

The checked forward pass ran on CPU and took about 0.6 seconds for one random input. No dataset, pretrained model, GPU score, or training run is required or was measured.

## Alternatives considered

- Adding padding to the convolutions can preserve feature-map sizes, but changes the released padding-free architecture.
- Interpolation or padding can align feature maps too, but changes how spatial information is represented.
- Cropping the decoder tensor instead of the encoder skip can align the shapes, but the usual U-Net approach crops the larger skip tensor. Cropping discards border features, so it can lose edge context.
- Skipping every class absent from the target, even when it has false-positive predictions, reproduces the notebook's erroneous self-check but is not the union-based IoU described in the task.

## Progressive hints

1. Track the height and width after every valid convolution, pool, and transposed convolution.
2. Compare the skip and decoder sizes; compute centered top and left offsets from their difference.
3. Form per-class boolean masks and calculate intersection with AND and union with OR.
4. Skip a class only when its union is zero. A false-positive prediction makes the union nonzero even if the class is absent from the target.

**One-line summary:** Center-crop the larger skip features for concatenation and average class IoUs computed from the stated intersection-over-union formula.

## Sources and reuse

- [SOTA task summary](https://checklist.sota-ai.org/problems/noai-singapore-2026-final-segmentation-fracture-unet/)
- [Official NOAI 2026 Programming Task 2 notebook](https://aisingapore.org/wp-content/uploads/question_2_v20022026.ipynb)
- [Released solution notebook](https://github.com/AISGNUSNOAI/NOAI-2026/blob/main/question_2_solution.ipynb)
- [Official NOAI learning-resources page](https://noai.aisingapore.org/learning-resources/)

The source does not state a reuse license. This report paraphrases the task and the solution code is independently written; the notebooks are linked rather than copied into this repository.
