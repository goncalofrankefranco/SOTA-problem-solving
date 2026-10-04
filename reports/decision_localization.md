# Lokalizacja decyzji — Decision Localization

**Problem domain:** Computer vision, weakly supervised object localization, visual explanations  
**Evaluation metric:** Mean IoU between thresholded heatmaps and COCO instance masks

## Abridged statement

Given a frozen binary ResNet-18 and 224×224 images containing its positive class, return a heatmap highlighting the object pixels that support the classifier's positive decision. The heatmap is compared with the object's COCO mask. The provided split has 1,200 labeled validation images; the hidden test split has 1,200 images.

## Dataset analysis and EDA

The starter notebook stores the validation examples in `data/val_data.npz`, with `val_ds` records containing a `uint8` RGB image and a `uint8` mask. Images are normalized with ImageNet channel means and standard deviations before inference. The frozen model is a ResNet-18 whose `encoder` returns its final 7×7 convolutional features, followed by adaptive average pooling and a one-logit linear head.

The released organizer repository contains the notebook and requirements but not the archive or model checkpoint. The notebook points to Google Drive file IDs `19jSIGHXp_BY_gkyGdYKKc7ls-QU7_Rcn` (validation archive) and `1b6WhjG_GDihBNKLIXtFcaclJKkTWbZZk` (checkpoint). A public mirror points to an alternate validation archive ID, `1hVn9m0KRuE-0x7zpPYcwkxDFMNoaM-U9`, but its solution notebook also retains the random baseline. Those assets are absent from this workspace, and direct Google Drive requests from this execution environment returned HTTP 403 at the connection tunnel. I therefore could not inspect the masks, plot examples, or calculate validation metrics.

**Inspected:** the complete official notebook, its dataset loader and executable evaluator, the organizer repository's tracked files, and a public mirror's translated starter/solution notebooks. Both solution notebooks still contain the random-mask baseline; the mirror did not provide a worked result.

**Observed from available materials:** there are 1,200 224×224 validation examples, the prediction is evaluated as a binary mask at a fixed threshold, and the classifier's last spatial feature map is 7×7. No image-level distribution, mask-shape, class-activation, or IoU observations are possible without the external files.

With the archive available, the first EDA checks should be mask area and connected-component distributions; image/mask overlays; positive-class logits; and overlap between high-activation regions and the annotated mask. These would show whether the classifier attends to the object, a discriminative part, or a correlated background cue.

## Solution strategy

The submitted implementation uses Grad-CAM on the final ResNet block:

1. Compute the positive-class logit from the final convolutional feature map.
2. Differentiate that logit with respect to the feature map and average each channel's gradient spatially to obtain channel weights.
3. Take the ReLU of the weighted channel sum, bilinearly upsample it to 224×224, and min-max normalize it to `[0, 1]`.

The evaluator wraps inference in `torch.no_grad()`, so the function re-enables gradients locally. It does not change model weights. The evaluator binarizes both prediction and target with `> 0.5`, then averages IoU across examples.

## Results and diagnostics

**Validation IoU and points: not measured.** The model checkpoint and validation archive could not be retrieved in this workspace, so there is no verified score and no evidence that this implementation reaches 100/100. The current code is a plausible baseline, not a validated optimum.

There is a scoring-text inconsistency in the official notebook. Its prose says IoU at or below 0.25 earns zero, but the executable `compute_score` instead calculates `100 * (mIoU - 0.191) / (0.245 - 0.191)`, clamps to `[0, 100]`, and rounds. Validation against the executable function is needed; under that code, a mean IoU of at least 0.245 earns 100.

## Implementation and compute footprint

The code is in [`solutions/stage3/lokalizacja_decyzji.py`](../solutions/stage3/lokalizacja_decyzji.py). It performs one forward pass and one gradient calculation per image. The task allows GPU execution and a three-minute total limit. Runtime was not measured here because the required PyTorch runtime, checkpoint, and archive are not present in this workspace.

Potential alternatives to compare on the released validation split include Grad-CAM++ and LayerCAM, and postprocessing that expands activation seeds along image edges. Occlusion maps may offer another signal but can require many more forward passes. No alternative was benchmarked; changing the current implementation without the validation set would be guesswork.

## Hints

1. Which model layer still retains spatial layout while encoding class-specific features?
2. How can the positive logit's gradient indicate which feature channels matter?
3. How should the coarse activation map be resized and scaled for the evaluator's fixed threshold?
4. Compare overlays and IoU on the released validation set before tuning thresholds or adding segmentation postprocessing.

**One-line solution:** Weight the final convolutional feature maps by the positive-logit gradients, then upsample and normalize the Grad-CAM map.

**Official task and starter notebook:** [Lokalizacja decyzji](https://github.com/OlimpiadaAI/III-OlimpiadaAI/tree/main/3_etap/lokalizacja_decyzji)
