# The Five Artists — Colombia AI Olympiad 2025 Final

**Problem domain:** Computer vision, supervised five-class image classification  
**Official metric:** Accuracy

## Abridged statement

Train a model to identify which of five artist siblings drew each illustration. Their styles are similar, so the classifier must learn subtle visual differences. Predict an artist ID from 0 through 4 for each image in the test set. The original statement is in Spanish; the [SOTA page](https://checklist.sota-ai.org/problems/colombia-ai-olympiad-2025-final-five-artists/) provides the task summary and notebook.

## Dataset analysis and EDA

The official notebook loads `eleon360/five-artists-dataset` and `eleon360/five-artists-test-dataset` from Hugging Face. The dataset viewer reports 10,000 labeled training images at 256×256 pixels, with integer `artist_id` labels from 0 to 4; the test dataset has 2,000 images with IDs 10,000–11,999. These counts and dimensions come from the [training dataset viewer](https://huggingface.co/datasets/eleon360/five-artists-dataset) and [test dataset viewer](https://huggingface.co/datasets/eleon360/five-artists-test-dataset).

The official statement says the artists' styles differ only in small personal details. SOTA's inspection of the viewer supports image width/height 256 and five label IDs. Exact class counts, pixel statistics, duplicates, and per-class image grids were not measured here because the 113MB train parquet and test image collection could not be downloaded from this execution environment.

## Experiments

| Experiment | Result |
|---|---|
| Official notebook baseline: flattened 32×32 RGB images, dense network, five epochs | Organizer-provided baseline; no reported held-out accuracy available |
| Stratified 85/15 validation with the residual CNN in this repo | Not run: image assets could not be downloaded; no measured accuracy |
| Kaggle public re-run submission | Not submitted; the listed public run ended 15 January 2026 |

No score is inferred from training-set accuracy or from Kaggle leaderboard status.

## Chosen method

[The runnable solution](../../solutions/colombia/five_artists.py) uses a small residual CNN operating on 128×128 RGB images. It applies modest rotation and color jitter, keeps horizontal reflection off to preserve any direction-dependent artist details, and optimizes cross-entropy with label smoothing. A stratified 15% holdout selects the best epoch by validation accuracy; the model then trains on all labeled images for that number of epochs and writes `image_id,artist_id` predictions.

The script defaults to 128-pixel inputs to keep GPU and CPU cost manageable while preserving more detail than the official baseline's 32×32 flattening. It prints class counts and per-epoch accuracy/macro-F1 when run, so its local validation result is reproducible.

## Score and provenance

**Measured score: none.** Accuracy is the task metric, reported by the SOTA task page. The local data-dependent score was not measured because the Hugging Face image files were not accessible from this execution environment. The official notebook does not provide a validation result in the published summary. No 100% or leaderboard score is claimed.

## Implementation and diagnostics

The script downloads the official Hugging Face splits at runtime, checks that labels are in `{0,1,2,3,4}`, makes a stratified validation split, prints per-class validation results, then writes a submission CSV using the public rerun's `image_id` and `artist_id` columns. The original starter notebook instead writes `predicted_artist`; check the active competition's sample submission before submitting. The public Kaggle rerun is now closed, so the output is useful for local validation or another authorized evaluation.

The translated notebook text says “convolutional neural network,” but its starter baseline actually flattens each 32×32 image and uses dense layers. The chosen method uses convolutional spatial structure, which better preserves local strokes and motifs.

## Compute and footprint

Training uses PyTorch and torchvision. A CUDA GPU is recommended for the 10,000 images; the organizer explicitly mentions a T4 runtime for larger neural networks. CPU inference and training are supported, though no local runtime was measured. No pretrained weights or image assets are included in this repository.

## Alternatives considered

- The organizer's small fully connected baseline is easy to run but discards spatial neighborhood structure after flattening.
- A pretrained ResNet/EfficientNet could help if weights are allowed and available, but no such result was tested here.
- Higher resolution, test-time augmentation, and a model ensemble are possible follow-ups, but should be selected using stratified validation rather than assumed to help.
- Horizontal flipping may be harmful if the artists use direction-specific strokes or compositions, so it is omitted by default.

These alternatives were not measured on the unavailable image data.

## Progressive hints

1. Confirm the mapping and counts for all five artist IDs before training; preserve image IDs for submission.
2. Compare class counts and inspect representative images per artist rather than only the first few rows.
3. Use a stratified validation split and preserve spatial structure with a CNN.
4. Track validation accuracy, then refit on all 10,000 labeled images using the chosen epoch count and match the submission column names exactly.

**One-line summary:** Learn the five subtle drawing styles with a convolutional classifier; the complete training pipeline is included, but the accuracy remains unverified because the image files could not be downloaded.

## Sources and reuse

- [SOTA task page and translated notebook](https://checklist.sota-ai.org/problems/colombia-ai-olympiad-2025-final-five-artists/)
- [Official Spanish statement/in-contest Kaggle page](https://www.kaggle.com/t/f1f01eefe5de41369c261d39b75e6524)
- [Public Kaggle re-run](https://www.kaggle.com/competitions/colombian-ai-olympiad-pr-3-five-artists)
- [Labeled training dataset](https://huggingface.co/datasets/eleon360/five-artists-dataset)
- [Unlabeled test dataset](https://huggingface.co/datasets/eleon360/five-artists-test-dataset)

The SOTA page says reuse is subject to the Kaggle competition rules. The Hugging Face card does not state a specific dataset license. This repository links both sources but does not redistribute the images or notebook.
