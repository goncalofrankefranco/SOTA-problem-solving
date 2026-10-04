# Poland 2025 Stage I — Maszynka do Liczenia Monet (Coin Counter)

**Domain:** Computer vision, multi-object detection and denomination classification  
**Metric:** COCO mean average precision averaged over IoU thresholds 0.50–0.95. The task maps mAP ≤0.20 to 0 points and mAP ≥0.85 to 100 points.  
**Official sources:** [SOTA checklist](https://checklist.sota-ai.org/) · [official task notebook](https://github.com/OlimpiadaAI/II-OlimpiadaAI/tree/main/1_etap/1_maszynka_do_liczenia_monet)

## Abridged statement

For each photograph, return bounding boxes, one of nine Polish coin denominations, and a confidence for every visible coin. The notebook evaluator consumes tuples `(x1, y1, x2, y2, label, confidence)`. The final check allows up to ten minutes total on GPU and five seconds per image; external datasets and pretrained weights are prohibited.

## Dataset analysis and EDA

The official loader expects `train.pkl` and `val.pkl` under `data/`; each record contains an image tensor, coin boxes, and labels 0–8. The notebook's starter EDA computes an average coin width of about 62 pixels and therefore uses a 64-pixel crop. The linked pickles are not in the official GitHub tree or local mirrors. Downloading them from the two Google Drive IDs in the notebook failed with `HTTP 403: CONNECT tunnel failed` in this environment, so image shapes, class counts, per-image coin counts, and validation overlays could not be inspected locally.

## Experiments and selected solution

The official notebook includes an instructional baseline, not a worked solution with a recorded score. It samples 128 crops per training image, assigns one of nine coin classes or background by crop/box IoU, trains a random-initialized ResNet-18 classifier for 30 epochs, and scans 64-pixel windows at stride 32. The notebook explains that the fixed, offset windows hurt localization, especially at higher IoU thresholds. Its execution outputs contain no measured baseline mAP.

I added a runnable candidate in [coin_detector.py](../../../solutions/2025/stage1/coin_detector.py). It trains a random-initialized ResNet-18 patch classifier and box-offset regressor on jittered positive crops and random background crops, scans multiple window scales, decodes predicted box offsets, and applies class-wise NMS. It uses only the official training dataset and contains no pretrained weights.

## Validation score and evidence

**No validation score is available.** The Google Drive assets are unreachable here, so neither the official baseline nor the candidate could be evaluated against the released validation split. This is the exact blocker; no score or secret-test performance is inferred. PyTorch 2.4.1 and torchvision 0.19.1 were available locally; the candidate imports, constructs, and passes a CPU forward smoke check on a synthetic 64×64 image, returning a six-field detection tuple. No GPU is available for the task's expected training runtime. The candidate should therefore be treated as unverified code, not as an optimal solution.

## Compute and runtime

The official limit is ten minutes total on GPU and five seconds per image. The organizer notebook's baseline uses 30 training epochs, 128 crops per image, and one 64-pixel crop per sliding-window position. The candidate uses 15 epochs by default, 64 generated crops per image per epoch, and batches multiscale windows in groups of 256. No wall-clock timing could be measured here.

## Alternatives

- Tune the crop classifier and use narrower sliding-window strides; this keeps the starter's classification setup but still needs regression or dense windows for accurate boxes.
- Train a Faster R-CNN-style detector from random initialization; this may improve localization but has a larger training and inference footprint.
- Detect circular contours and classify coin color/diameter; useful as a proposal generator, but not measured and potentially brittle under shadows, overlap, or camera changes.

## Progressive hints

1. Estimate the typical coin diameter from the training boxes before choosing window sizes.
2. Generate crops centered near labeled coins and include background crops so the model learns when to return no object.
3. Vary crop scale and position, then regress from the crop to the coin box and suppress duplicate detections with NMS.

**One-line summary:** A multiscale patch classifier with box regression and class-wise NMS is implemented, but no released-validation score can be measured until the Google Drive assets are available.
