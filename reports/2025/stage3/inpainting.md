# Inpainting (dynamic video)

## Problem, domain, and metric

- **Domain:** computer vision, video inpainting, and binary segmentation.
- **Task:** implement a coordinate-queryable `nn.Module` that maps normalized `(x, y, t)` coordinates to BGR pixel color and an object mask. The official starter requests an INR, and supplies full RGB/mask frames only for the training timestamps.
- **Metric:** official score is `7 * P_PSNR + 30 * P_acc`, rounded to an integer. `P_PSNR` saturates at PSNR 23.5 dB; `P_acc` saturates at mask accuracy 0.98. Validation/test queries cover a 64×64 rectangle in each held-out frame.
- **Abridged statement:** reconstruct the missing 64×64 region at held-out video times, using only the released training frames; no external data or pretrained weights are allowed, and training is capped at six minutes on GPU.

## Released data and EDA

The [official task notebook and archives](https://github.com/OlimpiadaAI/II-OlimpiadaAI/tree/main/3_etap/1_inpainting) contain 59 training frames and 10 released validation frames, all 256×256, from a 79-frame sequence. Training timestamps span 0–78. The validation rectangles are 64×64 and move around the image. The training frames include complete RGB images and masks; validation images/masks are used only by the official evaluator and this report's scoring run.

Validation frame IDs are `8, 23, 30, 39, 46, 49, 50, 55, 57, 63`. Their nearest bracketing training frames are at most four indices apart. This makes train-only temporal interpolation a strong baseline. The mask is a large moving silhouette, so predicting a static majority class would not capture its boundaries.

## Experiments

All results below use training frames only to construct predictions; released validation labels were used only to score/select settings.

| Method | Mean validation PSNR | Mean mask accuracy | Official rounded score |
|---|---:|---:|---:|
| Nearest training frame, per pixel | 26.69 dB | 95.37% | 95/100 |
| Linear RGB and mask interpolation | 29.33 dB | 95.48% | 95/100 |
| Temporal median of training frames | 20.41 dB | 84.67% | 46/100 |
| Bidirectional RGB optical flow, Farneback (`0.5, 4, 21, 5, 7, 1.5`) | 31.84 dB | 96.88% at the default mask threshold | 98/100 |
| Bidirectional mask flow plus 25% RGB-flow mask estimate; threshold 0.675 | **31.84 dB** | **97.871%** | **100/100** |

The selected score before rounding is **99.742/100**, rounded to 100 by the official evaluator. PSNR already saturates at 10 of the 10 available image-quality points; the remaining score is 29.742 of 30 mask points. The mask threshold and the 75:25 mask-flow/RGB-flow blend were selected on released validation; the model never fits to validation RGB or mask targets.

## Selected solution

[inpainting.py](../../../solutions/2025/stage3/inpainting.py) implements the official `YourSolution` interface. Its `train` entry point reads only `train_loader`, stores the known frames, and fills missing integer timestamps by bidirectional optical-flow warping of the bracketing training frames. It predicts color from RGB flow and mask from a 75:25 blend of mask-derived and RGB-derived flow, then thresholds the mask at 0.675. At inference it maps the input coordinate grid to the requested frame crop and gathers those cached values. It does not need gradient training or any validation/test assets.

### Score evidence and runtime

The implementation was run against the official released archives with the official coordinate normalization and scoring formula reconstructed locally. It achieved 31.843 dB mean PSNR, 0.978711 mean mask accuracy, raw score 99.742, and rounded score **100**. This is a locally measured released-validation score; hidden-test score is unknown.

Local CPU execution of training-frame collection plus optical-flow preprocessing took **12.2 seconds**. The serialized full-video uint8 buffer is about **20.7 MB**. GPU runtime should be lower, and inference is a table lookup over 4,096 points per frame. No GPU was available in this workspace, so the platform GPU runtime was not measured.

## Alternatives and caveats

An INR/SIREN fitted to the 59 training images is closer to the introductory framing and may handle continuous coordinates more naturally, but requires substantial GPU fitting and was not locally trained here. Nearest-frame lookup is simpler and loses about four dB on this validation set. Optical flow improves RGB quality, but the silhouette changes abruptly during gait; threshold tuning was needed to approach the mask ceiling. The validation set has only ten frames, so 100/100 should not be read as a hidden-test guarantee.

## Progressive hints

1. The observed timestamps nearly cover the whole video; inspect their gaps before training a large INR.
2. Predicting from the closest frame is a strong baseline, but motion causes blur or stale edges.
3. Warp both neighboring frames toward the query time with forward/backward optical flow and blend them.
4. Estimate mask motion from the mask itself, then blend in a smaller RGB-flow estimate to stabilize low-texture areas.

**One-line summary:** bidirectional flow interpolation from train frames scored 100/100 after official rounding on released validation (31.84 dB, 97.871% mask accuracy); hidden-test score remains unknown.
