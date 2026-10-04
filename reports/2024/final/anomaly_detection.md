# Anomaly Detection — 2024 Polish AI Olympiad Final

**Problem domain:** Unsupervised computer vision / one-class anomaly detection  
**Evaluation metric:** Accuracy on a balanced hidden test set. The score is `clip((accuracy - 0.60) / 0.30, 0, 1)`; accuracy at or above 90% earns 1 point.

## Abridged statement

Use a neural network to detect anomalous RGB images. Training contains only normal examples. The released validation set is labeled and balanced; the secret test set contains 10,000 similarly distributed examples. The submission exposes `BATCH_SIZE`, a trained `Model`, `forward()`, and `predict(batch)` and must fit and evaluate within 15 minutes on a Colab GPU.

## Dataset analysis and EDA

The supplied archive expands to 7,000 normal training images and 2,000 validation images (1,000 normal, 1,000 anomalous), each 128×128 RGB. Visual samples show a marked domain shift: normal images are aerial views with vegetation, fields, and buildings; anomalies are close-up natural images, including frogs. This makes coarse color, texture, and spatial-layout statistics useful alongside learned features.

Each image is resized to 32×32 and represented by 353 values: channel moments, quantiles and histograms; gray-level gradients and Laplacian summaries; saturation statistics; coarse RGB layout; and a grid of local texture standard deviations. The validation labels are used only to calculate reported metrics.

## Experiments

| Method | Validation evidence | Decision |
|---|---:|---|
| Participant Rayan notebook: convolutional image autoencoder, SSIM/L1 loss | 82.15% accuracy; score 0.7383 | Excluded as a compliant comparison: that notebook chooses its threshold by maximizing accuracy on validation labels. |
| MLP autoencoder latent alone + Ledoit–Wolf density | 87.61% ROC AUC; 73.25% accuracy at the training 95th-percentile threshold | Rejected; latent-only density lost useful low-level signals. |
| Standardized raw statistics + Ledoit–Wolf density | 91.55% accuracy and 0.97619 AUC at the training 95th-percentile threshold | Strong baseline, but does not satisfy the neural-network requirement by itself. |
| **Neural autoencoder embedding concatenated with standardized statistics + Ledoit–Wolf density** | **91.30% accuracy, 0.97534 ROC AUC, 1.000/1.000 official points** | Selected: uses a trained neural representation and only normal training images. |

The selected detector trains a 353→128→64→32→64→128→353 autoencoder for ten epochs using MSE on standardized image features. It then fits a Ledoit–Wolf shrinkage covariance model on the concatenation of the 32-dimensional learned representation and all 353 standardized features. Its decision boundary is fixed at the 95th percentile of scores measured on the normal training set; no validation labels determine the model, density, or threshold. The released balanced validation set received 48.5% positive predictions.

Other threshold quantiles and small weights on reconstruction error were checked as diagnostics; the 95th-percentile rule gives a plausible operating point for the explicitly balanced test population. AUC summarizes ranking quality independently of the threshold.

## Result, compute, and limits

Implementation: [anomaly_detection.py](../../../solutions/2024/final/anomaly_detection.py).

Fresh local run using the released archive and cached PyTorch 2.4.1 CPU completed fitting in **72.2 seconds** and validation inference in **18.3 seconds**. The measured accuracy was **91.30%**, yielding **1.000/1.000 released-validation points** under the official formula. The local run had no CUDA device; the required Colab T4 runtime was not measured. The model uses batch size 128 and performs fitting plus inference in-process, within the same design as the 15-minute evaluation constraint.

The 100% score is only the score formula's ceiling on released validation; it does not establish secret-test accuracy. Domain shift in the hidden set remains unknown. The participant notebook's 82.15% result is retained only as source-reported context because its threshold used validation targets.

## Progressive hints

1. Inspect a few normal-only training images before choosing an anomaly score.
2. Compare color distributions, local texture, and coarse spatial layout after downsampling.
3. Train a neural representation only on normal images, then measure how far learned and observed features fall from the normal distribution.
4. Set the threshold from training-only normal scores and report validation labels only as an evaluation metric.

**One-line summary:** A feature autoencoder plus shrinkage Mahalanobis distance, calibrated to the normal training-score 95th percentile, scored **91.30% accuracy and 1.000/1.000 points** on released validation without using its labels for fitting.
