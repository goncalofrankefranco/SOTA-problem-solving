# Imbalanced Classification — Stage I (2024)

**Problem name:** Imbalanced Classification (`niezbalansowana_klasyfikacja`)

**Domain:** Binary computer vision, imbalanced learning

**Metric:** Accuracy on a balanced normal/onion test set, converted to points by `(round(accuracy, 2) - 0.5) × 2` above 50%, capped by the task at one point. Full credit is 100% validation accuracy.

## Abridged statement

Train a PyTorch CNN to distinguish noisy light-gray shapes on black backgrounds from similar shapes with dark internal bands. The provided training set is imbalanced; validation/test data is balanced. The required interface has `create_with_training` and `load`, and the model must fit within 35 MB and run on CPU.

## Dataset analysis and EDA

The mirror contains 1,800 labeled 224×224 grayscale training images: 1,400 normal and 400 onion. The released validation split has 100 images, balanced at 50 per class. The filenames encode labels (`normal` → 0, `onion` → 1); the normal examples show noisy filled shapes, while onion images add dark layered bands. The official GitHub notebook downloads the data at run time from Google Drive; the dataset itself is not stored in that repository. The validation inputs/labels used here came from the task-data mirror.

## Experiments

- **Organizer dummy model:** always predicts normal; its worked example records 50% balanced accuracy and zero points.
- **Mirror's worked CNN:** two convolutional blocks with 32 and 64 channels, followed by a dense layer over all `64×56×56` features. Its checkpoint is 102,840,148 bytes (about 98 MiB), above the 35 MB task limit. The mirror notebook records 99% accuracy and 0.98 points.
- **Local reproduction of that checkpoint:** using the starter's raw-logit comparison `output > 0.5`, accuracy was 98% (two onion false negatives). Comparing `sigmoid(logit) > 0.5`, equivalently using the standard `logit > 0` decision boundary, gave 100% on the same split. The discrepancy demonstrates that the mirror checkpoint's raw logits are not calibrated to the grader's probability threshold.
- **Compact CNN:** four convolutional blocks (24/48/72/96 channels), GroupNorm/GELU, adaptive 7×7 pooling, and a small dense head; weighted sampling balances the classes, random horizontal/vertical flips augment only training examples, and the output is a sigmoid probability. In the seeded local experiment this design reached 100% validation accuracy after each of the first three epochs. Training gradients used only the training split. An exact source rerun reproduced the first epoch's recorded training loss, then was stopped during epoch two because of shared-CPU runtime; that rerun did not finish training, validation, or checkpoint writing.

## Selected solution

Use the compact CNN in [imbalanced_classification.py](../../../solutions/2024/stage1/imbalanced_classification.py). It returns probabilities so the starter grader's `>0.5` comparison corresponds to `logit>0`. A weighted sampler corrects the 3.5:1 training imbalance; horizontal/vertical flips are applied only to training images. The model has under one million parameters and the code writes the required `cnn-classifier.pth` when run from the task directory. The generated checkpoint is not included here.

## Validation score and evidence

The matching seeded compact-CNN experiment scored **100/100** after each of its first three epochs on the 100-image released validation split. This is 100% accuracy and one task point. The score is local evidence from the experiment, not from a completed run of the saved standalone source; the exact source rerun was interrupted during epoch two before validation. The result does not establish secret-test performance. On the same released split, the mirror's oversized checkpoint scored 98% under its raw-logit threshold, while the probability-corrected threshold scored 100%.

## Compute and runtime

The compact experiment used CPU PyTorch with four threads, batch size 32, and three epochs. The task permits 15 minutes for training and 2 minutes for CPU inference on 50 images. The standalone source has under one million parameters, so its float32 parameter payload is under 4 MB and fits the 35 MB model-file limit; the exact serialized file size was not measured because the source rerun did not complete. End-to-end training and inference wall-clock times remain unverified in this shared CPU environment. Inference and all training used no GPU; training did not read validation labels.

## Alternatives

The official worked convolutional architecture has a 25.7-million-parameter flattened dense layer and exceeds the checkpoint-size limit. Probability calibration fixes its local threshold mismatch but does not fix that size violation. Texture descriptors with an RBF SVM, HOG features, and smaller spatial pooling are alternatives not measured here.

## Progressive hints

1. Compare class counts in the training set with the balanced validation set.
2. Which visual feature distinguishes onion images from otherwise similar noisy shapes?
3. Balance minibatches and use flips to avoid learning accidental orientation cues.
4. Pool spatial features before the classifier head to keep the parameter file small; return a probability if the grader thresholds at 0.5.

**One-line solution:** Train a compact CNN with balanced sampling and flips, adaptive pooling, and a sigmoid output for the grader's 0.5 threshold.
