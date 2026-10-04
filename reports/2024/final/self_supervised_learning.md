# Self-Supervised Learning — 2024 Polish AI Olympiad Final

**Problem domain:** Multivariate time-series representation learning and few-label activity classification  
**Evaluation metric:** Accuracy on an organizer-only balanced test set. Below 70% earns 0 points, at or above 90% earns 1 point, and the score is `clip((accuracy - 0.70) / 0.20, 0, 1)`.

## Abridged statement

Learn a representation from unlabeled sensor sequences `train_x_big.pt`; use the small labeled training subset to fit a classifier; return one integer class index for each test sequence. The task uses six classes and data shaped `(N, 1, 3, 206)`. Evaluation must finish within two minutes on a Colab T4, and the submitted encoder checkpoint must be below 10 MB.

## Dataset analysis and EDA

The official task describes three sensor channels over 206 time steps, with a large unlabeled training set, a small labeled training set, and validation/test sequences. The local Rayan notebook reports that `self_supervised.zip` was 25.1 MB and records six classes. The ZIP and all five tensor files (`train_x_big.pt`, `train_x_small.pt`, `train_y_small.pt`, `val_x.pt`, `val_y.pt`) are absent from the available local directories. Google Drive downloads are blocked in this environment, so sample counts, signal distributions, and class balance beyond the statement's balanced test-set note could not be inspected.

The local `encoder.pt` artifact is **185,319 bytes**, far below the 10 MB limit. Its state dictionary contains a three-layer 1D convolutional encoder, BatchNorm, global temporal pooling, and a 64-dimensional projection head.

## Experiments and selected method

| Method/evidence | Accuracy | Official score | Provenance |
|---|---:|---:|---|
| Starter random encoder with an MLP head | Not recorded | Unavailable | Official starter notebook only. |
| Contrastive temporal CNN + supervised MLP fine-tuning | **89.46295%** | **0.97315 / 1** | Accuracy printed by the local Rayan participant notebook; not freshly reproduced here. |
| Symmetric normalized InfoNCE or nearest-class-prototype head | Not measured | Unavailable | Candidate alternatives; labeled and unlabeled tensor files are missing. |

The participant run constructs two unlabeled views: Gaussian noise plus random scaling, and time masking plus random cropping. A one-way in-batch contrastive loss trains the encoder on unlabeled data. For evaluation, it freezes encoder weights, adapts BatchNorm statistics on the labeled subset, and trains a 128→60→6 MLP for 50 epochs using only `train_y_small`. The test sequences and validation labels do not enter the fit. The downloaded checkpoint and notebook use the same encoder architecture as the implementation below; the implementation also clones samples before in-place augmentations and handles remapped class IDs.

Implementation: [self_supervised.py](../../../solutions/2024/final/self_supervised.py); included weights: [encoder.pt](../../../solutions/2024/final/encoder.pt).

## Result, compute, and limits

The saved notebook output reports **0.8946295037 validation accuracy**, corresponding to **0.9731475/1** under the official formula. This is a participant-notebook result, not an organizer score and not a fresh run of this implementation. The notebook output contains no reliable fit or inference wall time. Our environment has CPU PyTorch but lacks the dataset tensors; the two-minute T4 constraint was therefore not directly measured. The copied checkpoint passes the size limit; hidden-test accuracy remains unknown.

The available score is already close to the formula ceiling. Whether a normalized symmetric contrastive objective, stronger time-series augmentations, a linear probe, or a class-balanced prototype/nearest-neighbor head improves it cannot be established without the missing training and validation tensors. These alternatives must continue to use `train_x_big` without labels and fit supervised heads only on `train_x_small` / `train_y_small`.

## Progressive hints

1. Preserve the temporal axis: encode each of the three sensor channels as a sequence, not as an unordered vector.
2. Form two plausible views of each unlabeled signal with mild time-series augmentations.
3. Train the views to agree while separating unrelated examples, then use the encoder's pooled representation.
4. Fine-tune only on the small labeled training subset; keep validation labels for scoring.
5. Confirm the checkpoint fits the size cap and batch test inference to respect the T4 time limit.

**One-line summary:** A contrastive 1D CNN checkpoint with an MLP trained on the labeled subset recorded **89.46295% validation accuracy and 0.97315/1 points** in the participant notebook; a fresh local score is blocked by missing tensors.
