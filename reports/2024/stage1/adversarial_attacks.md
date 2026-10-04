# Adversarial Attacks — Stage I (2024)

**Problem name:** Adversarial Attacks (`ataki_adwersarialne`)

**Domain:** Adversarial robustness, image classification

**Metric:** Mean structural similarity (SSIM) × decrease in classifier accuracy. The released validation notebook reports base accuracy as 91.9636% and the full-credit threshold is a criterion value above 42; values below 36 receive zero. The evaluator also rejects any pixel displacement above 0.3 in normalized `[-1, 1]` space.

## Abridged statement

Given a frozen MNIST-like classifier and grayscale images normalized independently to `[-1, 1]`, return perturbed images that reduce the model's accuracy while preserving as much structural similarity as possible. The attack function receives images but no labels; the hidden test images and labels are unavailable.

## Dataset analysis and EDA

The local mirror contains 48,128,128 bytes of training images and 8,624,128 bytes of released validation images, with 11,000 validation labels. Images have shape `28×28`; the organizer notebook's validation run gives 91.9636% clean accuracy. The corresponding 4,049,428-byte classifier checkpoint is supplied by the task and was loaded for local evaluation. The task notebook normalizes each image using its own minimum and maximum, then converts the returned float array to a tensor without another normalization.

## Experiments

- **Untouched-image baseline:** SSIM 1.0 but no accuracy reduction, so the criterion is zero.
- **Single-step FGSM, `ε=0.3`:** one sign-gradient step maximized cross-entropy for the model's initial predicted class. Locally measured final accuracy was 16.05%; reconstructed flattened-vector SSIM was 0.4552, for 34.56 criterion units. This is below the full-credit threshold.
- **Three-step projected sign-gradient attack, `ε=0.299`, step size 0.1:** final accuracy was 5.70%; reconstructed SSIM was 0.5639 and the criterion was **48.65**. This exceeds the full-credit threshold. The measured maximum pixel displacement was 0.29900002.

The SSIM implementation in the local experiment reproduces scikit-image's default 1D, seven-sample window calculation used by the notebook (`ssim` is applied after flattening each image). The runtime does not include scikit-image, so this SSIM and the resulting criterion are reconstructed from its standard formula, not obtained by executing the imported evaluator function. Clean and attacked accuracy were measured with the supplied checkpoint.

## Selected solution

Use untargeted PGD with the initial model prediction as a pseudo-label, three gradient-ascent sign steps, and projection into the per-pixel `L∞` ball around the input. The method does not use validation labels or update classifier weights. It is implemented in [adversarial_attacks.py](../../../solutions/2024/stage1/adversarial_attacks.py). The notebook-compatible `perturbe_dataset` loads the organizer checkpoint when no model is passed; the standalone helper accepts a model directly.

## Validation score and evidence

The strongest measured candidate scored **48.65 criterion units**, above the organizer's `>42` full-credit boundary (**100/100 task points**). Its measured components were 91.9636% clean accuracy, 5.70% attacked accuracy, reconstructed SSIM 0.5639, and max pixel displacement 0.29900002. The final source settings were measured on all 11,000 released validation images in 164.9 seconds on CPU. The local runner does not have scikit-image installed, so SSIM was reconstructed from scikit-image's documented standard 1D seven-sample-window formula rather than executed through the imported evaluator function. The score has a 6.65-unit margin over the full-credit boundary. No hidden-test result is claimed.

## Compute and runtime

The measured attack used CPU PyTorch with batch size 512 and three forward/backward passes per batch. The task allows five minutes in a GPU-backed Colab environment. The measured CPU run took 164.9 seconds, within that wall-clock limit even without GPU.

## Alternatives

One-step FGSM is faster but missed full credit on the released validation set. Targeted PGD, random-start PGD, and lower-budget attacks could trade attack strength against SSIM, but have not been measured. The three-step untargeted attack is the current best measured candidate.

## Progressive hints

1. The attack function gets no true labels; can the frozen classifier provide a usable pseudo-label?
2. Which objective will reduce confidence in that initial prediction?
3. How can each gradient step stay inside the per-pixel `L∞` limit?
4. Project the update back into the `ε` ball and compare the SSIM × accuracy-drop criterion.

**One-line solution:** Maximize loss for the classifier's initial prediction with three projected sign-gradient steps, staying just inside the 0.3 `L∞` bound.
