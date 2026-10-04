# Rozkład nienormalny — Non-Normal Distribution

**Problem domain:** Image denoising, noise-family classification, and Gaussian parameter estimation  
**Official metric:** Four weighted components, 25 points each: classification accuracy scaled from 0.50 to 0.95, mean PSNR scaled from 10 to 16 dB, and full credit for each of Gaussian `μ` and `σ` MSE below 0.005. The sum is rounded to an integer.

## Abridged statement

Given a noisy 28×28 grayscale image, return a denoised image, a probability for uniform noise (label 1; Gaussian is label 0), and estimates of Gaussian mean and standard deviation. Train only on paired clean/noisy images and noise-family labels; parameter labels are released only for validation/test scoring. Platform inference is limited to five minutes on GPU.

## Data analysis

- Official training data: 30,000 paired images, evenly split between Gaussian and uniform noise. Validation data: 2,000 pairs, evenly split, with parameters available for scoring.
- Images are stored as `28×28×1` `uint8` arrays and must be scaled to `[0, 1]`. Training rows contain `original`, `noised`, and `label`; they do not contain `params`.
- Validation Gaussian parameters have `μ` mean/std `0.2421 / 0.1439`, with range `0.0003–0.4997`; `σ` mean/std `0.1768 / 0.0711`, with range `0.0501–0.3000`. Uniform rows use parameters `[0, 0.4]`.
- The training data retains clean/noisy pairs, so clipping is observable. The solution uses the correct censored likelihood for pixels clipped to 0 or 1 rather than treating their residuals as exact noise samples.

## Experiments

The official notebook defines the evaluation function but its released starter solution is only a template. The mirror's alternative notebook contains a random-output placeholder; it has no recorded organizer score. No validation parameter targets are passed to either training stage.

| Experiment | Training inputs | Result |
|---|---|---|
| Official starter architecture / mirror placeholder | Paired images and labels | No released score recorded; mirror placeholder returns random predictions. |
| Shared CNN with residual denoising, binary noise head, and censored Gaussian likelihood head | `train.pkl` only; five CPU epochs | **98/100**; classifier accuracy 0.909. |
| Refit only the classifier's final linear head on frozen train-image features | `train.pkl` only; 40 CPU head epochs after the five-epoch fit | **100/100**; classifier accuracy 0.960. Denoising and Gaussian estimates are unchanged. |

## Chosen method

[The solution](../../../solutions/2025/stage2/rozklad_nienormalny.py) trains a shared convolutional network for residual denoising and global noise features, then refits only the classifier's final linear layer on frozen features from `train.pkl`. The classifier returns probabilities (the released evaluator thresholds the returned value directly at 0.5). Gaussian `μ` and `σ` are trained from the paired training pixels with a likelihood that treats clipped observations as left- or right-censored; validation/test parameter fields are never read. The code creates no checkpoint or copied data artifact.

## Measured score and compute

The five-epoch train-only fit initially scored **98/100** on the 2,000 released validation images: PSNR `19.8606 dB`, classification accuracy `0.909`, Gaussian-mean MSE `0.0006841`, and Gaussian-standard-deviation MSE `0.0010840`, for partial points `(25.00, 22.72, 25.00, 25.00)`. Refitting the final classifier head on the frozen features of the same 30,000 training images raised accuracy to `0.960` and the official formula returns **100/100**; the other metrics remain unchanged, with partial points `(25.00, 25.00, 25.00, 25.00)`. The head-only step took `54.72` seconds on CPU, in addition to the original `374.31` second fit. The official evaluator logic was applied to the released validation rows; validation parameters were used only for scoring. CUDA is unavailable on this host, so a platform GPU runtime estimate cannot be measured locally. Hidden-test performance is unknown.

## Alternatives considered

- The official starter trains a small U-Net/ConvNet for denoising and classification, but its provided model does not supervise Gaussian parameters.
- Estimating parameters as the mean/std of a predicted noise residual was considered; image detail and clipping can bias those statistics, so the solution supervises the parameter head with a censored likelihood on training pairs.
- A fixed mean/std prior would not meet the per-image MSE cutoffs reliably across the released parameter ranges.

## Progressive hints

1. Convert byte pixels to `[0, 1]` and exploit the paired clean/noisy training images for denoising.
2. Train the noise classifier jointly; return a probability because the evaluator compares the returned value with 0.5.
3. The parameter fields are absent from training, but the clean/noisy pair still gives a likelihood for the latent Gaussian parameters.
4. Account for clipping: an observed 0 or 1 is an inequality on the noise sample, not an exact residual.
5. Score mean PSNR and Gaussian parameter MSE with the notebook's validation evaluator; do not feed validation parameters into training.

**One-line summary:** A shared CNN, censored-Gaussian parameter learning, and train-only frozen-feature classifier refit score **100/100 on released validation**; hidden-test performance is unknown.
