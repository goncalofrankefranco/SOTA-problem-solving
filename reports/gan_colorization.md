# Kolorowanie z GANem — StyleGAN Face Colorization

**Stage:** 2, Poland 2026 selection  
**Problem domain:** Computer vision, image colorization, GAN latent inversion  
**Evaluation metric:** Weighted PSNR and LPIPS score: 25% PSNR points over 22–26 dB and 75% LPIPS points over 0.15–0.11 (lower is better). The combined result is rounded to an integer from 0 to 100.

## Abridged statement

Given a grayscale 256×256 face image and the provided pretrained StyleGAN2 face generator, return a plausible RGB colorization. There is no training split. The final model must use the supplied generator and complete evaluation of the secret set within five minutes on GPU, using only the listed libraries (`torch`, `numpy`, `torchvision`, and `Pillow`).

## Dataset analysis and EDA

The official validation archive is a tar file of aligned grayscale/RGB face photographs. Its public copy is at `/tmp/poland-2026-official/2_etap/kolorowanie_z_ganem/data/val.data`. It contains 500 JPEG files: 250 `GT` color images and 250 matching `GRAY` images. This conflicts with the notebook prose, which says there are 500 pairs; the notebook loader itself determines the pair count from the number of `GT` files. The inspected images are FFHQ-style 256×256 face crops with varied skin tones, hair colors, backgrounds, and lighting.

Replicating each grayscale channel as RGB gives validation MSE 0.02771535 in the notebook's `[-1,1]` range and aggregate PSNR 21.5934 dB, below the PSNR scoring floor. It preserves structure but discards all chroma. LPIPS could not be measured in this runtime.

## Solution strategy and experiments

Color recovery is ambiguous, so the solution uses the provided face generator as a prior. It maps random latent vectors through the generator's mapping network to estimate the center and scale of W space. For each input, it optimizes a shared W vector against grayscale reconstruction, then optimizes layer-wise W+ codes for better local alignment. The objective combines full-resolution grayscale error, downsampled structure error, edge error, and a penalty against drifting too far from the generator's latent distribution.

The starter's `style_to_image` synthesizer draws fresh random noise on every call. The solution instead supplies the generator's registered noise maps explicitly, making each optimization step deterministic. Once inverted, the generated face contributes color only: its luminance is removed and a confidence-scaled chroma residual is added to the original grayscale image. This keeps the observed input luminance intact while using the generator to propose plausible skin, hair, and background colors.

An independently published participant notebook was also inspected: it estimated mean W from 2,000 samples and used two Adam updates against grayscale MSE, with a small latent regularizer. It records no validation metrics. The implementation here uses fixed noise, an explicit W-to-W+ refinement, multiple image scales, edge matching, and input-luminance preservation.

## Result and diagnostics

**Score: unverified.** The public validation archive was available for inspection, but the generator checkpoint is fetched from a Google Drive URL that returns HTTP 403 in this environment. PyTorch, torchvision, and a GPU are also absent, so I could not run the official evaluator, measure LPIPS, or time inference. The grayscale baseline measurement above is the only measured score-related result; no full-credit claim is made. Secret-test performance remains unknown.

The code is designed to fit within the five-minute GPU limit by processing at most eight images together and using 18 latent-optimization updates per batch. This runtime was not measured, so it still needs to be checked on the competition GPU. If that limit is exceeded, reduce the W and W+ update counts in `_invert` before changing the method.

## Implementation and compute footprint

The implementation uses only the supplied StyleGAN generator and PyTorch. Generator weights stay frozen; gradients update only one latent code per input image. Fitting estimates W statistics once. Prediction performs 8 shared-W and 10 W+ updates, then one deterministic synthesis per image batch. No external pretrained model, internet access, or paired training data is used.

The solution code is [gan_colorization.py](../solutions/stage2/gan_colorization.py). It is written as the `YourModel` class from the official notebook and can be copied into that notebook's solution cell. The official statement, starter notebook, and generator implementation are at [Kolorowanie z GANem](https://github.com/OlimpiadaAI/III-OlimpiadaAI/tree/main/2_etap/kolorowanie_z_ganem).

## Hints

1. Which part of the supplied generator contains a prior for realistic faces?
2. How can the latent code be adjusted so that a generated face has the input's grayscale structure?
3. Why should the generator's noise maps remain fixed while optimizing a latent code?
4. How can generated color be added while preserving the input's measured luminance?

**One-line solution:** Invert each grayscale face into the generator's W/W+ latent space with fixed noise, then transfer a confidence-scaled generated chroma residual onto the original luminance.
