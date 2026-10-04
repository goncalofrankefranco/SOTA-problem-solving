# 2026 Poland Selection — Reference Solutions

These notes track all 13 tasks across the three stages in the [official 2026 Poland selection repository](https://github.com/OlimpiadaAI/III-OlimpiadaAI). The [SOTA checklist](https://checklist.sota-ai.org/) is included as a companion source. Solution code and reports are collected in this repository; official statements and full organizer notebooks remain linked to the organizers' repository.

## Validation summary

| Problem | Released validation result | Estimated points |
|---|---:|---:|
| Filtry konwolucyjne | official worked notebook: MSE 0.0078 / 0.0293 | 100/100 |
| Klasyfikacja wieloetykietowa | official worked notebook: macro F1 0.886 | 100/100 |
| Szept czy krzyk | official worked notebook: balanced accuracy 1.000 | 100/100 |
| Zmiany semantyczne | official worked notebook: balanced accuracy 0.8716 | 100/100 |
| Segmentacja multispektralna | official worked notebook: mIoU 0.823 | 100/100 |
| Lokalizacja decyzji | Not measured; the image archive and model are not present in this workspace | Unverified |
| Piksele | 63.45% accuracy | 100/100 |
| Pustka | 5/5 trigger pairs recovered | 100/100 |
| Ukryte Kategorie | 0.483 mean IoU | 100/100 |
| Optymalizator malarza (Stage 2) | 0.004927 mean MSE | 100/100 |
| Drzewa decyzyjne | collection A solution under improvement | 58.8/100 prior run |
| Kolorowanie z GANem | assets not available in this workspace; solution under development | Unverified |
| Predyktor tokenów | model/data assets not available in this workspace; solution under development | Unverified |

The first five scores above are recorded in the organizers' public worked notebooks; the other scores come from local experiments against released validation data. None establishes secret-test performance. “Under development” entries are unresolved, not full-credit claims.

---

## 1. Lokalizacja decyzji — Decision Localization

**Problem domain:** Computer vision, weakly supervised object localization, saliency maps  
**Evaluation metric:** Mean Intersection over Union (IoU) between a thresholded heatmap and the annotated object mask

### Abridged statement

Given a frozen ResNet-18 binary classifier and images that contain its positive class, return a heatmap showing which image region supports the positive prediction. The heatmap is compared with the COCO object mask.

### Dataset analysis and EDA

The notebook describes 1,200 validation images from a COCO2017 subset, each resized to 224×224, along with masks and a binary ResNet-18 classifier. The hidden test set has the same size and format. Its loader uses ImageNet normalization and the model exposes a final convolutional feature map through `model.encoder`.

The supplied image archive and model checkpoint were not present in the workspace, so I could not inspect mask coverage, overlay examples, or measure a validation IoU. With those files available, the useful checks are mask area and shape distributions, image/mask overlays, positive-class logits, and whether activation peaks fall on the object or its context.

### Solution strategy

Use Grad-CAM on the last ResNet block. For feature map `A` and positive-class logit `s`, compute channel weights as the spatial mean of `∂s/∂A`; take the ReLU of their weighted feature-map sum; then bilinearly resize and normalize it to the input dimensions. The notebook evaluator thresholds returned values at 0.5.

The evaluator wraps the solution in `torch.no_grad()`, so the implementation explicitly re-enables gradients inside the function. It uses one forward pass and one gradient calculation per image and does not update model weights.

### Result and diagnostics

The implementation is in the notebook, but its IoU and score remain unverified until the Google Drive files referenced by the notebook are available. There is also a scoring-text inconsistency in the starter notebook: its prose says IoU ≤0.25 earns zero, while the executable scoring function maps mean IoU 0.191 to 0 points and 0.245 to 100 points, clamped to that range. The implementation should be judged with the executable evaluator unless the organizers clarify the prose.

### Compute and alternatives

Grad-CAM needs one forward/backward pass per image; the notebook allows GPU execution and a three-minute limit. Alternatives to compare on the released validation set are Grad-CAM++, LayerCAM, and occlusion maps. Occlusion can require many more forward passes. None of these comparisons was measured without the archive and checkpoint.

### Hints

1. Which internal model layer still preserves image location?
2. How can the positive logit’s gradient identify influential channels?
3. How should the coarse activation map be resized and normalized?

**One-line solution:** Use the positive-class gradient to weight the final convolutional maps, then upsample and normalize the Grad-CAM heatmap.

**Solution code:** [lokalizacja_decyzji.py](solutions/stage3/lokalizacja_decyzji.py)  
**Official statement and starter:** [lokalizacja_decyzji](https://github.com/OlimpiadaAI/III-OlimpiadaAI/tree/main/3_etap/lokalizacja_decyzji)

---

## 2. Piksele — Pixels

**Problem domain:** Sparse-image classification, classical ML, spatial density estimation  
**Evaluation metric:** Classification accuracy on digits 0–9

### Abridged statement

Classify a 28×28 MNIST image when only ten pixels are visible. The notebook supplies 500 labeled sparse training images, a 2,000-image validation set, and a CNN trained on complete MNIST images.

### Dataset analysis and EDA

The training archive has 500 images and the validation archive has 2,000. Every image inspected has exactly ten nonzero visible pixels. The local NPZ files contain `images` and `labels`; the notebook text also mentions an `originals` field, but it is absent from these files and the solution does not rely on it. The training label counts for digits 0–9 are `[57, 53, 43, 59, 32, 40, 45, 67, 47, 57]`; validation counts are `[180, 223, 210, 214, 190, 185, 199, 207, 184, 208]`.

The visible pixel coordinates carry most of the shape information. I compared class-conditional spatial maps and treated the input as a small point set. A direct full-image CNN prediction has a substantial input-distribution change, but local max filtering makes those points resemble thicker digit strokes and lets the supplied model contribute useful features.

### Solution strategy and experiments

The final rule combines spatial evidence with the supplied CNN:

1. **Smoothed class likelihood:** sum the visible training intensities by class and pixel location, smooth each 28×28 map with a Gaussian filter (`σ=0.9`), add a small pseudocount, and score the query’s visible pixels by log likelihood. Query intensities are square-root weighted.
2. **Blurred-mask RBF SVC:** blur the sparse image with `σ=1.5` and classify it with an RBF SVC (`C=10`).
3. **CNN feature SVC:** apply the loaded CNN to the raw, 3×3 max-filtered, and 5×5 max-filtered image. Concatenate the three 128-dimensional penultimate activations and train an RBF SVC (`C=1`). Use the 5×5-view CNN logits as a fourth score.

Standardize each ten-class score vector per image and combine them as `density + 0.4 × blurred_SVC + 1.25 × CNN_feature_SVC + 0.3 × 5x5_CNN_logits`.

| Method | Validation accuracy |
|---|---:|
| Raw-pixel RBF SVC | 36.7% |
| Raw-pixel multinomial Naive Bayes | 40.9% |
| Gaussian-smoothed spatial likelihood | 56.6% |
| Supplied CNN on raw sparse image | 19.3% |
| Supplied CNN on 5×5 max-filtered image | 44.9% |
| RBF SVC on CNN features from raw, 3×3, and 5×5 views | 59.7% |
| Spatial likelihood + blurred-mask SVC + CNN-feature SVC + CNN logits | **63.45%** |

The combined result clears the notebook’s 60% full-credit threshold. The local Python runner did not have PyTorch, so I loaded the supplied state dict and reproduced the network’s Conv/ReLU/Pool/Linear forward pass in NumPy for these experiments. I did not cross-check numerical parity against PyTorch in this runner; the reference notebook implementation uses the provided PyTorch model directly.

### Implementation and compute footprint

The classifiers are fit once, lazily, from `train_imgs` and `train_labels`, then reused for each test image. They use CPU: one RBF SVC on 500 vectors of length 784 and another on 500 concatenated 384-feature CNN representations. Each query runs three CNN views together. The NumPy experiment over the released validation set completed in under 20 seconds; the final notebook’s PyTorch runtime was not timed in this runner. The secret test set was not available.

Other approaches worth testing include point-cloud nearest neighbors, pairwise-distance features, sparse-mask augmentation, and alternate max-filter sizes. Point-cloud Chamfer k-NN reached about 50% on this validation set; pairwise displacement histograms did not beat the selected blend.

### Hints

1. Confirm how many visible pixels each example has and plot them by class.
2. Estimate where each digit class tends to place visible pixels.
3. Smooth those spatial patterns to tolerate small handwriting shifts.
4. Thicken the points, extract pretrained CNN features at several scales, and blend their scores with a spatial likelihood.

**One-line solution:** Combine class-specific spatial likelihoods with an RBF SVC over raw/dilated CNN features and the CNN’s 5×5-view logits.

**Solution code:** [piksele.py](solutions/stage3/piksele.py)  
**Official statement and starter:** [piksele](https://github.com/OlimpiadaAI/III-OlimpiadaAI/tree/main/3_etap/piksele)

---

## 3. Pustka — The Void

**Problem domain:** Model interpretability, activation analysis, concept discovery  
**Evaluation metric:** Exact recovery of five hidden `(concept, value)` pairs; 20 points per pair

### Abridged statement

Each report contains one value for each of 20 concepts. A hidden binary classifier returns positive if any of five unknown pairs appears. The classifier is unavailable, but its 10-dimensional activations, final linear weights, bias, and the report concepts are provided. Recover the five pairs.

### Dataset analysis and EDA

The release contains 5,000 reports and a `(5000, 10)` activation matrix. Recomputing `logit = activation @ w + b` gives 244 positives and 4,756 negatives. The reports contain 200 distinct observed pairs.

Filtering out any pair that occurs in a negative report leaves ten candidates. Five are true triggers; five are correlated decoys. The positive-only candidates’ mean logits separate the two groups: the five highest means are the five release answers. A set-cover-only rule is unreliable here: correlated decoys can cover all positive rows with fewer than five pairs.

| Candidate pair | Mean logit when present | Release label |
|---|---:|---|
| StarSystem = TRAPPIST-1 | 2.904 | Trigger |
| TechLevel = Crystal-Tech | 2.756 | Trigger |
| Cargo = Frozen Colonists | 2.746 | Trigger |
| GovernmentType = Anarchy | 2.684 | Trigger |
| EncounteredAnomaly = Dark Matter Cloud | 2.632 | Trigger |
| StarshipClass = Science Vessel | 2.510 | Correlated candidate |
| Destination = Outpost | 2.503 | Correlated candidate |
| Sector = Sector 005 | 2.459 | Correlated candidate |
| LastLogEntry = Shields raised as a precaution. | 2.373 | Correlated candidate |
| PrimaryExport = Alien Artifacts | 2.342 | Correlated candidate |

### Solution strategy

1. Reconstruct the model logit and binary label using the stated zero threshold.
2. Build one presence mask per `(concept, value)` pair.
3. Keep pairs that appear in at least one positive report and no negative report.
4. Rank those candidates by their mean logit on reports containing the pair; return the top five.

For the released validation set, the recovered pairs are `StarSystem = TRAPPIST-1`, `TechLevel = Crystal-Tech`, `Cargo = Frozen Colonists`, `GovernmentType = Anarchy`, and `EncounteredAnomaly = Dark Matter Cloud`. These match the release ground truth 5/5. The implementation does not read the ground-truth file.

### Implementation and compute footprint

The code uses NumPy to form 5,000×200 Boolean incidence masks and calculate conditional means. It requires little CPU time and no GPU. A fallback scoring branch keeps the function’s output length at five if a future release has fewer than five perfectly positive-only candidates.

Other plausible methods are sparse logistic regression on pair indicators, conditional logit comparisons, and set cover with a margin-based tie-breaker. The validation experiment shows why output labels alone can be ambiguous when non-trigger values co-occur with triggers.

### Hints

1. Can the given output layer reconstruct each report’s predicted label?
2. What must be true of a pair that is itself a trigger?
3. Why do several correlated pairs survive that filter?
4. Use the continuous logit margin to rank those candidates.

**One-line solution:** Keep pairs absent from every negative report, then select the five with the largest mean positive logit.

**Solution code:** [pustka.py](solutions/stage3/pustka.py)  
**Official statement and starter:** [pustka](https://github.com/OlimpiadaAI/III-OlimpiadaAI/tree/main/3_etap/pustka)

---

## 4. Ukryte Kategorie — Hidden Categories

**Problem domain:** Multi-label classification, recommender embeddings, hierarchical labels  
**Evaluation metric:** Mean set IoU; the notebook awards full points above 0.46

### Abridged statement

Learn from 12,145 256-dimensional product embeddings and their category paths. For each validation or test embedding, return the category set, regardless of order, with as many correct labels and as few extra labels as possible.

### Dataset analysis and EDA

There are 12,145 training products, 2,603 validation products, 256 features, and 1,095 distinct training labels. Every training row includes the root category `Sports & Outdoors`. The label distribution is long-tailed: 248 labels appear once, while 40 validation labels are absent from training. Paths contain 1–7 labels, with a mean length of about 4.61; most have four or five.

The embedding norms are tightly concentrated (median about 0.543; interquartile range about 0.515–0.580), so cosine neighbors are a useful similarity measure. Category paths preserve their hierarchy by list position, which makes votes by depth more appropriate than an unstructured global threshold.

### Solution strategy and experiments

Normalize training and query embeddings, retrieve the 80 highest-cosine neighbors, and weight each neighbor by `similarity⁴`. At each category depth, accumulate votes for every label and a `STOP` outcome for neighbors whose path ends before that depth. Predict the strongest category when its vote exceeds the depth-specific `STOP` vote multiplied by `[0, 1, 1, 3, 2, 1, 1]`.

| Method | Validation mean IoU |
|---|---:|
| Ridge multi-label scores, top three labels | 0.441 |
| Cosine k-NN votes, global top three labels | 0.472 |
| Depth-wise cosine votes, unadjusted stop decision | 0.472 |
| Depth-wise votes with calibrated stop bias | **0.483** |

The selected method reaches 0.483 mean IoU, above the notebook’s 0.46 full-credit threshold. The 40 validation-only categories are unseen classes; a train-vocabulary classifier cannot predict them.

### Implementation and compute footprint

The `YourSolution` class uses NumPy only and fits solely on the training embeddings and labels. Inference requires a cosine-similarity matrix for the query batch and 80-neighbor votes. It runs on CPU and stays within the notebook’s five-minute limit for the provided sizes.

Other candidates include one-vs-rest ridge/logistic models, global label thresholds, and voting for complete paths. Ridge underperformed the depth-wise neighbor approach; complete-path voting discarded useful agreement at individual hierarchy levels.

### Hints

1. Inspect path lengths and category frequencies before choosing a loss or output rule.
2. Do the embeddings support local similarity? Compare cosine neighbors.
3. Use the position in each category path as a hierarchy level.
4. How should a model decide whether to stop adding deeper categories?

**One-line solution:** Use similarity-weighted cosine neighbors to vote independently at each path depth, with a learned stop rule.

**Solution code:** [ukryte_kategorie.py](solutions/stage3/ukryte_kategorie.py)  
**Official statement and starter:** [ukryte_kategorie](https://github.com/OlimpiadaAI/III-OlimpiadaAI/tree/main/3_etap/ukryte_kategorie)

---

## Selected Stage 2 solution — Optymalizator malarza (Painter Optimizer)

**Problem domain:** Inverse graphics, differentiable rendering, numerical optimization  
**Evaluation metric:** Mean pixel MSE; 0.01 or lower earns 100 points

### Abridged statement

Each 64×64 RGB image is rendered by placing ten semi-transparent colored circles in sequence over a white background. From the rendered image alone, recover ten rows of `[x, y, radius, red, green, blue, alpha]` parameters so that the official renderer reproduces the image.

### Dataset analysis and EDA

The released NPZ has 50 training images with parameters and 10 validation images with parameters. All images are 3×64×64, and each parameter array has shape `(10, 7)` per image. The checked arrays contain no missing values. Centers fall roughly between 6.5 and 57.6 pixels, radii between about 4.9 and 16.9 pixels, RGB values in `[0, 1]`, and alpha values in `[0.5, 1]`. I inspected sample renders and parameter histograms; overlapping circles make the inverse problem underdetermined from any one local edge, while the exact renderer gives a strong image-level objective.

The renderer applies a sigmoid circle mask with sharpness 20, then alpha-composites each circle in its given order. At that sharpness the mask changes over a very narrow band, so direct optimization from arbitrary parameters often has weak gradients and gets stuck.

### Solution strategy and experiments

Optimize all 70 parameters against the target image using the analytic gradient of the renderer and Adam. Use continuation: begin with a smooth mask and increase sharpness through `0.12, 0.2, 0.35, 0.6, 1, 2, 4, 8, 20`. This gives the centers and radii useful gradients early, then fits the official sharp renderer at the final stages. Each stage runs 100 Adam steps at learning rate 0.15; parameters are clipped to their valid ranges.

The solver uses up to six deterministic random initializations and keeps the best render, stopping early once one image has MSE ≤0.0075. It does not train on `y_train` or use the supplied parameter labels. A single start was unreliable; on some validation images its MSE exceeded 0.01, while the bounded multistart search reduced every validation image below 0.0072. An exploratory SciPy L-BFGS-B implementation was not kept because SciPy is not on the task’s allowed-library list.

### Result and implementation diagnostics

| Released validation result | Value |
|---|---:|
| Mean MSE | **0.004927** |
| Score under the notebook rule | **100 / 100** |
| Worst per-image MSE | 0.007130 |
| Runtime for all 10 images | 56.7 seconds on CPU |

The self-contained [solver](2_etap/optymalizator_malarza/solution.py) uses NumPy for optimization and returns a PyTorch tensor with shape `(10, 7)`. The local runner did not have PyTorch, so the released validation metric was computed with a NumPy reproduction of the official renderer formula and a small tensor compatibility shim; the official notebook evaluator itself was not run here. The held-out test set is hidden, so 100 points are established on the released validation set, not guaranteed on the private test set.

The per-image loop has at most six starts and took under one minute for the released validation images. Even if every hidden image needed all six starts, the observed slowest image gives a rough upper estimate below the five-minute CPU limit.

### Hints

1. What exact function maps the 70 circle parameters to the image?
2. Why do the sharp masks make random-start gradients unreliable?
3. Can a smoother version of the same renderer guide the parameters toward a useful region?
4. How can deterministic restarts reduce the risk of a poor local minimum?

**One-line solution:** Fit the 10 circles with analytic-gradient Adam, gradually increasing mask sharpness and selecting the best of up to six starts.

**Official task and starter notebook:** [Optymalizator malarza](https://github.com/OlimpiadaAI/III-OlimpiadaAI/tree/main/2_etap/optymalizator_malarza)  
**Solution code:** [solution.py](2_etap/optymalizator_malarza/solution.py)

---

## Stage 2 selection note

I first explored **Drzewa decyzyjne**. Its solution scored 58.8/100 on released collection A, below the full-credit criterion, so I moved to **Optymalizator malarza**, which reached the published 100-point MSE threshold on its released validation set. The decision-tree experiment is not presented as a selected solution.

---

## Stage 1 — Official worked solutions

The organizers publish a full worked notebook for each Stage 1 problem. The EDA notes, implementation, and released-validation numbers below summarize those notebooks; they are attributed to the organizers and were not rerun in this workspace. Use the links for the complete, executable code.

### 1. Filtry konwolucyjne — Convolutional Filters

**Problem domain:** Image restoration, inverse problems, linear least squares  
**Evaluation metric:** MSE, with 70 points for the full-resolution filter and 30 for upsampling plus filtering

**Abridged statement:** Recover a shared spatial filter that reverses image corruption, then reconstruct a full-resolution image from a corrupted half-resolution image. The dataset provides 150 training and 50 validation triples of 256×256 RGB images, their corrupted copies, and 128×128 corrupted copies.

**EDA and experiments:** The corrupted full-resolution images show a consistent blur and dark border, suggesting a convolutional degradation. The low-resolution images lose spatial detail rather than simply representing a clean 2× resize. Kernel sizes 2–10 were explored; odd sizes tended to outperform the adjacent even sizes, with 9×9 a strong final choice. Nearest-neighbor, bilinear, and bicubic upsampling gave validation MSEs 0.041620, 0.036393, and 0.040749, respectively, after the first filter.

**Solution:** A single convolution is linear in its kernel weights. Accumulate the window Gram matrix `XᵀX` and target product `Xᵀy` over images, then solve the small least-squares system instead of storing all windows. For the second task, insert zeros between low-resolution pixels, apply a learned 7×7 reconstruction filter, then apply the 9×9 restoration filter.

**Official result:** MSE 0.0078 for the 9×9 filter and 0.0293 for the upsampling task; 70/70 + 30/30 = **100/100** on the released validation set. The organizer also reports that the linear solve is much faster than iterative training.

**Hints:** (1) Inspect what the corruption does at full and half resolution. (2) Which visual artifacts suggest blur or information loss? (3) Is the output linear in the filter weights? (4) Can you accumulate least-squares statistics without building a huge patch matrix?

**One-line solution:** Solve the shared convolution kernel by least squares, and use zero insertion plus a learned 7×7 reconstruction kernel before the 9×9 restoration kernel.

**Official solution notebook:** [Filtry konwolucyjne — opracowanie](https://github.com/OlimpiadaAI/III-OlimpiadaAI/blob/main/1_etap/1_filtry_konwolucyjne/filtry_konwolucyjne_opracowanie.ipynb)

### 2. Klasyfikacja wieloetykietowa — Multi-label Classification

**Problem domain:** Computer vision, transfer learning, multi-label classification  
**Evaluation metric:** Macro F1 over 10 clothing labels

**Abridged statement:** Classify which of 10 clothing types appear in 168×168 grayscale images. There are 6,318 labeled training images, 702 validation images, and 780 hidden-test images.

**EDA and experiments:** Pixel intensities range from 0 to 255, with mean 17.0916 and standard deviation 51.9870, so most of the image background is dark. Pairwise label co-occurrence is modest (about 32% conditional probability), with no strong pair that can substitute for visual recognition. A simple CNN trained from scratch reached F1 0.699 in the organizer’s report, while a naïve baseline reached 0.545.

**Solution:** Replicate each grayscale image to three channels, fine-tune a pretrained torchvision ResNet (the final reported model uses ResNet18), and train the 10-logit head with a multi-label loss. Search learning rate and weight decay over short runs, and tune one decision threshold per label on validation predictions instead of using 0.5 for every class.

**Official result:** Macro F1 **0.886**, above the 0.87 full-credit threshold, for **100/100** on the released validation set. The organizer’s report used ten short hyperparameter trials and selected a ResNet18 configuration.

**Hints:** (1) Check whether pixels are grayscale and how much of each image is background. (2) Inspect label frequencies and co-occurrence before choosing a model. (3) Why might a pretrained RGB backbone still help on grayscale inputs? (4) Should every label share the same probability threshold?

**One-line solution:** Fine-tune a pretrained ResNet18 for 10 independent binary outputs, then calibrate the per-label thresholds on validation data.

**Official solution notebook:** [Klasyfikacja wieloetykietowa — opracowanie](https://github.com/OlimpiadaAI/III-OlimpiadaAI/blob/main/1_etap/2_klasyfikacja_wieloetykietowa/klasyfikacja_wieloetykietowa_opracowanie.ipynb)

### 3. Szept czy krzyk — Whisper or Scream

**Problem domain:** Audio classification, signal processing, spectrogram CNNs  
**Evaluation metric:** Balanced accuracy across the audio classes

**Abridged statement:** Classify short audio signals as normal speech, whisper, or scream. Samples have variable length; the longest training recording contains 80,000 samples.

**EDA and experiments:** Waveform lengths vary, making raw fixed-length inputs awkward. The worked solution compares a raw-waveform recurrent model with a time-frequency representation. A quantized raw-signal BiGRU reached balanced accuracy 0.8873 (74 points), while the spectrogram CNN reached 1.0000.

**Solution:** Convert each recording to a log-mel spectrogram and train a convolutional classifier. The time-frequency representation exposes loudness and spectral-shape differences between whisper, ordinary speech, and screams while handling variable durations with padding or resizing. The organizer’s notebook supplies the feature extraction, model, and evaluation code.

**Official result:** Balanced accuracy **1.0000**, for **100/100** on the released validation set. The alternative raw-signal BiGRU result is a useful reminder that a more direct representation was less reliable here.

**Hints:** (1) Compare duration and amplitude distributions by class. (2) What frequency structure distinguishes a whisper from a scream? (3) Can an audio clip be represented as an image over time and frequency? (4) Compare that representation with a raw-signal recurrent baseline.

**One-line solution:** Train a CNN on log-mel spectrograms; the organizer’s validation run reached perfect balanced accuracy.

**Official solution notebook:** [Szept czy krzyk — opracowanie](https://github.com/OlimpiadaAI/III-OlimpiadaAI/blob/main/1_etap/3_szept_czy_krzyk/szept_czy_krzyk_opracowanie.ipynb)

### 4. Zmiany semantyczne — Semantic Change

**Problem domain:** NLP, historical word embeddings, representation alignment, binary classification  
**Evaluation metric:** Balanced accuracy

**Abridged statement:** Given historical word-vector representations from two periods, predict whether a word changed meaning between 1900 and 1990.

**EDA and experiments:** The worked notebook evaluates 832 labeled examples and emphasizes that raw cosine comparisons across independently trained embedding spaces are not directly comparable. Its features compare neighborhoods and similarity structure across time, after aligning the spaces. Candidate signals include first- and second-order similarities, anchor-neighbor ranks, and changes in nearest-neighbor ordering.

**Solution:** Align the 1990 embedding space to the 1900 space with orthogonal Procrustes. Build multiple similarity and neighbor-rank-shift features using frequency-based and sampled anchor words. Train base classifiers, including histogram gradient boosting and logistic regression, then stack out-of-fold probabilities in a logistic meta-classifier and tune the final decision threshold.

**Official result:** Balanced accuracy **0.8716**, reported as **100/100** on the released validation set.

**Hints:** (1) Why can embeddings from different training periods be rotated relative to each other? (2) Align the spaces before comparing vectors. (3) Do neighborhood ranks preserve useful information beyond cosine distance? (4) Could several complementary classifiers be calibrated together?

**One-line solution:** Orthogonally align the embedding spaces, engineer neighborhood and rank-shift features, then stack calibrated classifiers.

**Official solution notebook:** [Zmiany semantyczne — opracowanie](https://github.com/OlimpiadaAI/III-OlimpiadaAI/blob/main/1_etap/4_zmiany_semantyczne/zmiany_semantyczne_opracowanie.ipynb)

### 5. Segmentacja multispektralna — Multispectral Segmentation

**Problem domain:** Remote sensing, dimensionality reduction, semantic segmentation  
**Evaluation metric:** Mean IoU over four terrain classes; the input must be reduced from 12 bands to 3

**Abridged statement:** Segment 30×30 satellite patches into water, land, vegetation, and industrial terrain. The dataset has 48 training and 16 validation images, each with 12 spectral channels.

**EDA and experiments:** There are no missing values. Training pixels are distributed across the four classes as `[12049, 11348, 13384, 6419]`; validation distribution differs, especially for land. Several channels are highly correlated (often above 0.95), and visual inspection shows redundant bands. The worked solution reduces all 12 bands to 3 standardized principal components, augments the small image set with flips/rotations, and trains a compact pixelwise CNN. Its model uses four convolutions with batch normalization and ReLU, cross-entropy, AdamW, and a long OneCycleLR schedule.

**Solution:** Fit PCA on the training pixels, apply the same transform to each sample, and train the CNN on the three components. Keep spatial resolution intact so the network can use local boundaries; choose the best validation checkpoint by mIoU.

**Official result:** mIoU **0.823** with 3 channels, reported as **100/100** after 300 epochs (about 8 seconds in the organizer’s recorded run). A larger attention encoder-decoder scored mIoU 0.796 and 99 points, so the simpler model was better on this small dataset.

**Hints:** (1) Compare bands and label distributions before training. (2) Which channels are redundant? (3) Can PCA preserve most spectral information in three components? (4) How can flips and rotations help when there are only 48 training patches?

**One-line solution:** Standardize the 12 bands, project them to three PCA components, then train an augmented four-layer segmentation CNN.

**Official solution notebook:** [Segmentacja multispektralna — opracowanie](https://github.com/OlimpiadaAI/III-OlimpiadaAI/blob/main/1_etap/5_segmentacja_multispektralna/segmentacja_multispektralna_opracowanie.ipynb)
