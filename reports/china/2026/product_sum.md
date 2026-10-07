# NOAI China 2026 — Product · Sum

**Problem domain:** Computer vision, weakly supervised digit recognition, multi-target prediction  
**Evaluation metric:** Accuracy over both outputs: `(correct sums + correct products) / 2,000` for each 1,000-image split; maximum 1.0.

## Abridged statement

Each grayscale image contains four horizontally concatenated handwritten digits. Predict their arithmetic sum and product. Training labels disclose only those two aggregate values, not the four digit identities. The SOTA task page describes 10,000 labeled training images and separate 1,000-image validation and test sets. This is a paraphrase; consult the official statement for the exact file and submission rules.

## Dataset analysis and EDA

The official task summary specifies 28×112 grayscale images and four 28-pixel-wide digit regions. It reports `train_labels.csv` with ID, sum, and product, plus 10,000 training images; validation and test each contain 1,000 images. No assets are mounted in this workspace, so no image grid, label histogram, digit-distribution estimate, or validation score could be computed locally.

The weak supervision creates an unusual learning signal: many ordered digit tuples can share one sum/product pair, and a zero product implies at least one zero without revealing its position. However, the targets are deterministic functions of the four digits. The image width and fixed digit count make it possible to model each segment while summing the probabilities of every digit tuple consistent with a training label.

## Experiments and solution strategy

No competition-data experiment was run. The candidate in [the solution script](../../../solutions/china/2026/product_sum.py) uses a shared CNN encoder on each 28×28 segment and outputs a ten-class distribution per digit. For each labeled image, it enumerates all 10,000 ordered digit tuples, groups their probabilities by `(sum, product)`, and maximizes the likelihood of the observed aggregate pair. At inference, it aggregates the joint digit probabilities into pair probabilities and selects the most likely pair, rather than separately rounding two unconstrained regressors.

The script has a local holdout mode for the 10,000 labeled samples. With data access, inspect sum and product accuracies separately, compare exact marginal decoding against independently decoded digits and direct two-target regression, and tune training duration and crop/normalization choices. Since the labels are aggregates, a low training loss alone does not establish digit recognition or hidden-set accuracy.

## Score evidence and limits

| Result | Accuracy | Provenance |
|---|---:|---|
| This candidate | Not measured | The image/label files and Bohrium grader are unavailable locally. |
| Organizer baseline B | 0.0440 | Reported by SOTA. |
| Scientific Committee reference B | 0.9600 | Reported by SOTA; not reproduced by this candidate. |

The 0.9600 value is a reference benchmark, not our score. No full-mark or optimality claim is made.

## Implementation and diagnostics

The script expects training assets at `/bohr/train-rppd/v1/train_labels.csv` and `/bohr/train-rppd/v1/train_images/`, and evaluation folders `val/` and `test/` under `DATA_PATH` (default `/bohr`). It produces `submission.zip` with `submission_val.csv` and `submission_test.csv` at the archive root, each with `id,sum,product` headers. `--mode validate` holds out labeled rows and reports exact accuracy for each output.

No local model run or task diagnostic is available yet. The organizer baseline uses a small CNN trained by direct L1 regression, which is a useful comparison point. This candidate's constrained probability calculation is more aligned with the problem's latent digit structure, but needs measured holdout accuracy and the official practice grader before any quality claim.

## Compute and footprint

The candidate trains a compact CNN using PyTorch and can use CUDA when available. It does not download pretrained weights or external data. The task page specifies a Tesla L20 GPU and a 25-minute train/inference cap. Training time and memory were not measured here; the model's 10,000-tuple decoding has a fixed, modest computation per image.

## Alternatives to compare

- Direct sum/product regression as in the released baseline.
- Weakly supervised OCR with latent four-digit tuples and an exact aggregate-label loss, as implemented here.
- Generate synthetic digit drawings for pretraining, if allowed by the organizer's rule permitting contestant-made drawings; then fine-tune only on the official data.
- Add explicit constraints from the product and sum to decode a digit sequence, or use an ensemble of independent per-digit classifiers and aggregate decoders.

## Progressive hints

1. What is the input width divided by the number of digits?
2. Why are four independent target values unnecessary if the task asks only for their sum and product?
3. For each training `(sum, product)` pair, enumerate the digit tuples that could have produced it.
4. Train per-position digit probabilities using the total probability mass of compatible tuples, then choose the most likely aggregate pair at inference.

**One-line solution:** Recognize four digit distributions and sum the probability of every digit tuple that matches the observed sum/product pair.

## Sources and reuse

- [SOTA task page and English baseline translation](https://checklist.sota-ai.org/problems/noai-china-2026-round-2-product-sum/)
- The page links the official English/Chinese statements, baseline notebook, Bohrium task page, and source files.

SOTA reports the source license as **not stated**. The report paraphrases the task, and the implementation is independently written. The organizer images, labels, statement, and notebook are not copied into this repository.
