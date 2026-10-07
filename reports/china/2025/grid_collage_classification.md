# NOAI China 2025 — Grid Collage Classification

**Problem domain:** Weakly supervised binary image classification  
**Metric:** Accuracy  
**Official task page:** [SOTA checklist: Grid Collage Classification](https://checklist.sota-ai.org/problems/noai-china-2025-round-2-grid-collage-classification/) · [Bohrium task](https://www.bohrium.com/en/competitions/91782452314)

## Abridged task

Determine whether a product image is a grid collage. The 1,000 RGB training images are 256×256 and `train.csv` labels only the product category, not the collage target. The hidden validation set has 100 images and the hidden test set has 400. The task accepts one 0/1 label per image in sample order, no header, inside `submission.zip`; 1 means collage. Accuracy is the metric. The organizer statement linked from the task page controls exact formats and definitions.

## Data and license

The task's image data and `ANSWER_PATH` were unavailable in this workspace. The SOTA page lists the source license as **not stated**. Its baseline describes women's apparel and beauty/make-up product images. I did not copy organizer images, CSV files, or the baseline notebook into this repository.

## Dataset analysis and EDA

The task summary reports 1,000 RGB training images at 256×256 pixels. The annotations provide product categories but no collage labels; validation and test contain 100 and 400 images. The checklist baseline describes women's apparel and beauty/make-up items. With no images mounted locally, I could not assess class balance, collage prevalence, seam styles, image quality, or category overlap.

## Method

The candidate in [grid_collage_classification.py](../../../solutions/china/2025/grid_collage_classification.py) addresses the missing collage labels with weak synthetic supervision. Original product images are treated as weak negatives, while the script creates positive 2×2, 2×3, 3×2, 3×3, horizontal and vertical collages from images in the same product category. Random crop, uneven cell sizes, gutters, color changes and JPEG recompression make the synthetic examples less uniform. A ResNet18 learns to separate original images from these mosaics.

An optional CSV with manually annotated `id,label` pairs enables a supervised local validation path. The task organizer's own discussion recommends either manually labeling images or detecting grid seams; this implementation uses synthetic collages as the weak-label route. It writes the two required prediction CSVs and ZIP in the order of the provided annotation tables.

## Experiments and score evidence

| Result | Score | Provenance |
|---|---:|---|
| NOAI validation leaderboard | Not measured | No data, Bohrium account, or active grader was available. |
| NOAI hidden test leaderboard | Not measured | The hidden labels and final score are unavailable. |
| Candidate accuracy | Not measured | There are no target labels in the supplied training annotation; weak synthetic labels do not measure task accuracy. |

The linked Bohrium competition window is listed as closed on 20 June 2026. No organizer reference or participant score is attributed to this candidate.

## Implementation, diagnostics, and compute

The candidate trains ResNet18 on 224×224 crops and generates weak labels from synthetic collages. No actual organizer image was available, so neither task accuracy nor runtime/memory use was measured. The script can take manual image labels for a supervised local validation path; this is the more direct way to diagnose whether synthetic collages match the task distribution.

## Alternatives considered

- Manually annotate a subset of the official training images and use those labels for a supervised model.
- Detect horizontal and vertical grid seams using edge energy and changes between cells.
- Compare synthetic weak supervision against image embeddings and a small labeled set on the organizer validation split.

## Run

```bash
python solutions/china/2025/grid_collage_classification.py \
  --mode submit --train-dir /path/to/train \
  --train-csv /path/to/train.csv \
  --val-dir /path/to/val --val-csv /path/to/val.csv \
  --test-dir /path/to/test --test-csv /path/to/test.csv \
  --output-dir submission
```

Requires PyTorch, torchvision, Pillow, pandas, scikit-learn and NumPy. Default training starts from random weights and is offline. `--pretrained` may be used only when weights are already cached and permitted by the exact rules. The files are not downloaded by the script.

## Limitations and next steps

- Synthetic positives cannot fully reproduce the organizers' grid designs; they may teach seams or cell texture rather than the true collage distribution.
- Some original training photos may themselves be collages, so the weak-negative label is noisy.
- Manually annotating a few hundred official training images and then training on those labels is a strong comparison. Another candidate is a grid-seam detector using horizontal/vertical edge energy and cell-to-cell feature changes.
- Use the hidden validation leaderboard only after the local method is frozen; do not tune repeatedly against the final test labels.

## Progressive hints

1. Separate the available product-category labels from the missing collage target.
2. Inspect image layouts for repeated rectangular cells, gutters, and sharp internal seams.
3. Create weak training examples from realistic collages or label a small subset manually.
4. Validate on organizer examples and tune for accuracy; synthetic generation alone cannot establish task performance.

**One-line solution:** Train ResNet18 to distinguish product originals from category-matched synthetic mosaics; the NOAI accuracy remains unmeasured.

## Sources and reuse

- [NOAI task summary, image counts, labels and scoring](https://checklist.sota-ai.org/problems/noai-china-2025-round-2-grid-collage-classification/) — source license not stated.
- [Official IOAI article on the NOAI China Finals](https://ioai-official.org/noai-china-finals-2025-8-team-members-selected-for-chinas-national-team-for-ioai-2025/) confirms four practical tasks and A/B evaluation.
