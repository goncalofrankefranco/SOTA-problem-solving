# Data Prototypes

## Problem, domain, and metric

- **Domain:** dataset compression and nearest-neighbor classification in an embedding space.
- **Task:** produce exactly 150 labeled embeddings from the released training embeddings. Each hidden sample receives the label of its closest prototype under Euclidean distance.
- **Metric:** validation/test accuracy. Scores are 0 at accuracy ≤0.40, 100 at accuracy ≥0.68, and linear between those thresholds. Evaluation must fit within five minutes on GPU.
- **Abridged statement:** reduce a 40,000-example CIFAR-100 training representation to a 150-vector set that retains as much classification accuracy as possible.

## Released data and EDA

The [official task notebook](https://github.com/OlimpiadaAI/II-OlimpiadaAI/tree/main/3_etap/4_prototypy_danych) describes 40,000 training images and 10,000 validation images (about 400/100 examples per class) from CIFAR-100. Images are resized to 224×224 and normalized with ImageNet mean/std. The provided MobileNetV3 Small returns a 1,000-dimensional output embedding. Required output is 150 `float32` vectors and 150 labels in `[0,100)`.

The notebook downloads `train_val.gz`, a 10.3 MB MobileNet checkpoint, and validation embeddings from Google Drive. None of those data/checkpoint assets are present in the available workspace, so embedding-level EDA and a local candidate score cannot be computed. The notebook's saved baseline output is available and gives useful evidence about the task's difficulty.

## Experiments and score evidence

| Method | Validation accuracy | Official score | Evidence source |
|---|---:|---:|---|
| Starter: random 1–2 training embeddings per class | 18.34% | 0/100 | Saved output in the official/mirrored notebook |
| Different random starter draw | 17.97% | 0/100 | Saved output in the mirrored notebook |
| Classwise centers with one extra cluster for 50 high-spread classes | Not measured | Unknown | Candidate below; required embeddings are missing |

The two random scores are the only measured validation results. They are far below the 40% score floor. There is no local evidence that the candidate reaches the 68% full-credit threshold, and the hidden-test score is unknown.

## Selected candidate

[data_prototypes.py](../../../solutions/2025/stage3/data_prototypes.py) assigns every class one prototype at its training centroid, then allocates one additional center to each of the 50 classes with the largest ratio of within-class radius to nearest competing centroid distance. Those classes receive a deterministic farthest-first, two-center k-means fit; the other 50 retain one center. It reads only `train_embeddings` and `train_labels` and returns the required shapes and dtypes.

This is a train-only candidate, not a validation-verified solution. Its rationale is to spend the limited extra 50 prototypes on classes whose examples are broad relative to their separation from other class centers. The missing embedding archive blocks both validation comparison and runtime measurement in this workspace.

## Alternatives and caveats

The starter's random sample should be replaced by classwise cluster representatives. Other train-only choices include medoids, allocating extras by an internal train split's class-confusion matrix, or directly optimizing prototype locations against a train-only nearest-prototype loss. Those alternatives need the missing embedding assets before they can be compared. The candidate's radius/separation heuristic may misallocate capacity, and its accuracy should not be reported as a score until it is run in the official evaluator.

## Progressive hints

1. The starter wastes accuracy by picking prototypes at random.
2. Always allocate at least one center per class so no CIFAR-100 label disappears.
3. Use cluster representatives rather than raw random points to reduce within-class distance.
4. Allocate extra centers to broad or confusable classes using train-only statistics, then validate against the released held-out embeddings.

**One-line summary:** the notebook's random baseline scores 0/100 (18.34%); a train-only classwise k-means candidate is saved, but the absent CIFAR embeddings prevent local scoring and hidden-test accuracy is unknown.
