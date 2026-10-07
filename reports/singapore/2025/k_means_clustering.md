# NOAI Singapore 2025 — K-Means Clustering

**Problem domain:** Unsupervised learning, clustering, algorithm implementation  
**Evaluation metric:** Return final centroids rounded to four decimal places; the task is worth 25 marks. No public automated grader or hidden test set was found.

## Abridged statement

Implement K-means for a list of coordinate vectors, a requested cluster count, initial centroids, and a maximum iteration count. Assign each point to its nearest centroid, replace each nonempty cluster's centroid with its member mean, retain a previous centroid for an empty cluster, and stop at convergence or the iteration limit.

## Data analysis

The fixed example has six two-dimensional points. Three lie around `x = 2` and three around `x = 10`; the supplied starting centroids `(2, 2)` and `(10, 1)` already seed the two visible groups. This is a small unlabeled toy set, not a train/test dataset. There are no missing or categorical values.

## Experiments

The released example was run with `k = 2` and `max_iterations = 10`. Both the local implementation and the official notebook return:

```text
[(2.0, 7.3333), (10.0, 2.0)]
```

The first center is the mean of the three points with first coordinate 2; the second is the mean of the three points with first coordinate 10. No empty cluster occurs in this example.

## Solution

The implementation is [k_means_clustering.py](../../../solutions/singapore/2025/k_means_clustering.py). It uses vectorized squared distances and `argmin` assignment, updates each nonempty centroid with a mean, leaves an empty centroid unchanged, and rounds only the returned centers.

## Score evidence and limits

**Measured result:** Exact centroid match to the output of the official solution notebook on the released example. The task is worth 25 marks, but no public grader or further scored cases were located. The result supports a full-credit implementation for the released example; it is not an organizer-issued `25/25` score.

## Implementation and diagnostics

Tie assignments go to the first centroid, matching NumPy's deterministic `argmin` behavior. Convergence is checked with absolute tolerance `1e-12`; otherwise, iteration stops at the requested limit. Internal centers retain full precision, and only final values are rounded. Input shape checks catch mismatched dimensions and an invalid centroid count.

## Compute and footprint

The method uses CPU NumPy arrays and a few matrix-sized distance calculations. The supplied six-point case runs in milliseconds; no GPU or training framework is needed.

## Alternatives considered

- Random initialization is unsuitable because the task explicitly supplies starting centroids.
- Reseeding empty clusters can produce a different answer; retaining the existing center is the statement's suggested policy and matches the reference behavior.
- Rounding at every iteration can alter later assignments on finely balanced data; rounding only the result follows the stated output rule while preserving standard K-means updates.

## Progressive hints

1. Use the supplied centroids as-is and compute each point's distance to each one.
2. Assign by the smallest distance; ties can deterministically go to the first centroid.
3. Take coordinate-wise means for nonempty clusters and keep the previous center when a cluster is empty.
4. Stop on unchanged centers or after `max_iterations`; round the final centers to four decimals.

**One-line summary:** The supplied example converges to **`[(2.0, 7.3333), (10.0, 2.0)]`**, matching the official notebook.

## Sources and reuse

- [SOTA task summary](https://checklist.sota-ai.org/problems/noai-singapore-2025-final-k-means-clustering/)
- [Official NOAI 2025 programming questions and solution notebooks](https://github.com/AISGNUSNOAI/NOAI-2025)
- [Official NOAI learning-resources page](https://noai.aisingapore.org/learning-resources/)

The task source does not state a license. It specifies Python 3.9 standard library and NumPy. This report paraphrases the statement, and the code is independently written rather than copied from the official solution notebook.
