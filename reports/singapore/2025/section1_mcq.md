# NOAI Singapore 2025 — Section 1: Multiple-Choice Questions

**Problem domain:** AI and ML fundamentals: classical machine learning, deep learning, computer vision, and NLP  
**Evaluation metric:** Exact answer-key matches across 20 questions. The official PDF publishes one answer per item; it does not state a separate per-question point weight or provide a submission scorer.  
**Task value:** Section 1 of the NOAI 2025 final. The linked task page describes 20 four-option questions.

## Abridged statement

Select one answer (A–D) for each of 20 questions about AI and machine-learning concepts. The published official PDF includes an answer key.

## Data analysis

This is a fixed knowledge assessment rather than a dataset task. The questions cover:

- Classical ML: PCA, bagging, reinforcement learning, underfitting, and boosting.
- Optimization and deep learning: Adam, vanishing gradients, early stopping, residual connections, learning-rate schedules, and pooling.
- Computer vision: YOLO, region proposals, depth-wise separable convolution, and instance segmentation.
- NLP: attention, BERT, beam search, the CLS representation, and causal masking.

There is no train/validation split, feature preprocessing, or model fitting. I transcribed the official choices and wrote short independent rationales in the answer file without copying question stems.

## Experiments

The 20 selected options were checked item by item against the answer key in the official PDF. All 20 match. This establishes **20/20 answer-key consistency**; it is not a score returned by an organizer platform.

## Solution

The answer key and concise explanations are in [section1_mcq_answers.py](../../../solutions/singapore/2025/section1_mcq_answers.py). The file also has a small exact-match counter. The key is `B A B C A A B B B B B C C B B B B B B A` in question order.

## Score evidence and limits

**Measured result:** 20/20 items against the published key. The official answer PDF and the SOTA task page establish the question count and key; no online scoring endpoint, independent test, or contest submission result was found. Because the source does not state a mark allocation for these 20 questions, I report the answer count rather than claiming a contest score such as 20 marks.

## Implementation and diagnostics

The file contains an answer mapping and a one-line rationale for each item. A local key self-check returns 20/20. The rationale list covers the underlying concepts and is useful for reviewing the choices, but the exact-match score comes from the published key.

## Compute and footprint

No training or external data are needed. The answer checker is CPU-only and takes negligible time and memory.

## Alternatives considered

- Guessing options provides no reliable way to maximize exact matches.
- Reproducing all question text in this repository is unnecessary; the official PDF is linked as the source and the task license is not stated.

## Progressive hints

1. Group questions by area: ML, optimization/deep learning, vision, and NLP.
2. For each option, identify the algorithm's defining purpose rather than a nearby application.
3. Check that distinctions such as bagging versus boosting, YOLO versus region proposals, and encoder versus decoder masking are preserved.
4. Compare the completed choices to the published answer key before claiming a score.

**One-line summary:** The answer file matches the published key on **all 20 questions**; no organizer-submitted score is available.

## Sources and reuse

- [SOTA task summary](https://checklist.sota-ai.org/problems/noai-singapore-2025-final-mcq/)
- [Official NOAI 2025 MCQ PDF with answers](https://noai.aisingapore.org/wp-content/uploads/2026/09/NOAI-2025-MCQ-Assessment-Questions.pdf)
- [Official NOAI learning-resources page](https://noai.aisingapore.org/learning-resources/)

The task page says the source does not state a license. This repository contains the answer choices and original short rationales, not a copy of the question PDF.
