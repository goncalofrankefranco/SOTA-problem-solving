# NOAI Singapore 2026 Final Section 1 — derived answer key

This is a reasoned key based on the [released assessment PDF](https://aisingapore.org/wp-content/uploads/2026/04/NOAI-2026-Final-Assessment-Section-1-MCQs-Google-Forms.pdf), not an organizer answer key. `?` means no answer is supported uniquely by the released wording. `*` marks a likely intended answer with a caveat, or a forced guess where the released item supports no option.

| Question | Choice | Question | Choice |
|---:|:---:|---:|:---:|
| 1 | B | 11 | D* |
| 2 | B* | 12 | A* |
| 3 | B | 13 | A |
| 4 | D | 14 | B* |
| 5 | C* | 15 | C |
| 6 | D | 16 | B |
| 7 | C | 17 | A |
| 8 | B | 18 | A* |
| 9 | B | 19 | A |
| 10 | A | 20 | D* |

## Caveats

- **Q2:** B is appropriate when switching L2 boosting to L1 includes both sign pseudo-residuals and a median leaf correction. Some simplified gradient-boosting implementations only change pseudo-residuals, so the expected convention should be stated.
- **Q5:** C is the likely substantive intended answer: AdaBoost should reduce weights for correctly classified examples and increase weights for incorrectly classified examples. A also identifies a threshold tie edge case, while line D's update gives the wrong relative multiplier for the shown half-log alpha (`exp(alpha)` instead of `exp(2*alpha)`). The item is flawed and has multiple defensible defects, so this is not a unique key.
- **Q11:** D (color jitter on grayscale X-rays) is likely intended. Center-cropping can also remove diagnostically relevant regions, so the code description does not establish a single critical preprocessing flaw.
- **Q12:** A is likely intended for odd spatial dimensions, where floor downsampling can lose a row or column. The option's explanation blames the transposed convolution; with `kernel_size=2, stride=2` it doubles its input exactly. D depends on an unstated encoder padding choice.
- **Q8:** B is the likely issue for poor evaluation behavior: the running variance should use the unbiased batch-variance estimate as PyTorch BatchNorm does. Detaching updates is good hygiene but does not by itself explain the described numerical difference.
- **Q14:** The described order freezes the backbone, replaces the classifier head, and then constructs the optimizer. The new head is trainable and can be optimized, so none of the listed fault claims is supported. B is included only as a forced, very-low-confidence guess because the assessment has no penalty for wrong answers; it is not technically justified by the released snippet.
- **Q18:** A is likely intended because BOS and EOS consume two positions, leaving `max_len - 2` positions for content. The exact eight-word example fits exactly, so it does not actually truncate.
- **Q20:** D is the closest offered choice and may be the intended response, but the loss curves establish overfitting without establishing a high learning rate or catastrophic forgetting.

No score is measured because no official answer key or public scoring harness accompanies the PDF.
