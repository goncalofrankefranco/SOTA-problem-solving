# NOAI Singapore 2026 — The Attention Mechanism (Building the “Brain”)

**Problem domain:** NLP, Transformers, deep learning

**Evaluation metric:** 35 marks total: 10 for scaled dot-product attention, 10 for multi-head attention, 5 for positional encoding, 5 for the classifier, and a 5-mark causal-mask bonus. There is no released text-classification score or public grader result.

## Abridged statement

Implement a Transformer encoder's building blocks in PyTorch: scaled dot-product attention with optional masking, multi-head attention, sinusoidal positional encoding, an encoder layer and a small classification head. Implement a lower-triangular causal mask for the bonus. The notebook also has a short written question about why attention scores are scaled.

## Data analysis

The task releases no text corpus, labels, pretrained checkpoint, or accuracy benchmark. Its checks use synthetic tensors. The attention example uses query, key, and value arrays shaped `(2, 8, 10, 64)`. The multi-head example uses a `(2, 10, 512)` input with eight heads, while the classifier example draws token IDs from a 30,000-token vocabulary and checks for two output logits per sample. The notebook masks the final three keys in the attention check and the final two tokens in the classifier check. These checks assess tensor shapes and mask behavior, not a trained model's predictive quality.

## Experiments

The recorded local PyTorch checks passed for attention output and weight shapes, row-normalized weights, masking of future positions, multi-head output shape, positional encoding with an odd feature width, classifier output shape, and the lower-triangular causal mask. No accuracy or numeric mark total can be computed from synthetic inputs. The official notebook examples were inspected, not executed as a reference run.

## Solution

[The implementation](../../../solutions/singapore/2026/attention.py) projects queries, keys, and values, splits them into heads, computes `softmax(QKᵀ / sqrt(d_k))`, applies the weights to values, joins heads, and applies the output projection. It implements residual attention and feed-forward blocks, sinusoidal positions, and a mean-pooled classification head. The position encoding uses the standard exponential frequency formula and safely handles odd `d_model` values. The causal mask is a broadcastable lower-triangular keep-mask.

## Score evidence and limits

**No mark score was measured.** The 35 marks are the official maximum, not a measured result. The recorded evidence consists of local shape and mask assertions; there is no released grader result, labeled evaluation corpus, or validation accuracy. The task notebook has two internal inconsistencies: its positional-encoding section says a formula is broken, but the actual `div_term` assignment in that notebook already matches the standard formula; and the answer-cell label says “10 pts” even though Part 3 assigns 5 marks. The solution follows the standard equation and does not claim a score for either disputed cell.

## Implementation and diagnostics

For attention scores, the implementation divides by the square root of the head width, applies a broadcastable keep-mask before softmax, and uses a large finite negative value for masked scores. Multi-head attention reshapes and transposes the sequence and head axes before reversing that transformation after attention. Positional encoding alternates sine on even columns and cosine on odd columns; it slices the frequency vector correctly when the embedding width is odd. The classifier mean-pools over the sequence dimension and returns class logits.

**Rules and dependencies:** The task asks for the components to be implemented in PyTorch and explicitly says masked logits should use a large finite negative value such as `-1e9`, not `-inf`. The solution uses PyTorch modules/functions plus Python's `math`; it does not substitute a built-in `MultiheadAttention` layer for the assessed component.

## Compute and footprint

The checks used synthetic tensors on CPU. No runtime benchmark, training run, GPU result, pretrained weights, or external data was recorded or required.

## Alternatives considered

- PyTorch's built-in `MultiheadAttention` would conceal the projections, head reshaping, and scaled-attention operation the exercise asks to implement.
- A boolean keep-mask and an additive mask are both common. This implementation uses a broadcastable keep-mask and fills excluded scores with a finite negative value before softmax.
- A fixed even-width-only position encoder is shorter, but the general implementation supports odd embedding widths without a cosine-slice shape error.

## Progressive hints

1. Compute `QKᵀ / sqrt(d_k)`, mask excluded keys before softmax, then multiply the weights by `V`.
2. Project Q/K/V into `num_heads`, transpose to put heads before sequence positions, and reverse the reshape after attention.
3. Add sinusoidal positions before the encoder stack and mean-pool the sequence outputs for classification.
4. Use the same frequency scale for even sine and odd cosine columns, and check the final cosine slice for odd `d_model`.
5. Build the causal keep-mask with ones on and below the diagonal so no position can attend to a future token.

**One-line summary:** Build the Transformer pieces explicitly, preserve the head and sequence dimensions, and apply finite-score masks before softmax.

## Sources and reuse

- [SOTA task summary](https://checklist.sota-ai.org/problems/noai-singapore-2026-final-attention-mechanism/)
- [Official NOAI 2026 Programming Task 3 notebook](https://aisingapore.org/wp-content/uploads/question_3_v20022026.ipynb)
- [Released solution notebook](https://github.com/AISGNUSNOAI/NOAI-2026/blob/main/question_3_solution.ipynb)
- [Official NOAI learning-resources page](https://noai.aisingapore.org/learning-resources/)

The source does not state a reuse license. This report paraphrases the task and the solution code is independently written; the notebooks are linked rather than copied into this repository.
