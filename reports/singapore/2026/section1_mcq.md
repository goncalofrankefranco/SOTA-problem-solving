# NOAI Singapore 2026 Final — Section 1: Multiple-Choice Questions

**Problem domain:** Machine learning, deep learning, computer vision, NLP, and model evaluation  
**Official metric:** 1 mark per correct answer, 20 total; no penalty for wrong answers

## Abridged statement

Answer 20 conceptual and applied questions covering classical ML and modern deep learning. The released assessment PDF contains the questions and choices, but no answer key.

This is a retrospective solution based on the public release. The organizer's instructions prohibit AI assistance during the live assessment.

## Data analysis

There is no dataset or exploratory analysis. The useful “EDA” is to classify the items by topic: model selection and validation, optimization, computer vision, attention, and language-model training. Several items describe specific code or numerical examples and should be evaluated against those exact details rather than a generic rule.

## Experiments

I compared each answer choice with the released question text and its mathematical or implementation semantics. There is no local grader, answer key, or released response set against which to measure a score.

## Chosen method

The derived selections and item-level caveats are in [the answer key](../../../solutions/singapore/2026/section1_mcq_key.md). The key uses starred best guesses for questionable items. Q14 has no technically supported choice; B is included only as a forced, very-low-confidence guess because wrong answers carry no penalty.

## Measured score and evidence

**Score not measured.** The assessment is 20 marks, but AI Singapore has not released a key or machine-readable scoring harness alongside the PDF. The answer file is a reasoned reference, not an organizer key. Question 14 has no technically defensible choice as written; its listed answer is a forced guess only. Questions 5, 8, 11, 12, 18, and 20 are starred or caveated because the prompt admits multiple defects, assumptions, or an overbroad diagnosis.

## Alternatives considered

- Treating every item as having an unambiguous intended choice would conceal defects in the wording or code snippet.
- A key with no reasoning would be shorter but would make the few disputed selections hard to audit.

## Compute and footprint

Not applicable: this is an individual paper assessment with no model training or data processing.

## Progressive hints

1. Translate each question into a specific mathematical or implementation claim before looking at the choices.
2. For code questions, follow the actual execution order, including when modules are frozen, replaced, or added to the optimizer.
3. For losses and metrics, write the definition and apply it to the supplied values; do not rely on familiar labels alone.
4. If the choices do not contain a claim supported by the snippet, record that mismatch instead of inventing a fault.

**One-line summary:** Use the exact question details to derive the choices; the released PDF has no official key and several items are ambiguous.

**Sources:** [SOTA task summary and scoring](https://checklist.sota-ai.org/problems/noai-singapore-2026-final-mcq/) · [Official final assessment PDF](https://aisingapore.org/wp-content/uploads/2026/04/NOAI-2026-Final-Assessment-Section-1-MCQs-Google-Forms.pdf) · [NOAI learning resources](https://noai.aisingapore.org/learning-resources/)
