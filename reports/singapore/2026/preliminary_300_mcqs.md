# NOAI Singapore 2026 Preliminary — 300 MCQs

**Problem domain:** Mathematics, computing, classical ML, deep learning, generative AI, and ML operations  
**Official metric:** 1 mark per correct answer, 300 total; no deduction for wrong answers

## Abridged statement

Answer the 300 released questions spanning linear algebra, calculus, probability, Python, NumPy, PyTorch, classical ML, neural networks, CNNs, transformers, generative models, deployment, and MLOps. The post-event assessment PDF exposes question text and choices but not the official response key.

This is a retrospective solution based on the public release. The organizer's instructions prohibit AI assistance during the live assessment.

## Data analysis

There is no model-training dataset. The assessment is broad and mostly tests definitions, calculations, and diagnosis of short scenarios. The official PDF numbers these after ten personal-information fields: the key therefore indexes the 300 assessment questions in order (question 1 corresponds to form field 12; question 300 to field 311).

## Experiments

I derived the answer sequence from the released question text and choices, checking the mathematical items directly and using standard definitions for the programming and ML items. There is no official key or scorer with which to measure accuracy. A small number of items are underdetermined or use overbroad claims; they are marked rather than silently guessed.

## Chosen method

[The derived answer key](../../../solutions/singapore/2026/preliminary_300_mcq_key.md) groups choices by question range. `?` means no unique answer is supported by the released prompt; `*` marks the likely intended choice where the released wording is too broad or technically imperfect. The file is not an official key.

## Measured score and evidence

**Score not measured.** No organizer answer key, returned form data, or public grading harness was released with the PDF. The preliminary answer key is an auditable derivation, not an official score. In particular, no percentage or 300/300 claim is made.

## Alternatives considered

- Omitting all answers on uncertain items would be unhelpful where one option remains the best supported choice; the key flags those choices with an asterisk instead.
- Copying the organizer's online form as if its responses were official would be misleading; the public PDF shows the questions, not responses.

## Compute and footprint

Not applicable: this is a multiple-choice assessment, with no model training or data processing.

## Progressive hints

1. Identify the tested concept before choosing an option: dimension, derivative, estimator, metric, or failure mode.
2. For numeric questions, calculate the value rather than recognizing a familiar-looking choice.
3. For scenario questions, distinguish the observed symptom from one possible cause; missing evidence can make more than one diagnosis plausible.
4. Treat claims about library behavior as version- or implementation-dependent unless the question states the relevant implementation.

**One-line summary:** Use definitions and direct calculations for the released questions; the public PDF contains no official answer key, so ambiguous items remain flagged.

**Sources:** [SOTA task summary and scoring](https://checklist.sota-ai.org/problems/noai-singapore-2026-preliminary-300-mcqs/) · [Official preliminary assessment PDF](https://aisingapore.org/wp-content/uploads/2026/04/NOAI-2026-Preliminary-Round-Assessment-300-MCQs-Google-Forms.pdf) · [NOAI learning resources](https://noai.aisingapore.org/learning-resources/)
