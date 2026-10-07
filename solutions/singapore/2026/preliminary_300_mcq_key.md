# NOAI Singapore 2026 Preliminary — derived answer key

This key follows the order of the 300 assessment questions in the [released PDF](https://aisingapore.org/wp-content/uploads/2026/04/NOAI-2026-Preliminary-Round-Assessment-300-MCQs-Google-Forms.pdf). Form fields 1–11 collect participant information; assessment question 1 is form field 12. Letters refer to choices in their displayed order (A–D). This is a derived key, not an official answer key. `?` means the released information does not support a unique choice. `*` means the likely intended option has a technical caveat.

| Questions | Choices, in question order |
|---:|:---|
| 1–20 | A A C B B C B B B A B A A A B B A A B A |
| 21–40 | B A B A A B A B A A A B B B B A B A A A |
| 41–60 | A B A A A B B A B B D C D B B D C B D C |
| 61–80 | D C* B D B A C D B D C C A A A C C B A D |
| 81–100 | C B C B D A C A D C C D D C A B A C D B |
| 101–120 | D C A A B C D B C C A B D B A C C B C C |
| 121–140 | C D B C A D A A A B C C D B D D A C D D |
| 141–160 | C D B D C C C D A C D D C D D B D D C C |
| 161–180 | D D A C C D D D D D D D D C C C D D D D |
| 181–200 | D C C C C C C C D C C C D B B B C C C D |
| 201–220 | D A A C C D C C C B C C C C D C B C C D |
| 221–238 | B C C D B C C A A C B C D C B B C C |
| 239–260 | C A B C C C C C C B C A B D A C D B D A A* C* |
| 261–280 | C A C A D A D D C D B A* A B A C A* A* D* C |
| 281–300 | A* A A D* D A C D A A A C C D C B B C B B |

## Uncertain and technically imperfect items

- **Q62:** C is the likely choice for a p-value, but the option describes it loosely. A p-value is the probability, under the null, of results at least as extreme as those observed.
- **Q259:** A is likely intended for the missing-value scenario, but native handling is implementation-dependent; ordinary decision trees do not universally support missing values without preprocessing.
- **Q260:** C is the best offered clustering option, but DBSCAN is not robust to clusters with substantially different densities. The question's wording overstates its fit.
- **Q272:** A is the best diagnostic sequence among the choices, but anomaly detection and numeric-domain checks are both defensible without a stated NaN source.
- **Q277:** A is the closest listed explanation (softmax saturation), but zero final-layer gradients with nonzero earlier gradients do not uniquely identify it; some other choices would also affect earlier gradients or raise an error.
- **Q278:** A is likely intended (final-layer biases suppress classes 3–49), but the observed class collapse does not establish that initialization caused it after 100 epochs.
- **Q279:** D is the likely intended “sharp minimum” explanation, but the stated observations do not rule out other causes, and the learning-rate change in that option is not specified.
- **Q281:** A is the closest route to a 10× speedup, but 8-GPU scaling and convergence quality are hardware- and workload-dependent; the choices do not establish 10×.
- **Q284:** D is the likely intended adaptive-batch strategy, but the compute-budget wording conflicts with listed wall-clock times and no result is supplied for that strategy.

The key has **no measured score**: the public release contains no official key or scorer. These answers are not represented as organizer-verified.
