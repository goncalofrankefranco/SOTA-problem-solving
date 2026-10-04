# Stage 2 solution code

These implementations are designed for the matching official starter notebooks. The notebook supplies the task-specific model, data, evaluator, and allowed runtime.

| Task | Implementation | Report | Current validation status |
|---|---|---|---|
| Drzewa decyzyjne | [decision_trees.py](decision_trees.py) | [report](../../reports/decision_trees.md) | 66.0/100 on collection A; hidden collection B unverified |
| Kolorowanie z GANem | [gan_colorization.py](gan_colorization.py) | [report](../../reports/gan_colorization.md) | Generator-only score unverified; 100/100 cross-fit color-prior diagnostic is excluded because it uses validation targets for fitting |
| Optymalizator malarza | [solution.py](../../2_etap/optymalizator_malarza/solution.py) | [reference report](../../REFERENCE_SOLUTIONS.md#selected-stage-2-solution--optymalizator-malarza-painter-optimizer) | 100/100 on released validation |
| Predyktor tokenów | [token_predictor.py](token_predictor.py) | [report](../../reports/token_predictor.md) | 100/100 on 99 released validation examples; hidden-test performance unverified |

Scores in the reports distinguish measurements on released validation data from organizer-reported results and unverified candidates. The secret test sets are not available in this repository.
