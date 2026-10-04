# Poland 2024 task reports

Reports cover all 13 official tasks from the [2024 Polish AI Olympiad repository](https://github.com/OlimpiadaAI/I-OlimpiadaAI): nine Stage I tasks and four final-stage tasks. Each report includes the abridged problem, data analysis, experiments, evidence and limitations, implementation notes, alternatives, and progressive hints.

## Stage I

| Problem | Report | Best available evidence |
|---|---|---|
| Adversarial Attacks | [adversarial_attacks.md](stage1/adversarial_attacks.md) | Criterion 48.65 on released validation; implies 100/100, with SSIM reconstructed rather than run through scikit-image |
| Color Quantization | [color_quantization.md](stage1/color_quantization.md) | 100/100 locally measured; validation mean 6,950.8 |
| Dependency Parsing | [dependency_parsing.md](stage1/dependency_parsing.md) | Unscored; HerBERT/Transformers unavailable locally |
| Imbalanced Classification | [imbalanced_classification.md](stage1/imbalanced_classification.md) | 100/100 in local compact-CNN experiment; standalone rerun/runtime unverified |
| Object Tracking 1 | [object_tracking_1.md](stage1/object_tracking_1.md) | 2/2 supplied example clips; full validation unavailable |
| Object Tracking 2 | [object_tracking_2.md](stage1/object_tracking_2.md) | 2/2 supplied example clips; full validation unavailable |
| Object Tracking 3 | [object_tracking_3.md](stage1/object_tracking_3.md) | 1/2 supplied example clips; full validation unavailable |
| Pruning | [pruning.md](stage1/pruning.md) | 0.9721 validation score; 100/100 task points, official validator passed |
| Riddles | [riddles.md](stage1/riddles.md) | Unscored; query, definition, morphology, and embedding assets unavailable |

## Final stage

| Problem | Report | Best available evidence |
|---|---|---|
| Anomaly Detection | [anomaly_detection.md](final/anomaly_detection.md) | 100/100 official metric on released validation; 91.30% accuracy locally measured |
| Ciphers | [ciphers.md](final/ciphers.md) | Unscored; cipher corpus and ground truth unavailable |
| Machine Translation | [machine_translation.md](final/machine_translation.md) | Unscored; Multi30k data unavailable |
| Self-Supervised Learning | [self_supervised_learning.md](final/self_supervised_learning.md) | 0.97315/1 reported by the participant notebook; not freshly reproduced |

Released-validation results do not establish secret-test performance. Sample-only tracking results are explicitly separated from competition validation scores.
