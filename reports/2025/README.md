# Poland 2025 task reports

Reports cover all 12 official tasks from the [2025 Polish AI Olympiad repository](https://github.com/OlimpiadaAI/II-OlimpiadaAI), across Stages I–III. Each report includes the abridged problem, EDA, experiments, score provenance, compute notes, alternatives, and progressive hints.

## Stage I

| Problem | Report | Best available evidence |
|---|---|---|
| Coin Counter | [coin_counter.md](stage1/coin_counter.md) | Unscored; Google Drive data unavailable, CPU smoke check passed |
| Hallucination Detection | [hallucination_detection.md](stage1/hallucination_detection.md) | 100/100 reported by organizer notebook; not locally reproduced |
| ECG Anomaly Detection | [ecg_anomaly_detection.md](stage1/ecg_anomaly_detection.md) | 100/100 locally measured; 98.4413% balanced accuracy |
| Label Noise | [label_noise.md](stage1/label_noise.md) | 100/100 reported by organizer notebook; not locally reproduced |
| Hidden Subsequences | [hidden_subsequences.md](stage1/hidden_subsequences.md) | 98/100 reported by organizer notebook; data unavailable locally |

## Stage II

| Problem | Report | Best available evidence |
|---|---|---|
| Source Extraction | [ekstrakcja_zrodel.md](stage2/ekstrakcja_zrodel.md) | 100/100 locally verified through official cosine-search and scoring functions (nDCG@10 0.50365) |
| Taking Out a Loan | [kredytobranie.md](stage2/kredytobranie.md) | 100/100 locally verified with the official metric |
| Non-Normal Distribution | [rozklad_nienormalny.md](stage2/rozklad_nienormalny.md) | 100/100 locally measured; train-only model |

## Stage III

| Problem | Report | Best available evidence |
|---|---|---|
| Inpainting | [inpainting.md](stage3/inpainting.md) | 100/100 locally measured after official rounding |
| Unlearning | [unlearning.md](stage3/unlearning.md) | 99.0049/100 in the participant notebook; not locally reproduced |
| Translation Stylization | [translation_stylization.md](stage3/translation_stylization.md) | Unscored; Marian checkpoint unavailable |
| Data Prototypes | [data_prototypes.md](stage3/data_prototypes.md) | Candidate unscored; only organizer random baseline (0/100) available |

Scores are labeled by provenance. Drive-only assets and missing model weights block local measurements for several tasks; no secret-test results are claimed.
