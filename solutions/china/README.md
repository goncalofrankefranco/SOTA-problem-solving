# NOAI China candidate solutions

Runnable task-specific candidate code for the 12 reports in [reports/china](../../reports/china/README.md). The implementations use only authored code and expect the organizer files to be supplied locally; datasets, baseline notebooks, and model weights are not included.

## 2024

| Task | Code | Notes |
|---|---|---|
| Basketball shooting percentage | [basketball_shooting.py](2024/basketball_shooting.py) | Constraint-compliant 2→8→8→1 PyTorch MLP; mirror OOF only |
| Real or fake images | [real_fake_images.py](2024/real_fake_images.py) | Small CNN with global-pooling and dense-head choices; mirror holdout only |
| News text classification | [news_text_classification.py](2024/news_text_classification.py) | Sparse word/character TF-IDF with PyTorch multiclass head |
| Pendulum motion | [pendulum_motion.py](2024/pendulum_motion.py) | PyTorch least-squares initialization and nonlinear ODE fitting |

## 2025

| Task | Code | Notes |
|---|---|---|
| Compound word segmentation | [compound_word_segmentation.py](2025/compound_word_segmentation.py) | Character BiLSTM plus lexicon-aware boundary decoding |
| Chemical reaction kinetics | [chemical_reaction_kinetics.py](2025/chemical_reaction_kinetics.py) | Blended tree regressors on concentrations and derived features |
| Synthetic speech detector | [synthetic_speech_detector.py](2025/synthetic_speech_detector.py) | One-channel ResNet18 with light spectrogram masking |
| Grid collage classification | [grid_collage_classification.py](2025/grid_collage_classification.py) | Weak synthetic supervision from category-matched mosaics |

## 2026

| Task | Code | Notes |
|---|---|---|
| User intent recognition | [user_intent_recognition.py](2026/user_intent_recognition.py) | Local Chinese BERT embeddings blended with character n-grams and authored anchors |
| Isaac Sim2Real state prediction | [isaac_sim_state_prediction.py](2026/isaac_sim_state_prediction.py) | Per-joint residual regression conditioned on the visible prefix |
| Product · Sum | [product_sum.py](2026/product_sum.py) | Weakly supervised digit distributions with exact sum/product marginalization |
| Maze information prediction | [maze_information_prediction.py](2026/maze_information_prediction.py) | BFS/topology features and per-target ExtraTrees regression |

## Evaluation status

No Bohrium credentials or official grader data were available for this work. The 2026 organizer reference B scores are reported in the task reports for comparison; they were not reproduced. Local 2024 cross-validation and fit diagnostics use a third-party mirror whose repository does not declare a license; the mirror data are not included here. Treat every candidate as unverified on the official leaderboard unless its report states otherwise.
