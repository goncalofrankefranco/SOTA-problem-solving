# NOAI China task reports

Reports for all 12 NOAI China Round 2 tasks listed on the [SOTA checklist](https://checklist.sota-ai.org/), four per year from 2024 through 2026. Each report includes a task paraphrase, available data/EDA, method, score provenance, implementation notes, alternatives, progressive hints, and source/license notes.

| Year | Task | Report | Evidence in this workspace |
|---|---|---|---|
| 2024 | Basketball shooting percentage | [Report](2024/basketball_shooting.md) | 0.5978 five-fold OOF accuracy on an unlicensed community mirror; official score unmeasured |
| 2024 | Real or fake images | [Report](2024/real_fake_images.md) | 0.7279 local formula score on a mirrored-data holdout; participant-reported leaderboard accuracy 1.0000 |
| 2024 | News text classification | [Report](2024/news_text_classification.md) | 0.9724 duplicate-aware OOF macro-F1 on an unlicensed community mirror; official score unmeasured |
| 2024 | Pendulum motion with missing data | [Report](2024/pendulum_motion.md) | 0.000270-rad fit RMSE on the visible mirrored curve; participant-reported leaderboard B 0.9186 |
| 2025 | Compound word segmentation | [Report](2025/compound_word_segmentation.md) | 0.862135 exact-span F1 on related IOAI GAITE holdout; NOAI score unmeasured and data identity unverified |
| 2025 | Chemical reaction kinetics | [Report](2025/chemical_reaction_kinetics.md) | No local score; organizer data and grader unavailable |
| 2025 | Synthetic speech detector | [Report](2025/synthetic_speech_detector.md) | NOAI score unmeasured; related IOAI GAITE data identity unverified |
| 2025 | Grid collage classification | [Report](2025/grid_collage_classification.md) | No labeled collage targets available for local accuracy measurement |
| 2026 | User intent recognition | [Report](2026/user_intent_recognition.md) | Candidate score unmeasured; organizer reference B 0.8154 |
| 2026 | Isaac Sim2Real state prediction | [Report](2026/isaac_sim_state_prediction.md) | Candidate score unmeasured; organizer reference B 0.7861 |
| 2026 | Product · Sum | [Report](2026/product_sum.md) | Candidate score unmeasured; organizer reference B 0.9600 |
| 2026 | Maze information prediction | [Report](2026/maze_information_prediction.md) | Candidate score unmeasured; organizer reference B 0.8653 |

The reference/participant values above are not scores from the authored candidates. The 2024 mirrored source data were used only from temporary files and are not redistributed. The organizer source pages do not state a license. No task is described as full-score or optimal without an official evaluation.

See the [candidate code index](../../solutions/china/README.md).
