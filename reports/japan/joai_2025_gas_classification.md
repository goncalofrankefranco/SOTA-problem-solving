# JOAI 2025 — Classify Gases from Multimodal Information

**Problem domain:** Multimodal, four-class classification (sensor tabular data, infrared images, and captions)  
**Evaluation metric:** Weighted F1. The official evaluation example uses `sklearn.metrics.f1_score(..., average="weighted")`; a perfect score is 1.0.  
**Official task:** [JOAI 2025 competition overview](https://www.kaggle.com/competitions/joai-2025-competition/overview) · [SOTA English task page and translation](https://checklist.sota-ai.org/problems/joai-japan-2025-competition-multimodal-gas-classification/)

## Abridged statement

For every test sample, predict one of four states describing whether perfume, smoke, both, or neither is present. Each sample has readings from two gas sensors, a cropped infrared image, and a natural-language caption generated from the uncropped scene. The required output is one label for each test index. This is a paraphrase of the Japanese task statement; see the official source for exact wording and rules.

## Data and EDA

The public practice data contain `MQ5`, `MQ8`, `Caption`, and an image filename for each sample; the training labels are `Gas`. The available Hugging Face mirror reports 5,760 labeled training rows and four equally represented classes (1,440 each). The Kaggle package also provides unlabeled test rows and a sample-submission file. The mirror is used here only as a metadata reference; no data or images are included in this repository. [Dataset viewer](https://huggingface.co/datasets/kami634/joai-2025-dataset)

The organizers note an important modality mismatch: infrared images are crops, while the captions describe the original, uncropped images. Captions follow repeated patterns around temperatures, colors, and scene descriptions, and may describe a possible gas leak even for a `NoGas` target. This makes caption cues useful but potentially biased. [Task description](https://checklist.sota-ai.org/problems/joai-japan-2025-competition-multimodal-gas-classification/) · [Organizer's post-contest solution review](https://ioai-japan.org/joai2025-solution/)

The organizer's summary and public participant write-ups point to these EDA findings:

- The class counts are balanced, so weighted F1 is close to macro F1 on the training distribution.
- MQ5 and MQ8 are strong predictors. Their ratio and interaction features help distinguish some classes, while `NoGas` and `Perfume` are harder to separate with sensors alone.
- Captions provide regularized, extractable signals such as Celsius/Fahrenheit temperature ranges, colors, and scene terms. Some gas-related wording is noisy.
- Infrared patterns provide additional information. Upper solutions used either image models or image summaries/embeddings alongside the sensors and captions.
- Strong published approaches used stratified validation and blended or stacked predictions from multiple modalities.

These are source-backed observations, not a fresh plot-by-plot EDA run in this workspace: the competition files were not available locally and direct Kaggle downloads were unavailable here.

## Experiments and chosen method

No score from this implementation is reported. We could not run full-data experiments because `train.csv`, `test.csv`, and the image directory were not present in the workspace, and this environment has no Kaggle download credentials or direct download access. The solution script includes a stratified K-fold weighted-F1 validation mode so these comparisons can be run once the official files are placed in the expected directory.

The runnable candidate in [the solution script](../../solutions/japan/joai_2025_gas.py) uses two complementary CPU models and averages their class probabilities:

1. **ExtraTrees on dense engineered features:** raw/log sensor readings, sensor sums, differences, ratios and products; parsed caption temperatures, color/scene flags, caption-length statistics; and compact color/intensity summaries plus a 4×4 spatial pooling of each image channel.
2. **Logistic regression on sparse text plus scaled numeric features:** word and character TF-IDF from captions, joined with the dense sensor, caption, and image summaries.
3. **Equal probability blend:** averages both model outputs before selecting the class.

The script deliberately uses no pretrained weights. This avoids depending on the organizer's designated-weight list and keeps the code runnable with competition data, NumPy, pandas, Pillow, SciPy, and scikit-learn. The 5-fold validation command reports per-fold model scores, overall weighted F1, and a confusion matrix. No hyperparameter or blend weight has been selected using the competition leaderboard.

### Experiment evidence and score provenance

| Result | Score | Provenance |
|---|---:|---|
| This repository's full-data cross-validation | Not measured | Competition files were unavailable in this workspace; the script's `--mode validate` is ready to measure it. |
| Public participant write-up, open track | 0.984 at one reported submission point | Individual participant's contemporaneous report; not our run or an official final/private score. [Write-up](https://zenn.dev/setoeditor/articles/819d2173a8ec17) |
| Public participant write-up, selection track | 0.9906 mean CV weighted F1; reported 3rd on the public board at that point | Individual participant's report, not independently reproduced here and not a final/private leaderboard score. [Write-up](https://zenn.dev/kanda9685/articles/a6e9dc0cc89493) |

The maximum possible value of weighted F1 is 1.0. We have not verified that this solution reaches it or any other particular score, so it is **not presented as a full-score or optimal solution**.

## Implementation and diagnostics

Run on the Kaggle practice dataset, or on an equivalent local copy, after placing the files under `data/`:

```bash
python solutions/japan/joai_2025_gas.py \
  --data-dir /path/to/data --mode validate --folds 5

python solutions/japan/joai_2025_gas.py \
  --data-dir /path/to/data --mode submit --output submission.csv
```

Expected files are `train.csv`, `test.csv`, optional `sample_submission.csv`, and an `images/` directory. The script checks required columns, normalizes the `Mixed`/`Mixture` spelling to the competition label, reports missing images, and uses the sample submission's row order and index when present. It does not download or copy the competition assets. A two-fold synthetic smoke run and Python compilation were used to check the script mechanics; neither produces evidence about task accuracy.

## Compute and footprint

This is a modest classical-ML pipeline intended to run on CPU. Feature extraction resizes each image to 32×32 and stores only numerical summaries; the text model uses sparse TF-IDF features. The tree model uses 500 ExtraTrees and the text model is multiclass logistic regression. Runtime and memory on the actual task data were not measured in this workspace. No GPU or external model download is required.

## Alternatives and limitations

- A sensor-only gradient-boosted tree is a useful fast baseline, but published reports found value in using more than one modality.
- Hand-built caption fields are lightweight; organizer-reviewed strong solutions also extracted transformer embeddings from permitted DeBERTa/RoBERTa weights.
- CNN/ViT image fine-tuning, thermal-image reconstruction and image/text embeddings are stronger but need more dependencies and compute. The competition allowed only pretrained weights designated by its organizers; do not substitute arbitrary public checkpoints.
- LightGBM/CatBoost feature models and probability stacking were common stronger alternatives in the organizer's review. They can be compared against the provided baseline with the same stratified folds.
- The public practice dataset and competition data are licensed **CC BY-NC-SA 3.0 IGO**. Do not commit the original CSVs, images, model weights, or translated statement. The competition also restricted external datasets and designated pretrained weights. The 2025 live competition ended in May 2025 and its public practice copy closed on 31 May 2026. [Rules and license](https://www.kaggle.com/competitions/joai-2025-competition/rules) · [Practice competition dates](https://www.kaggle.com/competitions/playground-joai-competition-2025)

## Progressive hints

1. Which of the three modalities gives the clearest initial separation? Start with the two gas-sensor readings and inspect their joint distribution by label.
2. Does a ratio or log transform make the sensor groups easier to separate? Add interactions such as sum, difference, and product.
3. Which repeated numeric and descriptive patterns can be extracted from the captions without a large language model?
4. How can the infrared image add information while keeping the validation split stratified? Compare compact image summaries against a trained image model, then blend out-of-fold probabilities.

**One-line solution:** Combine sensor interactions, caption TF-IDF/parsed clues, and compact infrared image features, then tune a probability blend using stratified weighted-F1 validation; task score remains unmeasured here.

## Attribution and licensing

Task facts and the abridged description above are paraphrased from the JOAI Committee's 2025 competition overview and post-contest solution review, and from SOTA's English translation. The source task materials/data and SOTA translation are identified as **CC BY-NC-SA 3.0 IGO**; attribution is to the [JOAI Committee](https://ioai-japan.org/joai2025-solution/) and [SOTA task page](https://checklist.sota-ai.org/problems/joai-japan-2025-competition-multimodal-gas-classification/). The original data, images, and statement are not reproduced here.
