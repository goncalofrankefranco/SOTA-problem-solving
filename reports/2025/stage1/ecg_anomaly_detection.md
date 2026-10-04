# Poland 2025 Stage I — Wykrywanie Zaburzeń Sygnału EKG (ECG Anomaly Detection)

**Domain:** Synthetic biomedical time-series classification and signal feature engineering  
**Metric:** Five-class balanced accuracy. The task awards 100 points at BAC ≥98% and 0 points at BAC ≤75%.  
**Official sources:** [SOTA checklist](https://checklist.sota-ai.org/) · [starter notebook](https://github.com/OlimpiadaAI/II-OlimpiadaAI/tree/main/1_etap/3_wykrywanie_zaburzen_sygnalu_ekg) · [organizer worked solution](https://github.com/OlimpiadaAI/II-OlimpiadaAI/blob/main/1_etap/3_wykrywanie_zaburzen_sygnalu_ekg/3_wykrywanie_zaburzen_sygnalu_ekg_opracowanie.ipynb)

## Abridged statement

Classify a length-150 single-lead ECG segment into normal, AFib, PAC, PVC, or ST-elevation. Build at most four meta-features and train a Random Forest with at most 10 trees and maximum depth 10. Other classifiers are disallowed; feature construction may use Python, NumPy, and SciPy.

## Dataset analysis and EDA

The released NPZ contains 2,000 training and 1,500 validation signals, each with 150 `float64` samples. Training counts are 1,400 normal and 150 for each anomaly. Validation counts are 819 normal, 142 AFib, 191 PAC, 197 PVC, and 151 ST-elevation. The organizer verifies there are no missing values and notes extreme amplitudes are meaningful signal morphology, so they are retained. The data are synthetic approximations. Visual EDA describes AFib as irregular baseline activity, PAC as a premature atrial wave that can appear before or after the visible QRS, PVC as an abnormal ventricular complex, and ST-elevation as a raised post-QRS segment.

## Experiments

The organizer's v1 features are maximum-deviation position, unsigned maximum-deviation amplitude, total absolute changes, and a post-QRS two-window mean difference. With a balanced 10-tree, depth-10 forest, this obtains 82.9749% BAC and 35/100 points. The v2 experiment replaces the global deviation amplitude with a pre-QRS-only amplitude; it falls to 80.84% BAC because PAC changes may occur after the visible QRS and because post-QRS information also helps PVC and ST-elevation.

The organizer's v3 subtracts a slow sinusoidal baseline (`0.5*sin(i/45)`) and smooths the signal with a width-5 moving average before selecting the maximum deviation. Its four features obtain 97.3691% BAC and 97/100 points in the executed organizer notebook. I reproduced that exact score from the GitHub NPZ.

For the local optimization, I tried raw extrema and positions, R-relative window means/ranges/roughness, pre/post segment standard deviations, signed and unsigned deviations, and alternate ST-window differences. Replacing the unsigned amplitude with the **signed strongest deviation outside the QRS proxy** gave the best result. Keeping the other three organizer v3 features, this candidate reaches 98.4413% BAC. The best single replacements for the other slots were lower: alternative position 96.8887%, alternative roughness 97.5488%, and alternative ST difference 98.1507%. I also tested 64 permitted Random Forest hyperparameter combinations on the organizer features; none exceeded v3's 97.3691% BAC without the feature change.

## Selected solution

[ecg_solution.py](../../../solutions/2025/stage1/ecg_solution.py) implements baseline correction and QRS-proxy segmentation, then computes: (1) smoothed maximum-deviation position, (2) signed maximum deviation outside the QRS proxy, (3) total absolute changes there, and (4) the early/late post-QRS mean difference. It fits a 10-tree, depth-10 balanced Random Forest with `random_state=42`. It uses the official training labels only; validation labels are used only to score the fitted model.

## Validation score and evidence

**Local validation:** 98.4413% BAC, **100/100 points** under the official scoring rule. Reproduced from the official `train_validation_sets.npz` using the standalone runner. Command:

```text
python solutions/2025/stage1/ecg_solution.py path/to/train_validation_sets.npz
```

The 97.3691% organizer result is notebook-recorded; the 98.4413% result is locally measured. Neither establishes performance on the secret test split.

## Compute and runtime

The standalone fit-and-score command completed in about 10 seconds end to end on the local CPU, including Python and scientific-library startup. The model uses four scalar features and ten depth-10 trees, comfortably within the official one-minute CPU limit. No GPU is needed.

## Alternatives

The notebook discusses phase-aligning each signal at its R peak and using fixed offsets, as well as hybrid time/frequency features based on FFT and morphology. These were suggested but not measured in the executed notebook. Locally, direct R-relative regions and raw peak/roughness features were tested as one-feature replacements and were weaker than the selected signed deviation.

## Progressive hints

1. Check class balance before choosing the metric or forest weights.
2. Locate a rough QRS region from the largest positive and negative extrema.
3. Measure irregularity, dominant deviation, and post-QRS level change with four scalar features.
4. Preserve the sign of the strongest non-QRS deviation; it separates positive atrial waves from negative ventricular waves.

**One-line summary:** A four-feature 10-tree Random Forest with signed non-QRS deviation reaches 98.4413% released-validation BAC and 100/100 points.
