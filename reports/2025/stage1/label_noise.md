# Poland 2025 Stage I — Szum w Etykietach Danych (Label Noise)

**Domain:** Image classification under class imbalance and label noise  
**Metric:** Mean balanced accuracy of two fixed classifiers. BAC ≥0.80 earns 100 points; BAC ≤0.50 earns 0.  
**Official sources:** [SOTA checklist](https://checklist.sota-ai.org/) · [starter notebook](https://github.com/OlimpiadaAI/II-OlimpiadaAI/tree/main/1_etap/4_szum_w_etykietach_danych) · [organizer worked solution](https://github.com/OlimpiadaAI/II-OlimpiadaAI/blob/main/1_etap/4_szum_w_etykietach_danych/4_szum_w_etykietach_danych_opracowanie.ipynb)

## Abridged statement

Train two copies of the fixed `SmallMobileNet` to classify two image classes. Training labels contain noise and are imbalanced; validation and test labels are clean. The submission is a batch-selection function receiving labels and per-example losses from both models. It must select separate index sets for the two models.

## Dataset analysis and EDA

The official training archive contains 10,000 grayscale images: 6,298 label-0 and 3,702 label-1 examples. The worked notebook reports no missing filenames or labels and no missing image files. PCA followed by t-SNE on a 2,000-image sample does not show clear class separation; per-image mean and standard-deviation histograms also overlap. The validation archive is also provided by Google Drive. Neither archive is tracked in the GitHub repository or the local mirrors; downloading the Drive assets failed with `HTTP 403: CONNECT tunnel failed` here.

## Experiments

The default selection rule uses every example for both models; the executed baseline obtains 0.50 mean BAC and 0/100 points. The organizer then explores a Co-Teaching-style rule: each model selects low-loss examples for the other, separately within each class. Batch-sampling rates examined were `[0.3,0.3]`, `[0.8,0.8]`, `[0.5,0.5]`, `[0.3,0.8]`, `[0.8,0.3]`, `[0.5,0.85]`, and `[0.5,0.92]`. The `[0.5,0.92]` rule gives a nearly even selected batch despite the original 63/37 class split. The final function uses `[0.5,0.93]` and cross-selects the smallest losses from the opposite model.

## Selected solution and validation evidence

The standalone callback in [label_noise_selection.py](../../../solutions/2025/stage1/label_noise_selection.py) implements the organizer's per-class cross-selection rule. The fixed `SmallMobileNet` trains for six epochs with AdamW, learning rate `0.01`, weight decay `0.001`, and batch size 128. The organizer's final validation outputs report BAC 0.8686728 for model 1 and 0.8937886 for model 2; their mean is 0.8812307, giving **100/100 organizer-reported points**. This is the best recorded result in the worked notebook. It was not rerun locally because the data are Drive-only; PyTorch is available locally, but no GPU is available. The reported labels used for training are the official noisy training labels; no validation targets are used to fit the models.

## Compute and runtime

The official evaluation limit is five minutes on GPU. Organizer training output shows 79 training batches per epoch taking roughly 8–10 seconds and 8 validation batches taking about one second for each of six epochs. The reported run is around one minute; exact total wall-clock time was not recorded.

## Alternatives

The notebook measures the default all-sample baseline and simulates the seven class-sampling rates above before running the selected Co-Teaching variation. Standard oversampling, undersampling, augmentation, and class-weighting are discussed as general options but are not reported as trained comparisons. The model architecture cannot be changed.

## Progressive hints

1. Compare the noisy training class proportions with the clean validation split.
2. A high loss can flag a mislabeled image, but one model can make mistakes of its own.
3. Let each model select easy examples for the other model.
4. Select different fractions from the two classes to balance the clean examples used for training.

**One-line summary:** Cross-selecting low-loss examples at 50% for class 0 and 93% for class 1 yields organizer-reported mean BAC 0.88123 and full points.
