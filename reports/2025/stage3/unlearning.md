# (Un)learning in Machine Learning

## Problem, domain, and metric

- **Domain:** machine unlearning for image classification.
- **Task:** modify the LeNet feature extractor to forget one selected Fashion-MNIST class, while keeping the final `fc2` layer and model architecture unchanged.
- **Metric:** equally weighted scores for target-class accuracy (lower is better), remaining-class accuracy (higher), parameter L2 distance from the base checkpoint (lower), and KL divergence from uniform predictions on the target class (lower). Hard caps reject models with retain accuracy below 0.75, target accuracy above 0.5, L2 above 8, or KL above 1.75. The main score thresholds are target accuracy 0.09/0.30, retain accuracy 0.87/0.90, L2 1.3/3.0, and KL 0.2/0.5.
- **Abridged statement:** unlearn whichever one of the ten Fashion-MNIST classes is selected at evaluation, preserving other-class accuracy and changing the pretrained model as little as possible.

## Released data and EDA

The [official notebook](https://github.com/OlimpiadaAI/II-OlimpiadaAI/tree/main/3_etap/2_oduczanie) downloads full Fashion-MNIST and filtered class archives from Google Drive; those archives are not present in the available local mirrors. The supplied base checkpoint is 184,562 bytes. The official input transforms 28×28 grayscale images using mean 0.286 and standard deviation 0.353. The notebook's validation target is class 9, while the task warns that any class may be selected on hidden evaluation.

The model has two convolution/BatchNorm blocks, two hidden linear layers, and an 84-to-10 output layer. Only feature-extractor parameters may change. The supplied full-data loader is explicitly passed to `unlearn`, so training against target-class examples is part of the official interface; no hidden-test labels are needed.

## Experiments

| Candidate | Released score evidence | Result |
|---|---|---:|
| Add independent Gaussian noise (`std=0.1`) to every feature-extractor parameter | Dummy cell output: target 9, KL 2.06, retain accuracy 0.50, L2 20.73, forget accuracy 0.82 | 0/100 |
| Four epochs of retain cross-entropy + target-to-uniform-over-other-classes KL + L2 anchoring, with AdamW at `1e-4` | Recorded output in mirrored `Unlearning/solution.ipynb`, target 9 | 99.0049/100 |

For the latter run the notebook reports KL 0.21, remaining-class accuracy 0.90, L2 distance 0.66, and target accuracy 0.07. The KL value is just above its full-credit threshold, accounting for almost all of the lost point. The mirrored notebook reports 938 full-data batches per epoch and approximately 18–20 seconds per epoch on its GPU.

## Selected solution

[unlearning.py](../../../solutions/2025/stage3/unlearning.py) implements the scored method. It deep-copies the base model, freezes `fc2`, trains the remaining parameters on the official full-data loader, applies cross-entropy to non-target examples, matches target examples to a uniform distribution over the other nine classes, and anchors trainable weights to their initial values with an L2 penalty. It takes four epochs and keeps the last classifier untouched.

### Score evidence and runtime

The reported **99.0049/100** is the score stored in the mirrored candidate notebook, not a fresh local run of the file above. The implementation follows that candidate's loss and optimizer. The Fashion-MNIST archives needed for local reproduction are missing, and this workspace has CPU-only PyTorch; the checkpoint is available. The candidate notebook's recorded training loop is about **75 seconds** on GPU for four epochs, excluding downloads and evaluation. Hidden-test score, including generalization to a different forgotten class, is unknown.

## Alternatives and caveats

A slight target probability below 0.1 (rather than zero) could lower KL-to-uniform while retaining a low target argmax rate; it is an unmeasured tuning idea, not the selected evidence-backed method. Per-target-class tuning also needs care because validation only demonstrates class 9. The single validation target and same full Fashion-MNIST train loader make the result less informative about a different hidden target class than a held-out-class protocol would be.

## Progressive hints

1. Preserve the output layer exactly; only the feature extractor can change.
2. Use mixed batches so retain examples continue to receive their original labels.
3. On target examples, discourage the target class and spread probability among the other classes.
4. Add an L2 anchor to the starting feature weights; the score rewards both forgetting and small edits.

**One-line summary:** the mirrored four-epoch unlearning candidate scored 99.0049/100 on released target class 9; the checkpoint is local, but Fashion-MNIST archives are missing and hidden-target generalization is unknown.
