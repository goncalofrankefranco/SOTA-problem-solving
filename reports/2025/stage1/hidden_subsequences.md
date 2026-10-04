# Poland 2025 Stage I — Ukryte Podciągi (Hidden Subsequences)

**Domain:** Binary sequence modeling, hidden-pattern regression/classification  
**Metric:** Mean squared error. The score is `round(max(100*(64-MSE)/64, 0))`; a validation MSE of about 0.32 or lower is needed for a rounded 100.  
**Official sources:** [SOTA checklist](https://checklist.sota-ai.org/) · [starter notebook](https://github.com/OlimpiadaAI/II-OlimpiadaAI/tree/main/1_etap/5_ukryte_podciagi) · [organizer worked solution](https://github.com/OlimpiadaAI/II-OlimpiadaAI/blob/main/1_etap/5_ukryte_podciagi/5_ukryte_podciagi_opracowanie.ipynb)

## Abridged statement

Given a fixed-length binary sequence, predict the sum of the weights for three hidden binary subsequences that occur in it. A subsequence preserves order but need not be contiguous. The submission must be a learnable PyTorch model with fewer than 50,000 parameters, trained for no more than 4,000 iterations, and must run within four minutes on GPU.

## Dataset analysis and EDA

The organizer notebook reports 511,999 training and 511,999 validation rows. Each input is a 24-bit binary vector with no missing values. Targets take exactly eight values: `[-19, -8, 0, 11, 28, 39, 47, 58]`. Train and validation target means are 22.67 and 22.70, standard deviations are both 24.24, and the mean input bit is 0.50 in both splits. The notebook notes that the number of ones alone has no simple linear relation to the target. Data are downloaded from Google Drive and are not in GitHub or the local mirrors; access failed here with `HTTP 403: CONNECT tunnel failed`.

## Experiments

- A last-state LSTM regressor with 42,113 parameters reaches validation MSE 56.3366 (12/100 points).
- Adding attention over all LSTM outputs uses 42,177 parameters and reduces MSE to 7.2511 (89/100 points).
- Regression outputs from the attention model are all between the eight valid target values rather than exactly equal to them, motivating classification.
- The selected eight-class LSTM uses all 24 hidden states, maps classes back to the eight target values, has 49,512 parameters, and is trained for the allowed 4,000 iterations. Its validation MSE is 1.0468, corresponding to **98/100 points**.
- The notebook also discusses 1-D CNN regression and GRU/LSTM pooling as alternatives, but supplies no measured results for them.

The organizer's example MLP and the LSTM v1 use the final sequence state; v2 adds attention; v3 changes the output to classification. No additional local experiments were possible without the CSVs; the shared PyTorch runtime is available on CPU, but the task expects a GPU run.

## Selected solution and validation evidence

The standalone script in [hidden_subsequences.py](../../../solutions/2025/stage1/hidden_subsequences.py) reworks the selected `ClfLSTM` and its 4,000-step training loop. It derives the class list from training targets and uses the validation CSV only for final MSE scoring. The organizer's notebook records **validation MSE 1.0468 and 98/100 estimated points**. This is organizer-reported, not locally reproduced. The CSV download requests are blocked by the environment's Google Drive 403. PyTorch is available locally, but no GPU is available for the four-minute GPU limit; the inaccessible data prevent local pursuit of the remaining two points or alternative validation. No hidden labels or secret-test claims are made.

## Compute and runtime

The task permits at most 4,000 training iterations, 50,000 parameters, and four minutes on GPU. The selected model has 49,512 parameters and uses 128-example minibatches. A precise end-to-end wall-clock time is not recorded in the executed notebook output.

## Alternatives

The measured alternatives are a final-state LSTM regressor and an attention LSTM regressor. The model notebook discusses 1-D convolutions for local motifs and mean/max pooling over recurrent states, but gives no validation measurements for those ideas. Recovering the exact subsequence logic is an attractive diagnostic, but the competition explicitly requires a learnable model and the official solution stays within that requirement.

## Progressive hints

1. Verify that targets take a finite set of values before choosing regression or classification.
2. A subsequence depends on ordered bits across the whole input, not only adjacent pairs.
3. Use all recurrent time-step states or attention so early and middle patterns remain visible to the output layer.
4. Map class predictions back to the eight legal target values.

**One-line summary:** An 8-class, 49,512-parameter LSTM scores MSE 1.0468 (98/100) on organizer-released validation; Google Drive blocking prevents local pursuit of 100 points.
