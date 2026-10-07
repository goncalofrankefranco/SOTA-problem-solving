# Canada solutions

- `loan_default.py` trains a one-hot encoded histogram gradient boosting model, tunes its threshold for the F1 score of defaults on a stratified validation split, and writes `submission.csv`.
- `ai_fundamentals_solution.md` records the verified exam format and the blocker to completing an exact answer key. The source PDF is linked from the organizer page; it was not downloadable from this execution environment.

Example for the applied task:

```bash
python loan_default.py --train /path/to/train.csv --test /path/to/test.csv \
  --released-test-target /path/to/test_with_target.csv --output submission.csv
```

The released test labels are only used for the optional post-contest evaluation. They are never included in training.
