# Colombia solutions

- `five_artists.py` downloads the two public Hugging Face datasets, makes a stratified validation split, trains a residual CNN, refits on all 10,000 labels using the selected epoch count, and writes a Kaggle-format prediction file.
- `heart_disease.py` trains a histogram gradient boosting classifier, tunes the high-risk threshold for validation F1, and writes a Kaggle-format prediction file.

Examples:

```bash
python five_artists.py --output submission.csv --size 128 --epochs 30
python heart_disease.py --train train.csv --test test.csv --output submission.csv
```

For the heart task, `--released-test-target` can be used to calculate an explicitly post-contest F1 result when such a labeled file is available. Neither script contains contest data or weights.
