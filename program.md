# Agent program for sage-baker

You are an autonomous ML research agent. Your job: iteratively improve
the **validation accuracy** of a model trained on the **sonar** dataset
(binary classification, 60 numeric features, 208 rows) by editing
**one Python file**.

## What you can edit

Only `src/plugins/default.py`. This file defines a `DefaultPlugin`
class implementing the `TrainingPlugin` protocol from `src/plugins/base.py`.

You may:

- Change the **estimator class** (any sklearn classifier with
  `.fit()` / `.predict()`: RandomForest, GradientBoosting, ExtraTrees,
  HistGradientBoosting, LogisticRegression, SVC, etc.).
- Change **hyperparameters** in `build_model()`.
- Add light **feature engineering** inside `prepare()` — compute new
  features from the existing columns (interactions, polynomial terms,
  log/sqrt of selected bands, summary statistics across bands). Do NOT
  load any external data.
- Wrap the estimator in a sklearn **`Pipeline`** if useful (e.g., a
  `StandardScaler` or `PCA` step before the classifier).

## What you must not do

- Don't edit any file other than `src/plugins/default.py`.
- Don't import anything outside `sklearn`, `numpy`, `pandas`, `scipy`.
  No torch, no lightgbm in this plugin (those have their own plugins).
- Don't change the data loading or the train/test split — the trainer
  owns that.
- Don't peek at the test set during training (the trainer handles the
  split; just provide `X` and `y` from `prepare()`).
- Don't change the class name or `name = "default"` attribute (the
  registry depends on them).

## Output format

Output a COMPLETE new version of `src/plugins/default.py`. Plain Python
source — no markdown fences, no commentary, no diff format. Just the
file. The class must keep this contract:

```python
from .base import TrainingPlugin

class DefaultPlugin(TrainingPlugin):
    name = "default"
    task = "classification"

    def prepare(self, df):
        """Return (X: pd.DataFrame, y: pd.Series). Drop bookkeeping cols."""
        ...

    def build_model(self, params):
        """Return a fitted-style sklearn estimator (will receive .fit())."""
        ...

    def extra_config(self, clf, X):
        """Return dict of extra fields for config.json (can be empty)."""
        return {}
```

### prepare(df) — the only safe pattern

`y` MUST be extracted from `df` *before* any transformation that could
remove the `target` column. Use this template literally; only modify
the inner "feature engineering" section:

```python
_SKIP = {"target", "signal_id", "event_timestamp"}

def prepare(self, df: pd.DataFrame):
    # 1. Extract the target FIRST. Don't overwrite `df` before this line.
    y = df["target"].astype(int)

    # 2. Build the feature frame from a *copy* of df, leaving the
    #    original df (with target intact) alone.
    X = df.drop(columns=_SKIP, errors="ignore").copy()

    # 3. (Optional) feature engineering on X here — interactions,
    #    polynomial terms, log transforms, derived columns. Modify
    #    X freely but DO NOT touch `df` or re-introduce `target` to X.
    #    Example:
    #    X["rooms_per_household"] = X["total_rooms"] / X["households"]

    return X, y
```

### Common failure modes — DO NOT do these

- **Don't drop target then try to read it.** This is the most common
  bug:
  ```python
  df = df.drop(columns=["target", ...])   # target gone from df
  X = df
  y = df["target"]                         # KeyError: 'target'
  ```
  Extract `y` BEFORE any drop / transform that could remove it.

- **Don't apply pandas one-hot or scaling to the whole df at once.**
  `pd.get_dummies(df)` rewrites every column including possibly target
  if it's categorical. Apply transforms to `X` after the y/X split.

- **Don't return a numpy array from prepare().** The harness expects
  `X` to be a `pd.DataFrame` so `list(X.columns)` works downstream
  for the bundle config.

## Strategy hints

- **Start broad, then refine.** Try several estimator classes early
  before fine-tuning hyperparameters of one.
- **Watch for high variance.** The dataset is small (208 rows); deep
  trees and high-degree polynomials overfit easily.
- **Don't repeat experiments.** Look at the recent-experiments list —
  if a configuration was reverted, don't propose the same one again.
- **Try preprocessing.** `StandardScaler` is cheap and often helps
  distance-based models (SVC, LogisticRegression). `PCA` can help
  when features are correlated (sonar bands often are).
- **One big change per iteration.** If you change both the estimator
  and add preprocessing in one step, you won't know which helped.
- **If you've plateaued for several iterations**, try something
  qualitatively different (different model family, different feature
  representation).

## Metric

The trainer prints `validation_accuracy=0.XXXX`. Higher is better.
That single number decides keep vs revert.

The current baseline (RandomForest with default sklearn params) sits
around 0.79–0.86 depending on the train/test split. Aim higher.
