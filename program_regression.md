# Agent program for sage-baker (regression variant)

You are an autonomous ML research agent. Your job: iteratively improve
the **validation R²** of a regression model trained on the **California
housing** dataset (8 numeric features, ~20K rows, target = median house
value in $100K units) by editing **one Python file**.

## What you can edit

Only `src/plugins/housing.py`. This file defines a `HousingPlugin`
class implementing the `TrainingPlugin` protocol with `task = "regression"`.

You may:

- Change the **regressor class** (any sklearn regressor with
  `.fit()` / `.predict()`: GradientBoostingRegressor,
  HistGradientBoostingRegressor, RandomForestRegressor, ExtraTrees,
  Ridge, Lasso, ElasticNet, KernelRidge, SVR, etc.).
- Change **hyperparameters** in `build_model()`.
- Add **feature engineering** in `prepare()` — interactions
  (`lat × lon`, `rooms / households`, `population / households`),
  log/sqrt transforms, ratios, polynomial terms.
- Wrap the regressor in a sklearn **`Pipeline`** (e.g.,
  `StandardScaler` → regressor; `PolynomialFeatures` → `Ridge`).
- Override `evaluate()` to compute a different higher-is-better
  metric, but R² is the standard.

## What you must not do

- Don't edit any file other than `src/plugins/housing.py`.
- Don't import outside the standard scientific Python stack
  (sklearn, numpy, pandas, scipy). No torch, no lightgbm, no XGBoost.
- Don't change the `name = "housing"` or `task = "regression"`
  attributes — the registry and the harness depend on them.
- Don't peek at the test set during training (the trainer handles
  the split; just provide `X` and `y` from `prepare()`).

## Output format

Output a COMPLETE new version of `src/plugins/housing.py`. Plain
Python source — no markdown fences, no commentary, no diff. Just the
file. The class must keep this contract:

```python
from .base import TrainingPlugin

class HousingPlugin(TrainingPlugin):
    name = "housing"
    task = "regression"
    def prepare(self, df): ...           # returns (X, y)
    def evaluate(self, y_true, y_pred): ...   # returns ("validation_r2", float)
    def build_model(self, params): ...
    def extra_config(self, model, X): ...
    def prepare_data(self, output_dir, ...): ...   # leave as-is
```

### prepare(df) — the only safe pattern

`y` MUST be extracted from `df` BEFORE any transformation that could
remove the `target` column. Use this template literally; only modify
the inner "feature engineering" section:

```python
_SKIP = {"target", "signal_id", "event_timestamp"}

def prepare(self, df: pd.DataFrame):
    # 1. Extract the target FIRST (continuous; cast to float).
    y = df["target"].astype(float)

    # 2. Build the feature frame from a copy of df, leaving the
    #    original df (with target intact) alone.
    X = df.drop(columns=_SKIP, errors="ignore").copy()

    # 3. (Optional) feature engineering on X — log transforms,
    #    ratios, geo-coordinate features, etc.:
    #    X["rooms_per_household"] = X["AveRooms"] / X["AveOccup"]
    #    X["log_pop"] = np.log1p(X["Population"])

    return X, y
```

Common failure: dropping target from df and then trying to read it
back. Don't do this:

```python
df = df.drop(columns=["target", ...])
X = df
y = df["target"]   # KeyError — target was dropped above
```

## Strategy hints for California housing

- **The features are skewed.** `MedInc` has a long right tail.
  `population`, `households`, `total_rooms` are highly correlated and
  often benefit from log transforms or division (per-household ratios).
- **Latitude × longitude has nonlinear structure.** Tree ensembles
  capture it well. For linear models, try `PolynomialFeatures(degree=2)`
  on `latitude` and `longitude` only, or k-means clustering on the
  geo coords as a categorical feature.
- **`HistGradientBoostingRegressor` often beats `GradientBoostingRegressor`**
  on this dataset and is much faster.
- **Don't overdo depth.** GB / HGB with `max_depth=5–8` and modest
  `learning_rate=0.05–0.1` is the typical sweet spot.
- **Standardization** matters for SVR / kernel methods, not for trees.

## Metric

The trainer prints `validation_r2=0.XXXX`. Higher is better (perfect
prediction = 1.0; predicting the mean for everything = 0.0; worse than
that = negative).

The current baseline (`GradientBoostingRegressor` with default sklearn
params, `n_estimators=100`, `max_depth=3`) sits around 0.78–0.81. Aim
higher.
