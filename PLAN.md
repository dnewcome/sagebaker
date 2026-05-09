# Plan

## Phase 1 — Regression task + plugin (DONE 2026-05-08)

Goal: extend sage-baker beyond binary/multiclass classification so the
plugin abstraction covers the full supervised-learning shape. Forces
out hidden classification-specific assumptions in the harness.

### Dataset

`sklearn.datasets.fetch_california_housing` — 8 numeric features
(median income, house age, etc.), continuous `target` (median house
value), ~20K rows, sklearn-bundled (no download). Public, no auth,
runs on any clone.

### Refactor scope (what's classification-only today)

| File | Classification assumption |
| ---- | ------------------------- |
| `src/train.py` | imports `accuracy_score` directly, prints `validation_accuracy=…`, records `clf.classes_` in config.json |
| `evaluate.py` | precision/recall/f1, casts target to int, dispatches binary vs macro by class count |
| `local_serve.py` | counts predicted-vs-actual "matches" |
| `agent.py` | parses `validation_accuracy=` from stdout |
| `src/plugins/base.py` | docstring says "integer labels" |
| `src/plugins/default.py` | `RandomForestClassifier`, `target.astype(int)` |
| `program.md` | strategy hints all assume classification |

### Approach

**Plugins declare their task and own their metric.** Add to the base:

```python
class TrainingPlugin:
    task: str = "classification"   # or "regression"
    def evaluate(self, y_true, y_pred) -> tuple[str, float]:
        # default = accuracy; regression plugins override
        ...
```

Higher-is-better convention everywhere (R² for regression — bounded
above by 1, comparable across datasets, agent's `>` comparison still
makes sense).

Trainer prints `validation_<metric>=…` (the plugin chose the name)
and the agent's grep widens to match any `validation_\w+=` line.

### Files to add

- `prepare_housing.py` — writes `data/california.csv` + `lineage.json`
- `src/plugins/housing.py` — `HousingPlugin(task="regression")` using
  `GradientBoostingRegressor`. Overrides `evaluate` to return R².
- `program_regression.md` — agent constraints + strategy hints for
  the regression case (different model classes, no class imbalance
  concern, RMSE/R² intuition)

### Files to modify

- `src/plugins/base.py` — add `task` attr + `evaluate()` default
- `src/plugins/default.py` — explicit `task = "classification"`,
  override `evaluate()` to record the explicit accuracy default
- `src/train.py` — call `plugin.evaluate(y_test, clf.predict(X_test))`
  instead of hardcoded `accuracy_score`. Skip `clf.classes_` for
  regression. Use the metric name in the log/metadata.
- `evaluate.py` — read `task` from config.json, dispatch metrics
  accordingly (classification: precision/recall/f1; regression:
  R²/RMSE/MAE)
- `local_serve.py` — dispatch on task (regression: print residuals
  not "matches")
- `agent.py` — broaden the metric regex
- `Makefile` — `data-housing`, `train-housing` targets

### Smoke test

```bash
make data-housing
make train MODEL_DIR=./models/housing --plugin housing
make serve MODEL_DIR=./models/housing
.venv/bin/python evaluate.py --model ./models/housing --test ./data --output ./eval_housing
cat ./eval_housing/metrics.json
```

Expectation: bundle has `task: "regression"` and `validation_r2` in
metadata; `local_serve` shows residuals; `evaluate.py` produces R² +
RMSE + MAE.

---

## Phase 2 — Public-data recommender (DONE 2026-05-08)

Goal: make `make train-als` work without synthetic data, on a real
public dataset, so the recommender path is exercisable on a clone.

### Result

`prep/prepare_movielens.py` fetches MovieLens-100K (1.7 MB, no auth) from
grouplens.org, maps the schema to what the ALS plugin already
expects (`user_id` / `item_id` / `weight` + bonus `timestamp`),
writes `data/movielens.csv` + `lineage.json`. ALS plugin needed zero
changes.

End-to-end:

```
make data-movielens && make install-recommender && make train-als
```

Metrics on the real data: hit_rate@10 = 0.82, recall@10 = 0.16,
ndcg@10 = 0.21 — within the typical ALS-on-MovieLens-100K range.

Single make target added: `data-movielens`.

---

## Phase 3 — Smoke test suite (now)

Goal: lock in the contracts that this session has stabilized — bundle
round-trip, plugin protocol, evaluate-signature dispatch, config-only
model rebuild — so future edits surface regressions instead of
silently breaking the `/productionize` skill or `agent.py` loop.

### Tests to write

- **`test_bundle_roundtrip.py`** — train HousingPlugin on a 200-row
  fixture, save bundle, reload via `train.model_fn`, assert
  predictions match. Covers the joblib path; one parametrized
  variant covers the skops path.
- **`test_plugin_contract.py`** — every registered supervised plugin
  has `name`, `task` ∈ {classification, regression}; `prepare(df)`
  returns `(DataFrame, Series)`; `build_model({})` returns a fittable
  estimator with `.fit` / `.predict`.
- **`test_evaluate_signatures.py`** — `train.py` accepts both
  `evaluate(y_true, y_pred)` (legacy) and
  `evaluate(model, X_test, y_true)` (new). Exercises the
  `inspect.signature` dispatch.
- **`test_config_rebuild.py`** — the `/productionize` invariant: rebuild
  the estimator from `config.json` (class + params, no pickle) → fit
  → predictions are byte-identical to the bundled weights. If this
  fails, `/productionize` is lying.
- **`test_lineage_capture.py`** — prepare-script writes `lineage.json`,
  `train.py` reads it, lineage shows up in `metadata.json`.

### Constraints

- Run in <30s on a clean clone (HousingPlugin's `prepare_data()` is
  sklearn-bundled, no network).
- No fixtures larger than ~200 rows.
- Use `pytest`. Already installed in the venv.

### Files to add

- `tests/conftest.py` — shared fixtures: tmp model dir, the housing
  dataframe, a mock plugin for the signature-dispatch test.
- `tests/test_*.py` — one file per area above.
- `requirements-dev.txt` — `pytest` (and that's it for now).
- `Makefile` target `test` — runs `pytest -q tests/`.

### Out of scope this phase

- GitHub Actions wiring. Easy follow-on once the suite is green
  locally; just adds `pip install -r requirements.txt
  -r requirements-dev.txt && make test` to a workflow.
- Recommender plugin tests (different harness, separate file shape).
- BigQuery / Feast / DLC tests — these need credentials or extra
  dependencies; they'd be integration tests, not smoke.

---

## Phase 4+ candidates (not committed to)

Stuff that came up in the planning conversation as natural follow-ons.
Listed in rough value × ease order; pick whatever the next push needs.

- **Per-dataset agent Makefile targets** — `make agent-housing`,
  `make agent-sonar`, `make agent-bq`. One-line shortcuts for the
  multi-dataset workflow.
- **Run history persistence** — `runs/<plugin>/<timestamp>.json` per
  iteration of `agent.py`. Today the loop is amnesic between
  invocations; persisting it lets you compare strategies across runs
  and is the data layer the eventual REST/MCP store would query.
- **Dataset fingerprinting helper** — small util that hashes
  `(shape, target_dtype, target_distribution, feature_dtype_counts)`.
  Prerequisite for similarity-based recall.
- **Knowledge backing store (REST first, MCP later)** — once run
  history + fingerprints exist there's something real to serve. REST
  is the simpler protocol to iterate schema on; MCP shim layers on
  top once the surface stabilizes.
- **GitHub Actions CI** — wraps Phase 3 once the test suite is green.

---

## Out of scope for the foreseeable future

- **Hyperparameter tuning beyond agent.py** — Optuna / sklearn
  HalvingGridSearch could plug into the same harness later.
- **More public datasets** (adult-income for categoricals,
  fashion-mnist for torch) — same pattern as housing once the
  abstraction is in place.
- **Drift detection** (Evidently/TFDV) — orthogonal; do once we have
  enough deployed-model usage to make it relevant.
