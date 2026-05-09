# sagebaker — researcher playbook

A sequential walkthrough: from `git clone` to a model trained,
tracked in MLflow, and serving inference on SageMaker. Each stage has
a goal, the commands to run, and what "this stage worked" looks like.

This is the *happy path*. When something doesn't fit, peek at the
README for depth, the relevant example for a parallel, or `make help`
for what's available.

## Audience

A researcher who:
- Has trained ML models before (sklearn / torch / similar).
- Is new to this codebase.
- Wants to bring their own model — data prep, plugin, deploy — without
  re-deriving the whole architecture.

## Stage 0 — Setup (one-time)

```bash
git clone git@github.com:<you>/sagebaker.git && cd sagebaker
make install                                     # base venv + deps
make install-jupyter                             # for the productionize workflow
make install-mlflow 2>/dev/null || true          # if you'll use MLflow tracking
.venv/bin/pip install -e .                       # editable install — `import bundle, plugins` works venv-wide
```

Add a `.env` (gitignored) with whatever credentials you'll need:

```bash
# only if you're using BigQuery for data
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
GOOGLE_CLOUD_PROJECT=your-project

# only if you're deploying to MLflow tracking server
MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# only if you're deploying to SageMaker
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

**Success signal:** `make help` lists all targets. `.venv/bin/pytest -q`
returns 14 passed.

## Stage 1 — Run an example end-to-end

Goal: prove the whole pipeline works on your machine before you
introduce any of your own changes.

```bash
make data-sonar         # downloads + preps the binary classification example
make train              # trains the default plugin → models/sklearn/
ls models/sklearn/      # config.json  metadata.json  model.joblib
.venv/bin/python local_serve.py --model-dir ./models/sklearn  # round-trips
```

**Success signal:** `validation_accuracy=0.85` (give or take); `local_serve.py`
prints predictions matching ground truth on a held-out slice.

If this stage fails, stop and fix it before going further. The
later stages all assume this works.

## Stage 2 — Pick the closest existing example

Goal: find an example that already does *most* of what you want, so
you're modifying instead of starting from scratch.

```bash
ls examples/
# conversion-prediction  product-matching  semantic-search
# record-linkage         recommender       regression
```

Read `examples/README.md` — it maps each example to a problem shape:

| Your problem | Closest example |
|---|---|
| Predict a categorical label from tabular features | `conversion-prediction` |
| Predict a continuous value from tabular features | `regression` |
| Same canonical entity? (pair-level binary) | `product-matching` |
| Top-K relevant items per user (collaborative filtering) | `recommender` |
| Find similar items by free-text query | `semantic-search` |
| Link anonymous events to the same true user | `record-linkage` |

Run that example's quickstart end-to-end. **Read the plugin file**
(`src/plugins/<name>.py`) — it's usually <100 lines and tells you
exactly what shape you're targeting.

**Success signal:** you can articulate which plugin family you're in
(supervised / recommender / retrieval) and which existing plugin
you'll use as the structural template.

## Stage 3 — Drop in your plugin

Goal: a `MyProjectPlugin` that fits the contract, lives in
`src/plugins/private/` (gitignored — safe for company-internal code).

### 3.1 — Create the plugin file

```bash
mkdir -p src/plugins/private
cp src/plugins/default.py src/plugins/private/myproject.py
# edit: change class name, name="myproject", target column, build_model(), prepare()
```

The contract you have to fill (from `src/plugins/base.py`):

```python
class TrainingPlugin:
    name: str = "myproject"          # registry key
    task: str = "classification"     # or "regression"

    def prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split df into (X, y). Drop unused columns. Cast types."""
        ...

    def build_model(self, params: dict):
        """Return an unfitted estimator with .fit() / .predict()."""
        ...

    def evaluate(self, model, X_test, y_true) -> tuple[str, float]:
        """Return (metric_name, value). Higher = better."""
        ...
```

Plugins in `private/` are auto-discovered at import time — no need
to register them anywhere.

### 3.2 — Create the prep script

```bash
cp prep/prepare_sonar.py prep/prepare_myproject.py
# edit: source query / file, output path, lineage capture
```

The prep script's job: read raw data, write
`data/myproject/training.parquet` + `data/myproject/lineage.json`.
The lineage file is what makes runs reproducible later — record where
the data came from (BQ query + snapshot timestamp, or the URL +
fetched_at, or the input CSV's sha256).

### 3.3 — Wire up the Makefile

Optional but useful — gives you `make data-myproject` /
`make train-myproject` shortcuts:

```makefile
# in Makefile.private (gitignored)
data-myproject:
	$(PY) prep/prepare_myproject.py

train-myproject: ## Train the myproject plugin
	$(PY) src/train.py --train ./data/myproject \
	                   --model-dir ./models/myproject \
	                   --plugin myproject
```

**Success signal:** `make data-myproject` writes to `data/myproject/`
and `train-myproject` produces a `models/myproject/` bundle with
`config.json` listing your estimator and feature names.

## Stage 4 — Validate locally

Goal: prove the bundle is faithful — the metric is real, the bundle
is reproducible, and config-only rebuild matches the pickled weights.

```bash
make train-myproject

# auto-generate a productionize notebook
/productionize myproject     # in a Claude Code session

# or do the same checks manually:
.venv/bin/python -c "
import json, joblib, importlib, numpy as np
config = json.load(open('models/myproject/config.json'))
mod = importlib.import_module(config['estimator_module'])
EstimatorClass = getattr(mod, config['estimator'])
rebuilt = EstimatorClass(**config['params'])
print('rebuild from config OK:', rebuilt.__class__.__name__)
"
```

The productionize notebook in `notebooks/myproject_productionize.ipynb`
gives you the bundle inspect, data reload, config-rebuild,
sanity-check-vs-pickle, EDA, feature importance, and production
checklist — all auto-generated.

**Success signal:** the productionize notebook's
`np.testing.assert_allclose(...)` cell passes — predictions from the
config-rebuilt model match the pickled weights to 1e-5.

If this fails, the bundle is *lying* about its contents — either the
plugin's `params` are incomplete, or the trainer is doing something
the config doesn't capture. Fix this before deploying.

## Stage 5 — Track with MLflow

Goal: every training run is logged with params, metrics, and the
bundle as an artifact, so you can compare runs and promote winners.

### 5.1 — Start a tracking server

```bash
# terminal 1: tracking server (reads/writes mlflow.db locally)
make mlflow-server
# → http://127.0.0.1:5000
```

### 5.2 — Re-run training with MLflow active

```bash
# terminal 2: training; .env-loaded MLFLOW_TRACKING_URI points at the server
make train-myproject
```

The training driver auto-logs:
- All `params` from `build_model()`
- The metric returned by `evaluate()`
- The bundle (`models/myproject/`) as an MLflow artifact
- A registered model version under name `sagebaker-myproject` (configurable)

Browse runs at `http://127.0.0.1:5000`. Promote a version through the
registry stages (None → Staging → Production) once you're happy.

### 5.3 — Load by registry URI for inference

```bash
.venv/bin/python mlflow_serve.py --name sagebaker-myproject --version latest
# loads the bundle from MLflow Registry, runs predictions on a sample
```

**Success signal:** `mlflow.pyfunc.load_model("models:/sagebaker-myproject/Production")`
returns a wrapped model whose `.predict()` matches what the local
bundle returns.

## Stage 6 — Deploy to SageMaker

Goal: the same bundle running in a SageMaker endpoint, callable from
production code via the `boto3` SageMaker runtime client.

There are two paths. **BYOC** is simpler (you control the container);
**DLC** matches typical work-environment pipelines (AWS pre-built
container with a pinned framework version).

### Path A — BYOC (easier first deploy)

```bash
make sm-build           # builds a sagemaker-compatible Docker image
                        # of your trainer + bundle.py
.venv/bin/python drivers/local_train.py \
    --plugin myproject --train ./data/myproject \
    --model-dir ./models/myproject
# runs the BYOC trainer in SageMaker Local Mode (no AWS account needed)
```

Once that works, push the image to ECR and run a real Training Job.

### Path B — DLC (matches AWS-canonical pipelines)

```bash
.venv/bin/python drivers/local_train_dlc.py
# uses the AWS scikit-learn DLC; src/ is uploaded as source_dir
```

DLC won't pip-install anything you didn't add to a `requirements.txt`
inside `src/` — see the README warning about leaking the outer
`requirements.txt` (it'll upgrade numpy and break sklearn).

### 6.3 — Deploy the endpoint

```bash
.venv/bin/python deploy_endpoint.py \
    --model-package-arn <arn-from-MLflow-or-build-step> \
    --endpoint-name sagebaker-myproject-prod
```

### 6.4 — Smoke test the endpoint

```bash
.venv/bin/python -c "
import boto3, json
client = boto3.client('sagemaker-runtime')
response = client.invoke_endpoint(
    EndpointName='sagebaker-myproject-prod',
    ContentType='application/json',
    Body=json.dumps({'instances': [[...your features...]]}),
)
print(response['Body'].read())
"
```

**Success signal:** the endpoint returns a prediction whose value
matches what `local_serve.py` returns for the same input. If they
diverge, suspect framework version drift between host and container
(see the README's "training/serving skew" section).

## Stage 7 — Iterate

Goal: improve the model without re-doing the whole pipeline each time.

### Manual iteration
- Edit `src/plugins/private/myproject.py`
- `make train-myproject` → MLflow logs a new run
- `/productionize myproject` regenerates the analysis notebook
- Promote the new version in MLflow if it wins
- `deploy_endpoint.py` with the new version

### Agent-loop iteration
```bash
make agent PLUGIN=myproject PROGRAM=program_myproject.md \
           BUDGET_SECONDS=1800
```

The agent edits your plugin file in place, trains, keeps if better,
reverts otherwise. Five safety features (baseline anchor,
byte-identity guard, failure feedback, stuck signal, compact diff)
keep it converging instead of churning.

After convergence: `/productionize myproject` for the analysis,
promote in MLflow, redeploy.

## What success looks like at the end

- `models/myproject/` exists locally with config + metadata + weights.
- MLflow registry has a `sagebaker-myproject` model with one or more
  Production versions.
- `aws sagemaker describe-endpoint --endpoint-name sagebaker-myproject-prod`
  returns `EndpointStatus: InService`.
- Predictions from `local_serve.py`, `mlflow_serve.py`, and the
  SageMaker endpoint all agree to numerical tolerance on the same
  inputs.

## Where it goes wrong (common gotchas)

- **`pickle` version mismatch on load.** The bundle's `framework_version`
  field tells you what was trained; the inference container has to
  match. Either pin or use `weights_format: skops` (allowlist-based
  pickle) or LightGBM's text format (no pickle).
- **Lineage file missing at training time.** `train.py` looks for
  `data/<plugin>/lineage.json` — if your prep script forgot to write
  it, lineage doesn't end up in `metadata.json` and reproducibility
  is on you.
- **MLflow logs but model doesn't appear in registry.** The trainer
  logs runs by default but only registers when `MLFLOW_REGISTERED_MODEL`
  is set — see the README's MLflow section.
- **SageMaker container can't import your plugin.** The container
  uploads `src/` as `source_dir` (flat). If your plugin is in
  `src/plugins/private/`, it goes along; if it's outside `src/`, it
  doesn't.
- **`requirements.txt` inside `src/` blows up the DLC container.**
  See README — SageMaker auto-pip-installs anything in `source_dir`,
  which can upgrade numpy and binary-incompatibilize sklearn/pandas.
  Don't put a top-level `requirements.txt` in `src/`.

## See also

- [OVERVIEW.md](OVERVIEW.md) — mental model when the project feels too big
- [README.md](README.md) — full reference, deep dives, command recipes
- [PLAN.md](PLAN.md) — phased roadmap and what's planned next
- [examples/README.md](examples/README.md) — problem-to-plugin mapping
