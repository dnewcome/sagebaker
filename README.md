# sage-baker

Local SageMaker training sandbox using **SageMaker Local Mode**, with two
interchangeable paths: a **bring-your-own-container (BYOC)** image for fully
offline use, and the **AWS Deep Learning Container (DLC)** image for
production-parity workflows.

## What this is

SageMaker has a "Local Mode" where the SDK runs training/inference jobs in
Docker containers on your machine using the same `/opt/ml/...` directory
contracts as real SageMaker. By default it pulls Deep Learning Container
(DLC) images from a regional ECR registry, which requires real AWS credentials
(any account works — the images are public-read, but ECR still demands a real
auth token to issue the pull).

This project supports both:

- **BYOC** (`local_train.py`) — small local image, follows the algorithm
  container contract (a `train` command on `PATH` reading `/opt/ml/input/`
  and writing `/opt/ml/model/`). Fully offline. No AWS account needed.
- **DLC** (`local_train_dlc.py`) — official AWS scikit-learn DLC image.
  Requires real AWS creds for the initial ECR pull; runs locally after that.
  Uses the `entry_point` + `source_dir` flow, so edits to `train.py` don't
  require any rebuild.

## Repo layout

```
Dockerfile             minimal Python + scikit-learn image with a `train` command (BYOC)
src/bundle.py          generic helpers for the standard model-bundle layout
src/tracking.py        opt-in MLflow tracking helpers (no-op when unconfigured)
src/train.py           sklearn training example — bundle layout, model_fn
src/train_torch.py     torch training example — same bundle layout, safetensors weights
src/train_lightgbm.py  LightGBM example — pickle-free native text format
src/train_feast.py     sklearn trainer pulling features via Feast (point-in-time join)
feature_repo/          Feast feature definitions (entities.py, features.py, store.yaml)
prepare_data.py        writes data/iris.csv + lineage.json (toy multiclass dataset)
prepare_sonar.py       writes data/sonar.csv + Feast parquets + lineage.json
prepare_bigquery.py    materializes a BigQuery query to parquet + lineage.json
demo_categorical.py    runnable demo of "new enum value at inference" bug + 3 fixes
local_train.py         BYOC driver — uses the local image, no AWS account
local_train_dlc.py     DLC driver  — uses the AWS scikit-learn DLC image
local_train_feast_dlc.py DLC + Feast — host-side feature retrieval, container trains
local_serve.py         placeholder — does not work yet (see "Serving", below)
requirements.txt       sagemaker<3, boto3, mlflow, scikit-learn, pandas, docker
requirements-torch.txt opt-in extras for the torch example: torch, safetensors
requirements-lightgbm.txt opt-in extras for the LightGBM example: lightgbm
requirements-skops.txt    opt-in: skops (safer-pickle for sklearn)
requirements-feast.txt opt-in extras for the Feast feature-store example
requirements-bigquery.txt opt-in: google-cloud-bigquery + db-dtypes
requirements-jupyter.txt  opt-in: jupyterlab + ipykernel + matplotlib + seaborn
```

The training script lives in `src/` so the DLC's `source_dir` can point at a
clean directory containing only training code. SageMaker auto-`pip install`s
any `requirements.txt` inside `source_dir`; if the project's outer
`requirements.txt` (sagemaker, boto3, etc.) leaks into the container it
upgrades numpy and binary-incompatibilizes the pre-installed sklearn/pandas.
Don't put a `requirements.txt` inside `src/` unless you genuinely need extra
deps in the training container.

The training script follows SageMaker conventions:

| Path                                          | Purpose                              |
| --------------------------------------------- | ------------------------------------ |
| `/opt/ml/input/data/<channel>/`               | Input data (mounted per channel)     |
| `/opt/ml/input/config/hyperparameters.json`   | Hyperparameters (string-typed)       |
| `/opt/ml/model/`                              | Where the model is written           |
| `/opt/ml/output/`                             | Where failure outputs go             |

## Setup

Requirements: Docker, Python 3.10+, ~1 GB free disk. There's a Makefile
covering all the common workflows; `make help` lists them.

```bash
make install              # base venv + main requirements
# add any optional extras you want:
make install-torch        # torch + safetensors
make install-lightgbm     # LightGBM
make install-skops        # safer-pickle for sklearn
make install-feast        # Feast feature-store
make install-bigquery     # google-cloud-bigquery
make install-jupyter      # jupyterlab + ipykernel + matplotlib
# or grab everything in one go:
make install-all
```

Equivalent without `make`: `python3 -m venv .venv && .venv/bin/pip
install -r requirements.txt` plus any of the `requirements-*.txt`
files you want.

## Running training

Generate a dataset (pick one — each script wipes `data/` and writes one CSV):

```bash
make data-iris      # iris (3-class, 150 rows)
# or
make data-sonar     # Rocks vs Mines (binary, 208 rows; also writes Feast parquets)
```

`train.py` reads whichever CSV is in the train channel — no code changes
needed to swap datasets, as long as the CSV has a `target` column.

### BYOC (offline)

```bash
make train-byoc     # builds the image if needed, runs local_train.py
```

### DLC (with AWS credentials)

AWS now recommends IAM Identity Center (SSO) with short-lived credentials
over long-lived access keys. One-time setup:

```bash
aws configure sso                 # creates a profile entry in ~/.aws/config
```

Then before each session:

```bash
aws sso login --profile your-profile
export AWS_PROFILE=your-profile
make train-dlc
```

(Long-lived `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars still work
if you need them.)

The DLC image (~3 GB) is pulled once and cached in your local Docker daemon;
subsequent runs are offline.

Both paths produce the same shape of output:

```
algo-1-XXXX  | validation_accuracy=1.0000
algo-1-XXXX exited with code 0
model artifact: file:///.../sage-baker/.sm-scratch/.../compressed_artifacts/model.tar.gz
```

The model artifact (`model.tar.gz` containing `model.joblib`) lives under
`.sm-scratch/`.

## Training paths at a glance

```mermaid
flowchart LR
  CSV[data/sonar.csv]
  PQ[(feature_repo/<br/>parquets)]

  subgraph byoc_path["BYOC &nbsp;<i>local_train.py</i>"]
    direction TB
    BD[local Docker image<br/>sage-baker-sklearn]
    BC["container<br/>train.py reads CSV/parquet<br/>writes /opt/ml/model"]
  end

  subgraph dlc_path["DLC &nbsp;<i>local_train_dlc.py</i>"]
    direction TB
    DD[AWS DLC pulled from ECR]
    DC["container<br/>SKLearn entry_point runs<br/>src/train.py"]
  end

  subgraph feast_path["DLC + Feast &nbsp;<i>local_train_feast_dlc.py</i>"]
    direction TB
    FH[host: Feast<br/>get_historical_features]
    MAT[materialized.parquet]
    FC["container<br/>train.py reads parquet"]
  end

  CSV --> BD --> BC --> B_OUT([bundle])
  CSV --> DD --> DC --> D_OUT([bundle])
  PQ --> FH --> MAT --> FC --> F_OUT([bundle])
```

All three paths converge on the same `model_dir/` bundle layout — the
`model_fn(model_dir)` loader doesn't care which path produced it.

## When to use which

| Use BYOC when …                              | Use DLC when …                              |
| -------------------------------------------- | ------------------------------------------- |
| no AWS credentials available                 | you have any AWS account                    |
| you want a small (~200 MB) image             | image size doesn't matter                   |
| deps are simple (sklearn / pandas / etc.)    | you want AWS-tested framework + GPU stack   |
| training script is stable                    | you want to iterate on `train.py` fast      |
| serving doesn't matter                       | you want `/ping` + `/invocations` for free  |

The big practical wins of DLC are the `entry_point` flow (no rebuilds on
script edits) and the inference toolkit (working serving in one `.deploy()`
call). BYOC's wins are zero AWS dependency and a small, fully-controlled
image.

## Architecture: separating code from weights

The single most important design decision in a training system is **what
gets persisted with the model and what stays in code**. Pickle (and
`torch.save(model, ...)`, and the default `mlflow.<flavor>.log_model`)
freezes the running Python object — including a reference to the class
that defined it. When the class moves, gets renamed, or has a method
added, unpickling either explodes or silently loads the wrong thing. This
is the trap where every code change forces a retrain.

The fix is a layered model: **code in git, weights as data, config as JSON.**

There is no single industry-standard format for this; the closest thing is
HuggingFace's `save_pretrained` / `from_pretrained` (which writes
`config.json` + `model.safetensors` + tokenizer files), and the layout in
this repo is an extension of that idea. MLflow's "MLmodel" format, TF's
SavedModel, TorchServe's `.mar`, and ONNX are all alternatives at different
abstraction levels — none of them solve the code/weights coupling problem
unless you opt out of their default flavors.

```mermaid
flowchart LR
  subgraph code["Code (git, hot-editable)"]
    direction TB
    CL[Model class<br/>SonarMLP / RFClassifier]
    LB[bundle.py helpers]
    LD[model_fn loader]
  end

  subgraph dir["model/ &nbsp; (bundle on disk)"]
    direction TB
    CFG[config.json<br/><i>class + init args + weights pointer</i>]
    W[model.safetensors<br/>or model.joblib<br/><i>just numbers</i>]
    META[metadata.json<br/><i>git sha, metrics, timestamp</i>]
  end

  CL -.->|"save_config({class, params, weights_file})"| CFG
  LB -.-> CFG
  LB -.-> META
  CL -->|"save_file / joblib.dump"| W

  CFG -->|"read"| LD
  W -->|"read"| LD
  CL -.->|"instantiate from class registry"| LD
  LD -->|"return"| OUT([assembled model])
```

### Bundle layout

The training script writes a directory with this shape, regardless of
framework:

```
model/
├── config.json           how to build the model (arch, hyperparams,
│                         weights_file pointer, feature schema)
├── <weights_file>        the actual numbers (model.joblib, model.safetensors, …)
├── preprocessor.json     [optional] preprocessor state (scaler stats,
│                         label maps, vocab refs)
└── metadata.json         provenance: timestamp, python version, git sha,
                          training metrics
```

`config.json` knows the name of the weights file. The loader reads the
config, instantiates the model class with those args, and loads weights
from the file the config points at. The class definition lives in your
repo — versioned, debuggable, hot-editable. Want to fix a bug in
`forward()`? Edit, reload weights, done. No retrain.

### Format choices

| Thing                          | Format                                       |
| ------------------------------ | -------------------------------------------- |
| Torch / JAX / TF tensors       | **`safetensors`** (mmap, no pickle, no RCE)  |
| Embeddings (raw arrays)        | safetensors or `.npz`                        |
| Tokenizers, HF models          | `tokenizer.save_pretrained(dir)`             |
| HF model end-to-end            | `model.save_pretrained(dir)` — already does the layout above |
| Hyperparameters / model config | JSON                                         |
| sklearn pipelines              | `joblib` (pickle) is least-bad — pin `scikit-learn==X.Y` and accept it. For *important* models, extract coefficients / tree structures into JSON manually. |

`safetensors` is the boring-good default for tensor data: zero-copy mmap,
no arbitrary code execution on load, supported by torch / JAX / TF / HF.

### Beyond pickle: alternatives by framework

The boring defaults above (joblib, safetensors, JSON) are the right
starting point. When they aren't enough — most often when you want
stricter security, version-decoupling, or framework-portability —
here's what to reach for.

#### sklearn

`joblib` is canonical for sklearn pipelines, but it *is* pickle under
the hood. Two real risks live here:

1. **Framework-version coupling.** Tree node formats, parameter
   layouts, and class names change across sklearn versions and can
   silently corrupt loads. We hit this live in this repo: a model
   trained with sklearn 1.2 in the DLC failed to load with sklearn
   1.3 on the host —
   `ValueError: node array from the pickle has an incompatible dtype`.
   The `framework_version` field in `config.json` is the warning
   signal. Mitigation: lock the trainer's sklearn version to the
   inference container's.
2. **Arbitrary code execution on load.** Pickle deserializes by
   importing classes and calling their constructors. A malicious pkl
   from an untrusted source = remote code execution.

Alternatives:

- **`skops`** — sklearn-team-blessed safer pickle. `skops.io.dump` /
  `skops.io.load` use an explicit allowlist of trusted classes and
  refuse anything else, fixing the RCE risk. Same shape as joblib;
  trivial drop-in. **Doesn't fix the version-coupling problem** — the
  underlying object format is still sklearn's. Wired in here as a
  swappable weights format:
  ```bash
  .venv/bin/pip install -r requirements-skops.txt
  .venv/bin/python src/train.py --train ./data --model-dir ./model_skops \
                               --weights-format skops
  .venv/bin/python local_serve.py --model-dir ./model_skops   # round-trips
  ```
  `train.py` dispatches the writer by flag; `model_fn(model_dir)`
  dispatches the reader by the `weights_format` field in `config.json`.
  No caller of `model_fn` knows or cares which format produced the
  weights — that's the whole point of the bundle's pointer-style design.
- **`skl2onnx`** — converts a fitted sklearn pipeline to ONNX. The
  resulting graph loads in ONNX Runtime without sklearn at all, fully
  decoupled from sklearn versions and from pickle. Cost: not every
  sklearn estimator has an ONNX path, and you lose sklearn-specific
  introspection (`feature_importances_`, `decision_function`, etc.).
- **Switch frameworks**: if you're willing to use **LightGBM** instead
  of sklearn, you escape the pickle problem entirely — see below.

#### LightGBM (and the boring-good answer to pickle)

If you're using a tree-based model on tabular data, LightGBM is
typically faster and more accurate than sklearn's RandomForest, and
its native serialization is **completely pickle-free**:

```python
booster.save_model("model.txt")             # human-readable text
booster = lgb.Booster(model_file="model.txt")
```

The text file is the trees, threshold values, and feature names laid
out as plain text — `cat model.txt` works. No Python class needed at
load time, no version-coupling, no RCE risk. Wired in here as
`src/train_lightgbm.py`:

```bash
.venv/bin/pip install -r requirements-lightgbm.txt
.venv/bin/python src/train_lightgbm.py --train ./data --model-dir ./model_lgb
.venv/bin/python local_serve.py --model-dir ./model_lgb   # round-trips
```

Same bundle envelope as the sklearn path — just `weights_file:
"model.txt"` instead of `model.joblib`. `local_serve.py` dispatches on
the bundle's `framework` field to pick the right loader automatically.
On sonar, LightGBM hits ~0.83 vs RandomForest's ~0.79 with default
hyperparameters; that's typical.

#### torch

For weights-only, `safetensors`. For "model in a box" — graph + weights,
loadable without the Python class — there are two mainstream paths:

- **TorchScript** (`torch.jit.script` or `torch.jit.trace` → `.pt`):
  serializes the forward graph alongside weights. The class definition
  isn't needed at load time. Tradeoff: only a JIT-compatible subset of
  Python works — you have to type-annotate (`script`) or write a
  trace-friendly forward (`trace`). Mature; common in mobile.
- **`torch.onnx.export`**: emits ONNX. Maximally portable — runs in
  ONNX Runtime (C++), TensorFlow.js, etc. Cost: not every torch
  operator maps to an ONNX op; you sometimes refactor the model to get
  a clean export.

#### Framework-agnostic

- **ONNX** (Open Neural Network Exchange) — the strongest answer to
  "I want the model independent of the training framework." Stores the
  computation graph + weights as protobuf. Visually inspectable with
  Netron. Loadable from C++ / Python / JS / Rust without requiring
  torch / TF / sklearn. Best when you need to serve in non-Python
  environments or want hard separation from training code.
- **TF SavedModel** — TF-specific but mature: graph + weights +
  signatures in a directory. Less relevant unless you're on TF.

In this bundle layout, you can swap the *weights file format* without
changing the bundle envelope: `weights_file: "model.onnx"` is just as
valid as `model.joblib` or `model.safetensors`. `model_fn(model_dir)`
reads `config.json` to know which loader to use. The code/weights/config
separation holds regardless of which weights serializer you pick.

### How this maps to other systems

- **SageMaker.** Whatever you put in `/opt/ml/model/` gets tarred to
  `model.tar.gz`. Write the bundle there during training; the inference
  container's `model_fn(model_dir)` calls your `load(model_dir)` to
  reassemble. Same function, two consumers.
- **MLflow.** Two clean paths. Either treat MLflow as tracking only and
  use `mlflow.log_artifacts("model/")` to log the bundle as opaque files —
  load via your own `load(dir)`, never `mlflow.X.load_model`. Or wrap
  `load(dir)` in a custom `mlflow.pyfunc.PythonModel` so
  `mlflow.pyfunc.load_model(uri)` does the right thing. The point is to
  never let MLflow pickle your class.
- **Plain `pickle` / `torch.save(model, ...)` / `mlflow.X.log_model`.**
  These are exactly what we're avoiding. They can't survive code changes
  and they're a remote-code-execution hazard on load.

### How this maps to MLflow

Two clean ways to use the bundle layout with MLflow tracking:

```python
# 1. MLflow as tracking only — log the bundle as opaque artifacts.
mlflow.log_artifacts("model/")
# Loading: mlflow.artifacts.download_artifacts(...) then call your load(dir).

# 2. Custom PyFunc — wrap bundle.load in mlflow.pyfunc.PythonModel.
class BundleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = your_load_fn(context.artifacts["bundle"])
    def predict(self, context, model_input):
        return self.model.predict(model_input)

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=BundleModel(),
    artifacts={"bundle": "model/"},
)
# Loading: mlflow.pyfunc.load_model(uri) — uses your loader, not pickle.
```

Either way, MLflow never touches your model class.

### What's in this repo

- `src/bundle.py` — generic JSON helpers (`save_config`, `load_config`,
  `save_metadata`, `load_metadata`). Framework-agnostic.
- `src/train.py` — sklearn example. Trains a `RandomForest`, writes the
  bundle via `bundle.py`, exposes `model_fn(model_dir)`. Weights stored
  as `model.joblib`.
- `src/train_torch.py` — torch example. Trains an MLP, writes the *same*
  bundle layout via `bundle.py`, exposes the *same* `model_fn(model_dir)`
  shape. Weights stored as `model.safetensors`. Run standalone with
  `python src/train_torch.py --train ./data --model-dir ./model_torch`
  (install deps via `pip install -r requirements-torch.txt` first).

The two trainers prove the point: the `model_fn(model_dir) -> model`
contract is identical across frameworks. The weights file format and the
class definition differ; the bundle envelope and the loader signature do
not. To add a new framework, drop another `src/train_<x>.py` that calls
the same `bundle.save_config(...)` / `bundle.save_metadata(...)` and
writes its weights with whatever format that framework prefers.

## Tracking with MLflow

The trainers call MLflow unconditionally via `src/tracking.py`, which
no-ops when `MLFLOW_TRACKING_URI` is unset. Set the env var to enable
logging — params, metrics, tags, and the full bundle dir are all captured.

Quickstart with a local SQLite-backed server (the file-based backend is
deprecated as of MLflow 3 — use SQLite even for trivial local use):

```bash
# terminal 1: start a local server
.venv/bin/mlflow server --host 127.0.0.1 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts

# terminal 2: train
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
.venv/bin/python src/train_torch.py --train ./data --model-dir ./model_torch
.venv/bin/python local_train.py     # BYOC — see "Inside docker" below

# browse runs in the UI
open http://127.0.0.1:5000
```

What gets logged:

- **Params** — hyperparameters, dataset filename
- **Metrics** — validation accuracy; per-epoch train_loss for torch
- **Tags** — `framework=sklearn|torch`, plus MLflow's auto-tags (git commit, source)
- **Artifacts** — the entire bundle dir (`config.json`, `metadata.json`,
  weights file). Loading happens via `model_fn(model_dir)`, never via
  `mlflow.X.load_model` — so MLflow doesn't pickle your class.

### Inside docker (BYOC / DLC)

The drivers (`local_train.py`, `local_train_dlc.py`) automatically pass
`MLFLOW_TRACKING_URI` through to the container if it's set on the host,
rewriting `localhost` / `127.0.0.1` to `host.docker.internal` so the
container can reach the host. This works out of the box on Mac and
Windows.

**Linux limitation.** Docker for Linux does *not* resolve
`host.docker.internal` by default (it's a Mac/Windows convenience). For
container-side MLflow logging on Linux you also need:

1. Bind the server to all interfaces:
   `mlflow server --host 0.0.0.0 --port 5000 ...`
2. Tell the trainer to use the host's LAN IP instead — set
   `MLFLOW_TRACKING_URI=http://<your-lan-ip>:5000` (find with
   `hostname -I | awk '{print $1}'`) before running the driver.

Both of those are external-config tweaks, not code changes here. For now
the simplest workflow on Linux is to log MLflow runs from host-side
trainer invocations (`python src/train_torch.py ...`) and use the BYOC
container only for testing the SageMaker-deployment-shaped path.

For a remote MLflow server (e.g. company's), no rewrite happens — the
driver passes the URL through unchanged.

## Feature store: Feast prototype

Feast solves three real problems training pipelines tend to botch:
**training/serving skew** (same feature computed differently in batch and
at inference), **point-in-time correctness** (joining features without
leaking future data into past examples), and **reusability** (define
features once, consume from many models).

This prototype runs entirely on free local components — SQLite + parquet
files — and translates to a SageMaker workflow by swapping backends:

| Component       | Local (here)            | SageMaker / production       |
| --------------- | ----------------------- | ---------------------------- |
| Offline store   | parquet files           | S3 (just change `path:` in `FileSource`) |
| Online store    | SQLite                  | Postgres on RDS, Redis on ElastiCache (~$15/mo each) |
| Registry        | local sqlite file       | S3 file or Postgres          |

Note: **DynamoDB is one of several online-store options, not required.**
Feast supports SQLite, Postgres, Redis, MySQL, Cassandra, and others.
This prototype skips DynamoDB entirely.

```mermaid
flowchart LR
  KAGGLE[Kaggle / UCI / ...<br/>CSV] --> PREP[prepare_sonar.py]
  PREP --> CSV[data/sonar.csv<br/><i>non-Feast trainers</i>]
  PREP --> FEAT[(feature_repo/<br/>sonar_features.parquet)]
  PREP --> LBL[(feature_repo/<br/>sonar_labels.parquet)]

  APPLY["feast apply<br/><i>registers entity + view</i>"]
  FEAT -.-> APPLY

  subgraph train["Training"]
    LBL --> EDF[entity_df]
    EDF --> JOIN["store.get_historical_features<br/><i>point-in-time join</i>"]
    FEAT --> JOIN
    JOIN --> ENRICHED[enriched dataframe]
    ENRICHED --> RF[RandomForest.fit]
    RF --> BUNDLE([model bundle<br/>config.feature_refs])
  end

  subgraph serve["Serving"]
    MATR["feast materialize-incremental<br/><i>push to online store</i>"]
    FEAT -.-> MATR --> ONLINE[(SQLite online store)]
    REQ[predict_one signal_id=N] --> ONLOOK["store.get_online_features"]
    ONLINE --> ONLOOK
    ONLOOK --> CLF[load bundle, predict]
    CLF --> PRED[prediction]
  end
```

### Setup

```bash
.venv/bin/pip install -r requirements-feast.txt
.venv/bin/python prepare_sonar.py    # also writes feast parquets

# register entities + feature views
cd feature_repo && ../.venv/bin/feast apply && cd ..

# push features to the online store (run after data updates)
cd feature_repo && ../.venv/bin/feast materialize-incremental \
    $(date -u +%Y-%m-%dT%H:%M:%S) && cd ..
```

### Train and serve via Feast

```bash
.venv/bin/python src/train_feast.py
```

The trainer:
1. Reads the labels parquet as the *entity dataframe* (signal_id +
   event_timestamp + target).
2. Calls `store.get_historical_features(...)` for the 60 sonar bands.
   Feast does a point-in-time join — features as of each row's
   `event_timestamp`.
3. Trains a RandomForest on the resulting frame.
4. Saves the bundle, recording `feature_refs` in `config.json`. The
   inference path (`predict_one(model_dir, signal_id)` in
   `src/train_feast.py`) re-reads those refs, calls
   `get_online_features(...)` for live lookup, and runs the model.

Same feature definitions used for both training and serving — that's
the whole point of a feature store.

### Why CSV → Parquet at prep time

You can keep importing CSVs from Kaggle (`prepare_sonar.py` still pulls
the same dataset). Feast's `FileSource` is parquet-native, so we convert
once at prep time. Keeps your data-import workflow identical and lets
Feast do its job.

### Inference: feature lookup by entity ID

`local_serve.py` is Feast-aware. If the bundle's `config.json` records
`feature_refs`, the script switches from "predict on raw rows" to
"look up features online by entity ID":

```bash
.venv/bin/python local_serve.py --model-dir ./model_feast --signal-ids 0,50,100,200
# → fetches f0..f59 for each signal_id from the SQLite online store,
#   then predicts. Same model_fn, different feature source.
```

This is the realistic serving shape. Critically, **Feast can return
None for features that aren't available** (TTL expired, not yet
materialized, missing in source data). Whether the model handles those
nulls gracefully is a *training-time* decision — train with realistic
missingness via `get_historical_features` (point-in-time) and your
inference path inherits the same null semantics.

### Feast on the DLC path

`local_train_feast_dlc.py` ties Feast and the DLC together using the
**pre-fetch pattern**: Feast retrieval happens on the host, the joined
dataframe is materialized to a parquet, and *that* parquet is what the
DLC training container consumes via the standard SageMaker train
channel. The container has no Feast install — it just reads parquet,
trains, saves the bundle.

```bash
.venv/bin/python local_train_feast_dlc.py
```

This is the pattern that translates to a real SageMaker Pipeline: a
`ProcessingStep` does Feast retrieval and writes parquet to S3, then a
`TrainingStep` consumes that parquet. Same shape as here, just with S3
in place of local files.

The other approaches we considered and skipped:

- **Pip-install Feast at training start** (drop a `requirements.txt`
  into `src/`). Risky — the auto-install upgrades numpy/pyarrow and
  shatters the DLC's pre-built sklearn/pandas wheels. Avoid.
- **Bake Feast into a custom image** (`FROM <DLC>` + `pip install
  feast`, push to your own ECR). Works, but tightly couples the trainer
  image to the feature-store backend and means two systems to debug.

### Translating to SageMaker

Three changes when you move this to a SageMaker workflow:

1. `feature_store.yaml`: `online_store.type` → `postgres` or `redis`,
   point at your RDS/ElastiCache endpoint.
2. `feature_repo/features.py`: `FileSource(path=...)` → `s3://bucket/key`.
3. The trainer image needs Feast installed and read access to the
   registry + offline store (S3 read perms, Postgres connect perms).

The trainer code itself doesn't change at all — that's what the feature
store buys you.

## Training data: warehouses, materialization, and lineage capture

Training against data in a warehouse (BigQuery, Snowflake, Redshift)
divides into three real concerns: **how to get it**, **how to keep
your training run reproducible**, and **how to capture what you trained
on so the model can be audited later**.

### The pattern: materialize, then train

```mermaid
flowchart LR
  WH[(BigQuery /<br/>Snowflake / etc.)]
  PREP[prepare_bigquery.py<br/><i>SQL → parquet</i>]
  PQ[data/training.parquet]
  LIN[data/lineage.json<br/><i>query, snapshot_ts, sha256</i>]
  TR[trainer]
  BD([bundle/<br/>metadata.json includes data_lineage])

  WH --> PREP --> PQ
  PREP --> LIN
  PQ --> TR
  LIN -.->|"bundle.load_lineage embeds<br/>into metadata extras"| TR
  TR --> BD
```

You *can* call `pandas.read_gbq("SELECT ...")` directly from your
trainer and skip materialization. It works, but it's a reproducibility
hazard — the warehouse is mutable, so re-running "the same query" days
later doesn't necessarily get you the same data. Use direct querying
for **exploration** (notebooks); materialize for any **registered
training run**.

### What lineage to capture

For the model to point back at the data it saw, the bundle needs:

| Field | Why |
| ----- | --- |
| `source` | warehouse name (bigquery/snowflake/url/sklearn/...) |
| `query` | the exact SQL string |
| `snapshot_timestamp` | for warehouses with time-travel (BQ's `FOR SYSTEM_TIME AS OF`, Snowflake's `AT(TIMESTAMP => ...)`), the anchor |
| `dataset_sha256` | hash of the materialized parquet file — drift detector |
| `dataset_n_rows` / `n_cols` | shape sanity check |

`prepare_bigquery.py` writes all of these to `data/lineage.json`;
`bundle.load_lineage()` reads it inside the trainer; the bundle's
`metadata.json` ends up with a `data_lineage` block. Inspecting any
trained model now answers "what did this train on?":

```bash
python src/train.py --train ./data --model-dir ./model_sklearn
cat model_sklearn/metadata.json | jq .data_lineage
# {
#   "source": "url",
#   "url": "https://...sonar.csv",
#   "fetched_at": "2026-05-07T...",
#   "dataset_sha256": "cf6f5dcf...",
#   "dataset_n_rows": 208
# }
```

The same metadata flows into MLflow as the run's logged metadata, so
you can search/filter runs by dataset hash in the MLflow UI too.

### BigQuery specifically

```bash
make install-bigquery

# auth: either gcloud ADC, or a service-account key via .env (see below)
gcloud auth application-default login   # one option
# - or -
cp .env.example .env                    # then edit GOOGLE_APPLICATION_CREDENTIALS

# default query hits a public dataset (iris); edit prepare_bigquery.py for yours
make data-bigquery
make train MODEL_DIR=./model_bq
```

The Makefile auto-loads `.env` (gitignored) and exports its variables
to all sub-shells, so once `GOOGLE_APPLICATION_CREDENTIALS` is in
`.env`, every BQ-using target picks it up without further work.
`.env.example` is the committed template.

### Round-trip your own table through BigQuery

To prove the full "your data lives in BQ → train on it" loop using the
sonar dataset as a stand-in:

```bash
make data-sonar          # generate data/sonar.csv locally
make bq-upload-sonar     # one-time: creates $PROJECT.sage_baker.sonar
make bq-data-sonar       # materializes the table back via prepare_bigquery.py
make train MODEL_DIR=./model_bq_sonar
cat model_bq_sonar/metadata.json | jq .data_lineage
```

The `data_lineage` block now records the BQ query, project, snapshot
timestamp, and dataset hash — same shape regardless of whether the
source was iris-public or your own table. Replace the sonar table with
whatever real dataset you want to train on; the rest of the pipeline
doesn't change.

To reproduce a past training run later: re-run the recorded
`metadata.data_lineage.query` with `FOR SYSTEM_TIME AS OF
'<snapshot_timestamp>'`, hash the result, verify it matches
`dataset_sha256`. Within BQ's time-travel window (7 days default,
longer with table snapshots) this is byte-for-byte exact.

### Three levels of "freeze," in order of rigor

1. **Hash + query in metadata** (what we do here). Cheap, sufficient
   for most reproducibility needs. You can detect drift; you can
   re-query within the warehouse's time-travel window.
2. **Save the materialized parquet alongside the model.** Add
   `tracking.log_artifacts("data/")` (or upload to S3 next to the
   bundle). Now even if the warehouse table is dropped, the training
   data survives. Costs storage.
3. **Use a data-versioning system.** [DVC](https://dvc.org/) and
   [LakeFS](https://lakefs.io/) treat datasets like git refs — the
   bundle records a dataset commit SHA, and the dataset itself is
   immutable until you delete it. Heaviest setup, strongest
   guarantees. Worth it for audit/regulated workflows.

### Other patterns worth knowing

- **BigQuery ML** (`CREATE MODEL ... AS SELECT ...`): trains *inside*
  BigQuery. Supported algorithms are limited (linear/logistic, k-means,
  ARIMA, boosted trees, AutoML, can import TF/PyTorch for inference).
  Useful if your team is SQL-first and the algorithms cover your case.
- **BQ Storage Read API** (`google-cloud-bigquery-storage`): a separate
  client library that swaps BigQuery's standard REST endpoint for a
  gRPC + Apache Arrow streaming transport. The regular client
  (`google-cloud-bigquery`) returns paginated JSON — fine for small
  queries, painful past a few thousand rows. With
  `google-cloud-bigquery-storage` installed, `to_dataframe()` and
  `load_table_from_dataframe()` automatically use the Storage API
  (no code changes); without it, BQ falls back to REST and warns.

  Rough scale of the speedup:

  | Query size       | REST            | Storage API     |
  | ---------------- | --------------- | --------------- |
  | 200 rows         | <1 s            | <1 s — no diff  |
  | 100K rows        | ~10 s           | ~1 s            |
  | 10M rows         | minutes (often times out) | ~10–30 s, parallelizable |

  Already in `requirements-bigquery.txt`. Needs the `bigquery.readSessions.create`
  IAM permission on the service account (usually included in
  `roles/bigquery.user` or `roles/bigquery.dataViewer`); without it, the
  client falls back to REST and warns. Don't confuse with the **Storage
  Write API** (different package, used for streaming inserts).

## Training/serving skew: two real bugs and what fixes them

The two most common production bugs in ML systems aren't bad models —
they're shape mismatches between what the model saw at training and
what shows up at inference. Both look like "the model crashed" or
"the model output garbage in production"; the underlying causes are
different.

### Bug 1: missing data at inference

Training data is built post-hoc from the warehouse — every join is
complete, every column populated. Then at inference time the user's
session is mid-flight, the upstream service is slow, or the feature
just hasn't arrived yet. The model crashes on `None`, or worse,
silently mispredicts.

A feature store helps but **not by magic**. What it actually buys you:

- One feature definition used in both training and serving — eliminates
  the "computed differently" version of the bug.
- TTLs on features — Feast returns NULL after a configurable window, so
  staleness is explicit and consistent.
- Point-in-time historical join — when you ask for training features at
  some entity's `event_timestamp`, you get exactly what was *known* at
  that timestamp, not a backfilled "complete" view.

What the feature store does NOT do: teach your model to handle NULL.
That's a *training-time* responsibility:

1. **Train with realistic missingness.** Build the entity dataframe
   from the actual entity-creation events; let `get_historical_features`
   return NULL for things that hadn't happened yet. New users have no
   `last_purchase_at` at training, exactly like at inference.
2. **Match imputation rules in both places.** If you `fillna(-1)` at
   training, do it at inference. Feast's `transformation`s on a
   FeatureView centralize this in one place.

### Bug 2: unseen categorical values at inference

The classic case: model trained on `browser ∈ {chrome, firefox,
safari}`, deployed; six months later "Edge 127" hits 5% of traffic and
everything explodes. **A feature store does not help here** — this is
a model/encoding choice.

`demo_categorical.py` runs three responses to this bug end-to-end:

```
trained on browsers: ['chrome', 'firefox', 'safari']
5 inference rows now have unseen value 'edge'

--- Path 1: sklearn OneHotEncoder default ---
CRASH ValueError: Found unknown categories ['edge'] in column 0 during transform

--- Path 2: sklearn OneHotEncoder(handle_unknown='ignore') ---
OK   predictions: [1, 0, 1, 0, 0]

--- Path 3: LightGBM + OrdinalEncoder(unknown_value=-1) ---
OK   predictions: [1, 0, 1, 0, 0]
```

Defaults bite. The fixes, in order of how good the result is:

| Approach | What happens with unseen values |
| -------- | ------------------------------- |
| `OneHotEncoder()` (default `handle_unknown="error"`) | crashes — the typical production bug |
| `OneHotEncoder(handle_unknown="ignore")` | becomes all-zeros row; doesn't crash; **loses whatever signal that value carried** |
| `HashingEncoder` (feature hashing) | hash → fixed bucket; new values land somewhere; some collisions |
| **LightGBM with `categorical_feature=[...]`** | unseen values get their own group at split time; preserves signal |
| Train an `<UNK>` token on rare values | replace categories appearing fewer than K times with `"other"` during training; model learns a real "other" branch |

For tabular data with categorical features, the boring-good answer is
**switch to LightGBM with native categorical handling**. It's faster,
usually more accurate than RandomForest, and immune to this whole
class of bug by design.

### What also helps: validation at the inference boundary

Both bugs above hide inside Python type errors deep in the model
pipeline. Catch them earlier with a schema check on the input *before*
it reaches the model:

- **Pandera** — DataFrame schema validation; "this column must be one
  of {'chrome', 'firefox', 'safari', 'other'} and not null."
- **Pydantic** — request-shape validation for HTTP endpoints; standard
  in FastAPI inference services.
- **Drift monitoring** — log feature distributions at inference, alert
  on KL divergence or population-stability index (PSI) shifts. Tools:
  Evidently, TFDV, or roll your own with `scipy.stats.entropy`.

The boundary check turns "buried KeyError in the model layer at 3am"
into "clean 400 at the API layer with the offending value named."

## Creating a new model: end-to-end workflow

```mermaid
flowchart LR
  A[1. prepare_X.py<br/>fetch → CSV/parquet<br/>+ lineage.json]
  B[2. train_X.py<br/>read data → fit →<br/>save bundle]
  C[3. local_serve.py<br/>verify model_fn loads<br/>and predicts]
  D[4. MLflow<br/>track runs, register<br/>promising versions]
  E[5. Deploy<br/>SageMaker endpoint /<br/>downstream caller]
  A --> B --> C --> D --> E
```

Concrete steps:

1. **Get the data into the project.** Write a `prepare_<name>.py` that
   fetches it, shapes it (CSV or parquet with a `target` column), and
   drops `data/lineage.json`. Use `prepare_sonar.py` as a template.
2. **Pick or write a trainer.** The existing ones cover the common
   cases:
   - `src/train.py` — sklearn (RandomForest by default)
   - `src/train_lightgbm.py` — LightGBM (best for tabular w/ categoricals)
   - `src/train_torch.py` — custom torch `nn.Module`
   - `src/train_feast.py` — features sourced from Feast

   To add a new framework, copy whichever existing trainer is closest,
   swap the model class and weights serializer, **keep the bundle
   calls** (`bundle.save_config`, `bundle.save_metadata`,
   `bundle.load_lineage`) and the `model_fn(model_dir)` shape
   identical. The bundle envelope is the contract.
3. **Train and verify locally.**
   ```bash
   .venv/bin/python src/train_<name>.py --train ./data --model-dir ./model_<name>
   .venv/bin/python local_serve.py --model-dir ./model_<name>
   ```
   `local_serve.py` exercises `model_fn` end-to-end, the same contract
   SageMaker / MLflow / your custom inference container will use.
4. **Turn on tracking once it's working.**
   ```bash
   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   ```
   Retrain — every run now logs params, metrics, the bundle as
   artifacts, and registers a model version. Iterate from there.
5. **Promote and deploy.** Promote registered versions through
   MLflow's stage transitions (Staging → Production). Deploy to
   SageMaker via the `local_train_dlc.py` / `local_train_feast_dlc.py`
   drivers (or your work's equivalent), pulling the model_uri from
   MLflow's registry.

### What this scaffolding does NOT replace

The hard part of a new model — the *modeling* work — is universal and
sits outside this repo:

- **Define the metric.** Accuracy / F1 / AUC / latency / calibration —
  pick before you start. Hardest part for many teams.
- **Establish a dumb baseline.** "Predict majority class" for
  classification, "predict the mean" for regression. Anything below
  that is a bug, not a model.
- **Pick a model class that fits the data shape.** Tabular →
  LightGBM/RandomForest. Sequences/text/images → torch. Tiny dataset
  → linear / k-NN. Pretrained foundation model available → fine-tune
  via HuggingFace.
- **Validation strategy.** Simple holdout (what we use here; fine for
  IID data), k-fold cross-validation (more rigorous), **time-based
  split** (mandatory if your data has temporal structure — never
  random-split time series).
- **Iterate features before hyperparameters.** New informative
  features beat hyperparameter tuning ~10x in real projects.

The repo's job is to make the path from "trained model in a Jupyter
cell" to "deployed model with provenance" mechanical. The model
design itself is on you.

## Jupyter for exploration

Once `.venv/` is set up, two extra steps make the whole project
usable from notebooks:

```bash
.venv/bin/pip install -r requirements-jupyter.txt
.venv/bin/python -m ipykernel install --user --name sage-baker --display-name "Python (sage-baker)"
.venv/bin/jupyter lab
```

In a notebook, import from `src/` after adding it to `sys.path`:

```python
import sys; sys.path.insert(0, "src")
import bundle, train

# load any existing bundle by directory
model = train.model_fn("./model_sklearn")
preds = model.predict(X.head())

# or call the training pipeline directly with overrides
import importlib; importlib.reload(train)   # picks up edits
import sys; sys.argv = ["train.py", "--train", "./data", "--model-dir", "./model_nb"]
train.main()
```

There's a starter notebook at `notebooks/bigquery_exploration.ipynb`
that demonstrates: setup, BQ via cell magic and direct client, training
on the result, and inference — all in one place. Open it from the Lab
file browser (kernel: **Python (sage-baker)**).

A few patterns that pay off:

- **`%load_ext autoreload; %autoreload 2`** — edit `src/train.py` and
  re-run notebook cells without restarting the kernel.
- **MLflow searchable from the notebook**:
  `mlflow.search_runs(experiment_ids=["0"])` returns a DataFrame of
  all runs with their params/metrics. Useful for "which run had the
  best validation accuracy?"
- **Treat notebooks as the exploration layer, scripts as the
  production layer.** Code that's worth keeping graduates from
  notebook cell → `src/` module → trainer entry point. Anything still
  in a notebook is research, not infrastructure.

The trainers themselves are normal Python modules — there's nothing
notebook-specific about them, and nothing in the repo expects to be
run from one place. The same `model_fn(model_dir)` works from a
notebook, from `local_serve.py`, from MLflow's PyFunc, and from
SageMaker.

## Hyperparameters

`local_train.py` passes `n-estimators` and `max-depth` to the estimator;
SageMaker writes these to `/opt/ml/input/config/hyperparameters.json` inside
the container. `train.py` reads that file and applies them as `argparse`
defaults. SageMaker stringifies all hyperparameters, so cast to the type you
want when reading.

## Gotchas

A few things that bit us; worth knowing if you adapt this to a different
framework or environment.

- **SageMaker SDK v3 removed `sagemaker.local`.** Pin to `sagemaker<3` until
  Local Mode lands in v3 (or it stays gone — TBD).
- **Snap-installed Docker is confined.** It can't bind-mount paths under
  `/tmp`, which is where the SDK normally drops its `docker-compose.yaml` and
  per-job scratch dirs. `local_train.py` works around this by setting
  `TMPDIR` and `local.container_root` to `.sm-scratch/` under the project.
  If your Docker is from `docker.io` / `docker-ce` (apt) you can drop that.
- **Real AWS credentials are NOT required for BYOC**, but boto3 still needs
  *something* in its credential chain plus a region — `local_train.py` sets
  dummy values via env vars before constructing `LocalSession`.
- **`role=` is ignored in Local Mode** but the SDK still validates it as a
  string. Any ARN-shaped string works.
- **`image_uri=` bypasses the framework's ECR lookup.** As long as the
  reference has no registry prefix and the image is present locally, Docker
  will use it without trying to pull.

## Customizing for your model

To swap in your own training:

1. Replace the body of `train.py` with your training code, keeping:
   - reads from `args.train` (a directory of input files)
   - writes a model file to `args.model_dir`
2. Update `Dockerfile` deps if you need PyTorch / TF / etc.
3. Update `local_train.py` hyperparameters and the `inputs={...}` dict passed
   to `fit()` (one key per channel).

For larger projects you probably want `entry_point` + `source_dir` so you can
edit the script without rebuilding the image. That requires installing the
`sagemaker-training` toolkit in the image (the package that interprets the
`sagemaker_program` / `sagemaker_submit_directory` hyperparameters and runs
your script). It's heavier and its install can be finicky on slim Python
images, which is why this scaffold bakes the script into the image instead.

## Serving (not implemented)

`local_serve.py` is left over from an earlier attempt that used the AWS
scikit-learn DLC. It will not work against this BYOC image because the image
has no `serve` command.

To add serving, the image needs a `serve` command on `PATH` that starts an
HTTP server on port 8080 with two routes:

- `GET /ping` → `200` when ready
- `POST /invocations` → consumes the request body, returns predictions

Flask + gunicorn is the usual minimal setup. Then `local_serve.py` can use
`sagemaker.model.Model(image_uri="sage-baker-sklearn:latest", ...)` and
`.deploy(instance_type="local")` to spin up a local endpoint.

## Cleaning up

`.sm-scratch/` accumulates artifacts and per-job dirs. Safe to delete between
runs:

```bash
rm -rf .sm-scratch
```

Containers and the `sagemaker-local` Docker network are torn down by the SDK
at the end of each `fit()`.
