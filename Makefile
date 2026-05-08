.DEFAULT_GOAL := help

# Load .env if present (gitignored; .env.example is the template). The
# `-include` makes it optional, and `export` pushes all loaded vars to
# every sub-shell so trainers / prepare scripts pick them up.
-include .env
export

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Override on the command line, e.g. `make train MODEL_DIR=./other`
MODEL_DIR  ?= ./model_sklearn
DATA_DIR   ?= ./data
MLFLOW_URI ?= http://127.0.0.1:5000

# ---------- help (default) ----------------------------------------------

help: ## Show available targets
	@echo "Usage: make <target>    (override vars: MODEL_DIR=, DATA_DIR=, MLFLOW_URI=)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------- setup -------------------------------------------------------

install: ## Create venv and install base requirements
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-torch: ## Install torch + safetensors extras
	$(PIP) install -r requirements-torch.txt

install-lightgbm: ## Install LightGBM extras
	$(PIP) install -r requirements-lightgbm.txt

install-skops: ## Install skops (safer-pickle for sklearn)
	$(PIP) install -r requirements-skops.txt

install-feast: ## Install Feast feature-store extras
	$(PIP) install -r requirements-feast.txt

install-bigquery: ## Install BigQuery extras
	$(PIP) install -r requirements-bigquery.txt

install-jupyter: ## Install Jupyter and register the project kernel
	$(PIP) install -r requirements-jupyter.txt
	$(PY) -m ipykernel install --user --name sage-baker --display-name "Python (sage-baker)"

install-recommender: ## Install recommender extras (implicit, scipy, pyarrow)
	$(PIP) install -r requirements-recommender.txt

install-all: install install-torch install-lightgbm install-skops install-feast install-bigquery install-recommender install-jupyter ## Install everything

# ---------- data prep ---------------------------------------------------

data-iris: ## Prepare iris dataset (sklearn-bundled, 3-class)
	$(PY) prepare_data.py

data-sonar: ## Prepare sonar dataset + Feast parquets (binary)
	$(PY) prepare_sonar.py

data-als: ## Prepare synthetic ALS dataset (generic user × item interactions)
	$(PY) prepare.py --plugin als

data-bigquery: ## Materialize a BigQuery query (default: public iris dataset)
	$(PY) prepare_bigquery.py

bq-upload-sonar: ## Upload data/sonar.csv to BQ as $PROJECT.sage_baker.sonar
	$(PY) upload_sonar_to_bq.py

bq-data-sonar: ## Materialize the sonar table back from BQ (after bq-upload-sonar)
	$(PY) prepare_bigquery.py \
		--query "SELECT * FROM \`$(GOOGLE_CLOUD_PROJECT).sage_baker.sonar\`"

# ---------- training (host-side) ---------------------------------------

train: ## Host-side training (default plugin: RandomForest)
	$(PY) src/train.py --train $(DATA_DIR) --model-dir $(MODEL_DIR)

train-als: ## Host-side ALS training (run data-als + install-recommender first)
	$(PY) src/train_recommender.py --train $(DATA_DIR) --model-dir ./model_als --plugin als

train-torch: ## Host-side torch (MLP) training
	$(PY) src/train_torch.py --train $(DATA_DIR) --model-dir ./model_torch

train-lightgbm: ## Host-side LightGBM training
	$(PY) src/train_lightgbm.py --train $(DATA_DIR) --model-dir ./model_lgb

train-feast: ## Host-side sklearn + Feast (point-in-time historical join)
	$(PY) src/train_feast.py --feature-repo ./feature_repo --model-dir ./model_feast

# ---------- training (SageMaker Local Mode) ----------------------------

image: ## Build the BYOC training image
	docker build -t sage-baker-sklearn:latest .

train-byoc: image ## SageMaker Local Mode + local BYOC image (no AWS account)
	$(PY) local_train.py

train-dlc: ## SageMaker Local Mode + AWS sklearn DLC (needs AWS creds + ECR perms)
	$(PY) local_train_dlc.py

train-feast-dlc: ## SageMaker Local Mode DLC + Feast pre-fetch (needs AWS creds)
	$(PY) local_train_feast_dlc.py

# ---------- inference ---------------------------------------------------

serve: ## Run local_serve.py against $MODEL_DIR
	$(PY) local_serve.py --model-dir $(MODEL_DIR)

mlflow-serve: ## Load model via MLflow Model Registry
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PY) mlflow_serve.py

demo-categorical: ## Demo: 'new enum value at inference' bug + 3 fixes
	$(PY) demo_categorical.py

# ---------- infrastructure (long-running processes) --------------------

mlflow-server: ## Start MLflow tracking server (foreground; Ctrl-C to stop)
	$(VENV)/bin/mlflow server --host 127.0.0.1 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts

jupyter: ## Start Jupyter Lab (foreground; Ctrl-C to stop)
	$(VENV)/bin/jupyter lab

feast-apply: ## Register Feast entity/view + materialize features online
	cd feature_repo && ../$(VENV)/bin/feast apply
	cd feature_repo && ../$(VENV)/bin/feast materialize-incremental $$(date -u +%Y-%m-%dT%H:%M:%S)

# ---------- cleanup -----------------------------------------------------

clean: ## Remove scratch dirs (keeps venv, MLflow data, Feast registry)
	rm -rf .sm-scratch model_*/ materialized/

.PHONY: help install install-torch install-lightgbm install-skops install-feast install-bigquery install-recommender install-jupyter install-all
.PHONY: data-iris data-sonar data-als data-bigquery bq-upload-sonar bq-data-sonar
.PHONY: train train-als train-torch train-lightgbm train-feast
.PHONY: image train-byoc train-dlc train-feast-dlc
.PHONY: serve mlflow-serve demo-categorical
.PHONY: mlflow-server jupyter feast-apply clean

# Private plugin targets (gitignored; company-specific).
-include Makefile.private
