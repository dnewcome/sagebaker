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
MODEL_DIR  ?= ./models/sklearn
DATA_DIR   ?= ./data
MLFLOW_URI ?= http://127.0.0.1:5000

# ---------- help (default) ----------------------------------------------

help: ## Show available targets
	@echo "Usage: make <target>    (override vars: MODEL_DIR=, DATA_DIR=, MLFLOW_URI=)"
	@echo ""
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------- setup -------------------------------------------------------

# Dependency groups live in pyproject.toml ([dependency-groups], PEP 735).
# Requires pip >= 25.1.

install: ## Create venv and install base requirements
	python3 -m venv $(VENV)
	$(PIP) install --upgrade 'pip>=25.1'
	$(PIP) install --group base

install-torch: ## Install torch + safetensors extras
	$(PIP) install --group torch

install-lightgbm: ## Install LightGBM extras
	$(PIP) install --group lightgbm

install-skops: ## Install skops (safer-pickle for sklearn)
	$(PIP) install --group skops

install-feast: ## Install Feast feature-store extras
	$(PIP) install --group feast

install-bigquery: ## Install BigQuery extras
	$(PIP) install --group bigquery

install-jupyter: ## Install Jupyter and register the project kernel
	$(PIP) install --group jupyter
	$(PY) -m ipykernel install --user --name sagebaker --display-name "Python (sagebaker)"

install-recommender: ## Install recommender extras (implicit, scipy, pyarrow)
	$(PIP) install --group recommender

install-retrieval: ## Install semantic-search extras (sentence-transformers, faiss-cpu)
	$(PIP) install --group retrieval

install-agent: ## Install autoresearch-style agent extras (anthropic SDK)
	$(PIP) install --group agent

install-serve: ## Install HTTP serving extras (flask)
	$(PIP) install --group serve

install-dev: ## Install dev tooling (pytest)
	$(PIP) install --group dev

install-all: ## Install everything (every dependency group)
	$(PIP) install --group all

# ---------- data prep ---------------------------------------------------

data-iris: ## Prepare iris dataset (sklearn-bundled, 3-class)
	$(PY) prep/prepare_data.py

data-sonar: ## Prepare sonar dataset + Feast parquets (binary)
	$(PY) prep/prepare_sonar.py

data-als: ## Prepare synthetic ALS dataset (generic user × item interactions)
	$(PY) prep/prepare.py --plugin als

data-housing: ## Prepare California housing (sklearn-bundled, regression)
	$(PY) prep/prepare.py --plugin housing

data-movielens: ## Fetch MovieLens-100K (~1.7 MB) for the ALS recommender path
	$(PY) prep/prepare_movielens.py

data-simulate: ## Run a simulated scenario: SCENARIO=<name> OUTPUT=<dir>
	$(PY) prep/prepare_simulate.py --scenario $(SCENARIO) --output $(OUTPUT)

data-fuzzy: ## Generate fuzzy_clickstream scenario into ./data/fuzzy/
	$(PY) prep/prepare_simulate.py --scenario fuzzy_clickstream --output ./data/fuzzy/

data-products: ## Generate product_catalog scenario into ./data/products/
	$(PY) prep/prepare_simulate.py --scenario product_catalog --output ./data/products/

data-linkage: ## Build pair-level dataset from ./data/fuzzy/ for record-linkage training
	$(PY) prep/prepare_linkage.py --input ./data/fuzzy --output ./data/linkage --n-pairs 20000

data-matcher-pairs: ## Build pair-level dataset from ./data/products/ for product-matching training
	$(PY) prep/prepare_matcher_pairs.py --input ./data/products --output ./data/matcher --n-pairs 5000

data-bigquery: ## Materialize a BigQuery query (default: public iris dataset)
	$(PY) prep/prepare_bigquery.py

bq-upload-sonar: ## Upload data/sonar.csv to BQ as $PROJECT.sage_baker.sonar
	$(PY) tools/upload_sonar_to_bq.py

bq-data-sonar: ## Materialize the sonar table back from BQ (after bq-upload-sonar)
	$(PY) prep/prepare_bigquery.py \
		--query "SELECT * FROM \`$(GOOGLE_CLOUD_PROJECT).sage_baker.sonar\`"

# ---------- training (host-side) ---------------------------------------

train: ## Host-side training (default plugin: RandomForest)
	$(PY) src/train.py --train $(DATA_DIR) --model-dir $(MODEL_DIR)

train-als: ## Host-side ALS training (run data-als + install-recommender first)
	$(PY) src/train_recommender.py --train $(DATA_DIR) --model-dir ./models/als --plugin als

train-housing: ## Host-side regression on California housing (R² metric)
	$(PY) src/train.py --train $(DATA_DIR) --model-dir ./models/housing --plugin housing

train-clickstream: ## Host-side conversion classification on a fuzzy_clickstream dataset
	$(PY) src/train.py --train ./data/fuzzy --model-dir ./models/clickstream --plugin clickstream

train-clickstream-linkage: ## Train record-linkage model on pair-level data (run data-linkage first)
	$(PY) src/train.py --train ./data/linkage --model-dir ./models/clickstream_linkage --plugin clickstream_linkage

train-search: ## Build a FAISS semantic-search index on the product catalog (run data-products first)
	$(PY) src/train_retrieval.py --train ./data/products --model-dir ./models/search --plugin product_search

train-product-matcher: ## Train pair-classifier for product matching (run data-matcher-pairs first)
	$(PY) src/train.py --train ./data/matcher --model-dir ./models/product_matcher --plugin product_matcher

train-torch: ## Host-side torch (MLP) training
	$(PY) src/train_torch.py --train $(DATA_DIR) --model-dir ./models/torch

train-lightgbm: ## Host-side LightGBM training
	$(PY) src/train_lightgbm.py --train $(DATA_DIR) --model-dir ./models/lgb

train-feast: ## Host-side sklearn + Feast (point-in-time historical join)
	$(PY) src/train_feast.py --feature-store ./feature_store --model-dir ./models/feast

# ---------- training (SageMaker Local Mode) ----------------------------

image: ## Build the BYOC training image
	docker build -t sagebaker-sklearn:latest .

train-byoc: image ## SageMaker Local Mode + local BYOC image (no AWS account)
	$(PY) drivers/local_train.py

train-dlc: ## SageMaker Local Mode + AWS sklearn DLC (needs AWS creds + ECR perms)
	$(PY) drivers/local_train_dlc.py

train-feast-dlc: ## SageMaker Local Mode DLC + Feast pre-fetch (needs AWS creds)
	$(PY) drivers/local_train_feast_dlc.py

# ---------- inference ---------------------------------------------------

serve: ## Run local_serve.py against $MODEL_DIR
	$(PY) local_serve.py --model-dir $(MODEL_DIR)

serve-http: ## Single-model HTTP server: PLUGIN_NAME=fillrate MODEL_DIR=models/fillrate make serve-http
	@$(eval PLUGIN_NAME ?= fillrate)
	@$(eval MODEL_DIR ?= models/fillrate)
	@$(eval PORT ?= 8080)
	@echo "serving plugin=$(PLUGIN_NAME) model=$(MODEL_DIR) on port $(PORT)"
	@echo "POST to http://localhost:$(PORT)/predict"
	PLUGIN_NAME=$(PLUGIN_NAME) MODEL_DIR=$(MODEL_DIR) PORT=$(PORT) \
	  gunicorn -w 1 -t 1 --timeout 120 --bind 0.0.0.0:$(PORT) 'serve:app'

mlflow-serve: ## Load model via MLflow Model Registry
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PY) mlflow_serve.py

mlflow-serve-http: ## HTTP scoring server: curl POST to localhost:5001/invocations. Override NAME/VERSION/PORT.
	@$(eval NAME ?= sagebaker-sklearn)
	@$(eval VERSION ?= latest)
	@$(eval PORT ?= 5001)
	@echo "serving models:/$(NAME)/$(VERSION) on port $(PORT) — POST to http://127.0.0.1:$(PORT)/invocations"
	@echo 'payload format: {"dataframe_records": [{<feature_name>: <value>, ...}]}'
	PYTHONPATH=$(CURDIR)/src \
	PATH="$(CURDIR)/$(VENV)/bin:$(PATH)" \
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) \
	$(VENV)/bin/mlflow models serve \
	  -m "models:/$(NAME)/$(VERSION)" \
	  -p $(PORT) --host 127.0.0.1 \
	  --env-manager local

demo-categorical: ## Demo: 'new enum value at inference' bug + 3 fixes
	$(PY) tools/demo_categorical.py

agent: ## autoresearch-style agent loop — edits src/plugins/default.py iteratively
	$(PY) agent.py

agent-clickstream: ## autoresearch loop on src/plugins/clickstream.py against ./data/fuzzy/
	$(PY) agent.py \
	    --plugin src/plugins/clickstream.py \
	    --program agent_clickstream.md \
	    --data-dir ./data/fuzzy

test: ## Run smoke test suite (bundle round-trip, plugin contract, etc.)
	$(VENV)/bin/pytest tests/ -q

# ---------- infrastructure (long-running processes) --------------------

mlflow-server: ## Start MLflow tracking server (foreground; Ctrl-C to stop)
	$(VENV)/bin/mlflow server --host 127.0.0.1 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts

jupyter: ## Start Jupyter Lab (foreground; Ctrl-C to stop)
	$(VENV)/bin/jupyter lab

feast-apply: ## Register Feast entity/view + materialize features online
	cd feature_store && ../$(VENV)/bin/feast apply
	cd feature_store && ../$(VENV)/bin/feast materialize-incremental $$(date -u +%Y-%m-%dT%H:%M:%S)

# ---------- docker base images -----------------------------------------

docker-build-tabular: ## Build the tabular serving base image (sklearn DLC + lightgbm)
	docker build -t sagebaker-tabular:latest docker/tabular/

docker-build-embedding: ## Build the embedding serving base image (pytorch DLC + sentence-transformers + faiss)
	docker build -t sagebaker-embedding:latest docker/embedding/

# MODEL_DIR has a global default, so $(eval MODEL_DIR ?= …) won't apply
# our per-plugin default. `origin` lets us honour command-line overrides
# while computing models/$(PLUGIN) when nothing was passed explicitly.
docker-train: ## Train a plugin inside the tabular image. PLUGIN=sonar DATA_DIR=data MODEL_DIR=models/sonar
	@$(eval PLUGIN ?= sonar)
	@$(eval IMAGE ?= sagebaker-tabular:latest)
	@$(eval DATA_DIR := $(if $(filter command line,$(origin DATA_DIR)),$(DATA_DIR),$(CURDIR)/data))
	@$(eval MODEL_DIR := $(if $(filter command line,$(origin MODEL_DIR)),$(MODEL_DIR),$(CURDIR)/models/$(PLUGIN)))
	mkdir -p $(MODEL_DIR)
	docker run --rm \
	  -v $(DATA_DIR):/data \
	  -v $(MODEL_DIR):/model \
	  $(IMAGE) \
	  python -m train --train /data --model-dir /model --plugin $(PLUGIN)

docker-serve: ## Serve a local bundle via the tabular image. PLUGIN=sonar MODEL_DIR=models/sonar PORT=8080
	@$(eval PLUGIN ?= sonar)
	@$(eval IMAGE ?= sagebaker-tabular:latest)
	@$(eval MODEL_DIR := $(if $(filter command line,$(origin MODEL_DIR)),$(MODEL_DIR),$(CURDIR)/models/$(PLUGIN)))
	@$(eval PORT ?= 8080)
	docker run --rm -p $(PORT):8080 \
	  -v $(MODEL_DIR):/model \
	  -e PLUGIN_NAME=$(PLUGIN) \
	  -e MODEL_DIR=/model \
	  $(IMAGE)

# ---------- cleanup -----------------------------------------------------

clean: ## Remove scratch dirs (keeps venv, MLflow data, Feast registry)
	rm -rf .sm-scratch models/ materialized/

.PHONY: help install install-torch install-lightgbm install-skops install-feast install-bigquery install-recommender install-agent install-serve install-jupyter install-all
.PHONY: data-iris data-sonar data-als data-housing data-movielens data-bigquery bq-upload-sonar bq-data-sonar
.PHONY: train train-als train-housing train-torch train-lightgbm train-feast
.PHONY: image train-byoc train-dlc train-feast-dlc
.PHONY: serve serve-http mlflow-serve demo-categorical agent
.PHONY: docker-build-tabular docker-build-embedding docker-train docker-serve
.PHONY: mlflow-server jupyter feast-apply clean

# Private plugin targets (gitignored; company-specific).
-include Makefile.private
