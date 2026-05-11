# sagebaker serving images

Two base images covering the common model dependency families.
Plugin code and model weights are injected at runtime — nothing
model-specific is baked into the bases.

## Images

| Image | Base | Added deps | Use for |
|-------|------|-----------|---------|
| `tabular` | AWS sklearn DLC | LightGBM, imbalanced-learn | Classification/regression with LightGBM or sklearn |
| `embedding` | AWS PyTorch DLC | sentence-transformers, FAISS, spaCy | Recommenders, semantic search, dense embeddings |

Both install `sagebaker` directly from GitHub (`pip install
git+https://github.com/dnewcome/sagebaker.git@${SAGEBAKER_REF}`) — no
PyPI release needed. The `sagebaker-serve` console script is wired up
as the default `CMD`.

## Runtime configuration

Both images are configured entirely via environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `PLUGIN_NAME` | yes | Plugin to serve (e.g. `sonar`) |
| `MODEL_DIR` | yes | Local path **or** `s3://…` URI of the bundle (serve.py resolves S3 to a local cache transparently) |
| `PLUGIN_DIR` | no | Directory of plugin `.py` files to load on top of built-ins |
| `PORT` | no | Serving port (default: 8080) |

## Building

```bash
make docker-build-tabular         # or:
docker build -t sagebaker-tabular:latest docker/tabular/

make docker-build-embedding
```

Pin a specific sagebaker ref:

```bash
docker build \
  --build-arg SAGEBAKER_REF=v0.2.0 \
  -t sagebaker-tabular:0.2.0 docker/tabular/
```

## Sonar end-to-end

Sonar is built into `sagebaker` (`src/plugins/sonar.py`), so the
tabular base image already knows how to train and serve it — no
downstream Dockerfile required.

```bash
# 1. Build the base image once.
make docker-build-tabular

# 2. Train inside the container; bundle lands in ./models/sonar/ on the host.
make data-sonar
make docker-train PLUGIN=sonar

# 3. Serve the just-trained bundle locally.
make docker-serve PLUGIN=sonar
curl -s -X POST http://localhost:8080/predict \
     -H 'Content-Type: application/json' \
     -d '[{"f0":0.02,"f1":0.04, ...}]'

# 4. Push the same image to ECR for inference.
docker tag sagebaker-tabular:latest <acct>.dkr.ecr.us-west-2.amazonaws.com/sagebaker-tabular:sonar
docker push <acct>.dkr.ecr.us-west-2.amazonaws.com/sagebaker-tabular:sonar
```

In production the bundle isn't baked into the image — point `MODEL_DIR`
at an `s3://` URI and the container will download (and cache) the
bundle at startup:

```json
{ "name": "PLUGIN_NAME", "value": "sonar" },
{ "name": "MODEL_DIR",   "value": "s3://your-bucket/models/sonar/v42/" }
```

Rolling out a new model = new S3 path + ECS force-new-deployment. No
image rebuild.

## Adding an external plugin

For plugins that aren't shipped with `sagebaker`, layer a thin image:

```dockerfile
FROM sagebaker-tabular:latest
COPY plugins/ /opt/ml/plugins/
ENV PLUGIN_DIR=/opt/ml/plugins
```

Plugin `.py` files dropped into `PLUGIN_DIR` are auto-discovered at
import time (see `src/plugins/__init__.py`).
