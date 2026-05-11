# sagebaker — tabular serving image
#
# Dependency family: scikit-learn + LightGBM + imbalanced-learn.
# Suitable for: classification and regression models trained with LightGBM,
# sklearn pipelines, or any combination thereof.
#
# Runtime configuration (ECS env vars):
#   PLUGIN_NAME       name of the plugin to serve (required)
#   MODEL_DIR         path or s3://… URI of the bundle (required; serve.py
#                     resolves s3:// URIs to a local cache transparently)
#   PLUGIN_DIR        path to plugin .py files if not baked into image (optional)
#   PORT              serving port (default: 8080)
#
# Build args:
#   SAGEBAKER_REF      git ref (branch, tag, or SHA) to install (default: main)
#   LIGHTGBM_VERSION
#   IMBALANCED_VERSION

ARG SKLEARN_DLC=763104351884.dkr.ecr.us-west-2.amazonaws.com/sklearn:1.2-1-cpu-py3
FROM ${SKLEARN_DLC}

ARG SAGEBAKER_REF=main
ARG LIGHTGBM_VERSION=""
ARG IMBALANCED_VERSION=""

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "git+https://github.com/dnewcome/sagebaker.git@${SAGEBAKER_REF}" \
    "lightgbm${LIGHTGBM_VERSION:+==$LIGHTGBM_VERSION}" \
    "imbalanced-learn${IMBALANCED_VERSION:+==$IMBALANCED_VERSION}"

EXPOSE 8080

# Default entrypoint: sagebaker's HTTP serving harness, exposed as a
# console script by the sagebaker package.  Override PLUGIN_DIR at
# runtime to inject plugin code without rebuilding.
CMD ["sagebaker-serve"]
