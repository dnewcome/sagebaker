# sagebaker — embedding serving image
#
# Dependency family: PyTorch + sentence-transformers + FAISS + spaCy.
# Suitable for: nearest-neighbour product recommenders, semantic search,
# and any model requiring dense vector embeddings at inference time.
#
# Runtime configuration (ECS env vars):
#   PLUGIN_NAME         name of the plugin to serve (required)
#   MODEL_DIR           path or s3://… URI of the bundle (required; serve.py
#                       resolves s3:// URIs to a local cache transparently)
#   PLUGIN_DIR          path to plugin .py files if not baked into image (optional)
#   PORT                serving port (default: 8080)
#   TRANSFORMERS_OFFLINE set to 1 to disable HuggingFace hub lookups at runtime
#
# Build args:
#   SAGEBAKER_REF      git ref (branch, tag, or SHA) to install (default: main)
#   FAISS_PKG          faiss-cpu or faiss-gpu (default: faiss-cpu)
#   SPACY_MODEL        spaCy model to download (default: en_core_web_sm)

ARG PYTORCH_DLC=763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.3.0-cpu-py311
FROM ${PYTORCH_DLC}

ARG SAGEBAKER_REF=main
ARG FAISS_PKG=faiss-cpu
ARG SPACY_MODEL=en_core_web_sm

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "git+https://github.com/dnewcome/sagebaker.git@${SAGEBAKER_REF}" \
    sentence-transformers \
    "${FAISS_PKG}" \
    spacy \
    && python -m spacy download ${SPACY_MODEL}

ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8080

CMD ["sagebaker-serve"]
