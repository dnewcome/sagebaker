"""Single-model HTTP serving harness.

Each model runs as its own process/container, configured entirely via
environment variables.  No model registry, no hardcoded routes — just
one plugin + one bundle per process.

Environment variables
---------------------
PLUGIN_NAME   registered plugin name, e.g. "fillrate" or "clh"
MODEL_DIR     path to a sage-baker bundle directory, OR an S3 URI.
              Local:  /opt/ml/model/fillrate/
              S3:     s3://bucket/models/fillrate/run-20260510/
              Legacy: s3://bucket/models/cc_product_recommender/v1/model_run-123
PORT          HTTP port (default 8080)

S3 bundles are downloaded to a local temp directory at startup.  Both
full sage-baker bundles (config.json + weights file) and legacy single-pkl
production artifacts are handled automatically.

Routes
------
GET  /health   liveness probe — returns "ok"
POST /predict  JSON array of feature dicts → plugin-defined response dict

Production deployment
---------------------
Run under gunicorn rather than the Flask dev server:
    gunicorn -w 1 -t 1 --timeout 120 'serve:app'
"""
import json
import os
import sys

from flask import Flask, jsonify, request

# src/ is on the path via editable install; fall back to explicit insert.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import bundle  # noqa: E402
from plugins import get_plugin, get_recommender_plugin  # noqa: E402

_plugin = None
_model = None
_config: dict = {}


def _load() -> None:
    global _plugin, _model, _config
    plugin_name = os.environ["PLUGIN_NAME"]
    model_dir = os.environ["MODEL_DIR"]

    try:
        _plugin = get_plugin(plugin_name)
    except ValueError:
        _plugin = get_recommender_plugin(plugin_name)

    # Resolve S3 URI → local temp dir (no-op for local paths).
    local_dir = bundle.resolve_model_dir(model_dir)

    _model = _plugin.load_bundle(local_dir)

    config_path = os.path.join(local_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            _config = json.load(f)


app = Flask("sage-baker-serve")


@app.route("/health", methods=["GET"])
def health():
    return "ok"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)
        rows = body if isinstance(body, list) else [body]
        return jsonify(_plugin.serve(_model, rows, _config))
    except NotImplementedError as e:
        return jsonify({"error": str(e)}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Load at module level so gunicorn workers pick it up without needing
# __main__.  The console-script entry point and `python -m serve` paths
# also reuse the already-loaded model via main().
_load()


def main() -> None:
    """Console-script entry point (sagebaker-serve)."""
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
