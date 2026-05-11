"""Single-model HTTP serving harness.

Each model runs as its own process/container, configured entirely via
environment variables.  No model registry, no hardcoded routes — just
one plugin + one bundle per process.

Environment variables
---------------------
PLUGIN_NAME   registered plugin name, e.g. "fillrate" or "clh"
MODEL_DIR     path to a sage-baker bundle directory
              (must contain config.json + weights file)
PORT          HTTP port (default 8080)

Routes
------
GET  /health   liveness probe — returns "ok"
POST /predict  JSON array of feature dicts → plugin-defined response dict

Example
-------
    PLUGIN_NAME=fillrate MODEL_DIR=models/fillrate python serve.py
    curl -s -X POST localhost:8080/predict \\
         -H 'Content-Type: application/json' \\
         -d '[{"event_timestamp": "2024-06-01T14:00:00Z", ...}]'
"""
import json
import os
import sys

from flask import Flask, jsonify, request

# src/ is on the path via editable install; fall back to explicit insert.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

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
    _model = _plugin.load_bundle(model_dir)
    config_path = os.path.join(model_dir, "config.json")
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


if __name__ == "__main__":
    _load()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
