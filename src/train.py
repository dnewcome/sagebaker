"""Training script SageMaker runs inside the container.

SageMaker conventions:
  /opt/ml/input/data/<channel>/  -- training data
  /opt/ml/input/config/hyperparameters.json  -- hyperparameters
  /opt/ml/model/                 -- saved model artifacts (tarred to model.tar.gz)
"""
import argparse
import glob
import json
import os
import sys
import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

import bundle
import tracking

# Allow `import plugins` when running as a script from the src/ dir or via
# SageMaker's entry_point mechanism (which adds src/ to sys.path).
sys.path.insert(0, os.path.dirname(__file__))
from plugins import get_plugin, list_plugins  # noqa: E402
from plugins.base import TrainingPlugin  # noqa: E402

HP_PATH = "/opt/ml/input/config/hyperparameters.json"

# Map weights-format choice to (filename, dump fn, load fn). Loading is
# dispatched purely on the filename's extension recorded in config.json,
# so swapping formats is a write-time decision the loader picks up
# transparently.
def _save_joblib(obj, path):
    joblib.dump(obj, path)
def _load_joblib(path, **_):
    return joblib.load(path)
def _save_skops(obj, path):
    import skops.io as sio
    sio.dump(obj, path)
def _load_skops(path, trusted=None):
    import skops.io as sio
    # `trusted=[]` accepts only types skops considers safe (sklearn's
    # builtins). For custom classes you'd pass them by name here.
    return sio.load(path, trusted=trusted or [])

WEIGHTS_FORMATS = {
    "joblib": ("model.joblib", _save_joblib, _load_joblib),
    "skops":  ("model.skops",  _save_skops,  _load_skops),
}


def load_hyperparameters():
    """SageMaker writes hyperparameters to a JSON file; values arrive as strings."""
    if not os.path.exists(HP_PATH):
        return {}
    with open(HP_PATH) as f:
        return json.load(f)


def main():
    hp = load_hyperparameters()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=int(hp.get("n-estimators", 100)))
    parser.add_argument("--max-depth", type=int, default=int(hp.get("max-depth", 5)))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--weights-format", choices=list(WEIGHTS_FORMATS),
                        default=hp.get("weights-format", "joblib"),
                        help="how to serialize the trained model (joblib or skops)")
    parser.add_argument("--plugin", type=str,
                        default=hp.get("plugin", "default"),
                        help=f"training plugin (metric module). available: {list_plugins()}")
    args, _ = parser.parse_known_args()
    weights_file, save_weights, _ = WEIGHTS_FORMATS[args.weights_format]

    plugin = get_plugin(args.plugin)
    print(f"plugin: {plugin.name}")

    # Unified params dict passed to plugin.build_model().
    # Keys are normalized to underscores (SageMaker uses hyphens in hp.json).
    # CLI flags for n_estimators / max_depth override hp.json values.
    params = {k.replace("-", "_"): str(v) for k, v in hp.items()}
    params["n_estimators"] = str(args.n_estimators)
    params["max_depth"] = str(args.max_depth)

    os.makedirs(args.model_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.train, "*.csv"))
                   + glob.glob(os.path.join(args.train, "*.parquet")))
    if not files:
        raise SystemExit(f"no .csv or .parquet file found in {args.train}")
    # Prefer files named training.* — the canonical name for the model-input
    # dataset. Otherwise fall back to alphabetical first. This lets a prepare
    # script write sibling files (e.g. ground_truth.parquet from
    # simulate/scenarios/*) into the same dir without confusing the trainer.
    preferred = [f for f in files if os.path.basename(f).startswith("training.")]
    path = preferred[0] if preferred else files[0]
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    print(f"loaded {path}: {len(df)} rows, {len(df.columns)} columns")

    X, y = plugin.prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = plugin.build_model(params)
    run_params = {**params, "plugin": plugin.name, "dataset_file": os.path.basename(path)}
    with tracking.mlflow_run(run_name=f"{plugin.name}-train", params=run_params,
                             tags={"framework": type(clf).__module__.split(".")[0],
                                   "plugin": plugin.name}):
        clf.fit(X_train, y_train)

        # The plugin owns the metric — accuracy / ROC-AUC for
        # classification, R² for regression by default. Higher is
        # better, by convention.
        #
        # We tolerate two evaluate() signatures so old plugins still
        # work and the agent's regenerated plugins can use either:
        #   new: evaluate(model, X_test, y_true)   ← can use predict_proba
        #   old: evaluate(y_true, y_pred)           ← simpler, predictions only
        import inspect
        n_params = len(inspect.signature(plugin.evaluate).parameters)
        if n_params >= 3:
            metric_name, metric_value = plugin.evaluate(clf, X_test, y_test)
        else:
            metric_name, metric_value = plugin.evaluate(y_test, clf.predict(X_test))
        print(f"{metric_name}={metric_value:.4f}")
        tracking.log_metrics({metric_name: metric_value})

        # --- write the standard model bundle -----------------------------
        # config.json: how to rebuild the model. For sklearn the "code" is
        # the sklearn library itself, so we just need the estimator class
        # name, its init params, and the feature schema.
        config = {
            "plugin": plugin.name,
            "task": plugin.task,
            "framework": "sklearn",
            "framework_version": sklearn.__version__,
            "estimator": type(clf).__name__,
            "estimator_module": type(clf).__module__,
            "params": clf.get_params(),
            "weights_file": weights_file,
            "weights_format": args.weights_format,
            "feature_names": list(X.columns),
            "metric_name": metric_name,
        }
        # `classes_` only exists on classifiers — preserve the existing
        # field there but skip it for regressors.
        if hasattr(clf, "classes_"):
            config["classes"] = [
                int(c) if hasattr(c, "item") else c for c in clf.classes_.tolist()
            ]
        config.update(plugin.extra_config(clf, X))
        bundle.save_config(args.model_dir, config)

        # weights: dispatched by --weights-format. joblib (pickle) is
        # canonical; skops is a safer-pickle drop-in (allowlist of
        # trusted classes, no arbitrary code execution on load).
        save_weights(clf, os.path.join(args.model_dir, weights_file))

        # optional pickle-free bundle export — plugins that implement
        # export_bundle() write framework-native files here and override
        # load_bundle() to load from them instead of the joblib weights.
        plugin.export_bundle(clf, args.model_dir)

        # metadata: provenance + metrics. Augments, never gates loading.
        # If the prepare-* script wrote a lineage.json, embed it so the
        # bundle carries an audit trail back to the source data.
        extras = {
            metric_name: metric_value,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "dataset_file": os.path.basename(path),
        }
        lineage = bundle.load_lineage(args.train)
        if lineage:
            extras["data_lineage"] = lineage
        bundle.save_metadata(args.model_dir, extras=extras)

        # log the bundle as opaque MLflow artifacts (no-op if disabled).
        tracking.log_bundle(args.model_dir)

        # also register a PyFunc wrapper so the model appears in the
        # MLflow Models tab / Registry. Skip if the plugin owns its own
        # bundle format (load_bundle overridden) — the pyfunc wrapper
        # relies on the default joblib weights which may not exist.
        _uses_custom_bundle = (
            type(plugin).load_bundle is not TrainingPlugin.load_bundle
        )
        if _uses_custom_bundle:
            # Plugin owns its own bundle format — register the artifact
            # path directly so the model appears in the MLflow registry
            # without pickling anything.
            _registered_name = os.environ.get("MLFLOW_REGISTERED_MODEL")
            if _registered_name and tracking._enabled():
                import mlflow
                from mlflow.tracking import MlflowClient
                _client = MlflowClient()
                _run = mlflow.active_run()
                try:
                    _client.create_registered_model(_registered_name)
                except Exception:
                    pass  # already exists
                _client.create_model_version(
                    name=_registered_name,
                    source=f"{_run.info.artifact_uri}/model",
                    run_id=_run.info.run_id,
                )
        else:
            tracking.register_bundle_as_pyfunc(
                model_dir=args.model_dir,
                model_fn=model_fn,
                registered_name=os.environ.get("MLFLOW_REGISTERED_MODEL", "sagebaker-sklearn"),
            )


class _ThresholdedModel:
    """Wraps a binary classifier so .predict() applies a configured
    decision threshold (config.json's `prediction_threshold`) instead
    of sklearn's default 0.5.

    Forwards every other attribute (predict_proba, classes_,
    feature_importances_, etc.) to the underlying model. Identity is
    important here: callers that need the raw estimator can reach it
    via `.raw_model`.

    Lives here in train.py so every serving path that uses model_fn
    — local_serve.py, AWS DLC's SKLearn inference container, the
    BundleWrapper PyFunc — gets the same threshold behaviour for free.
    """
    def __init__(self, raw_model, threshold: float):
        self.raw_model = raw_model
        self.threshold = float(threshold)

    def predict(self, X):
        proba = self.raw_model.predict_proba(X)
        # Multiclass: threshold doesn't apply; fall back to argmax (which
        # is what sklearn's predict does anyway).
        if proba.shape[1] != 2:
            return self.raw_model.predict(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def __getattr__(self, name):
        # Only called if `name` not found on the wrapper itself; forwards
        # to the raw model for predict_proba / classes_ / etc.
        return getattr(self.raw_model, name)


def model_fn(model_dir):
    """Inference contract: read the bundle, return a model object.

    SageMaker's SKLearn inference container calls this. The same function
    works for any caller (a test, a local script, a custom MLflow PyFunc)
    because it knows nothing about SageMaker — it just loads the bundle.

    Dispatch on `weights_format` recorded in config.json so swapping the
    format at training time doesn't require updating any caller.

    If the bundle has a non-default `prediction_threshold` recorded and
    the model is a binary classifier with `predict_proba`, returns a
    `_ThresholdedModel` wrapper that applies the threshold. Otherwise
    returns the raw estimator. Threshold travels with the bundle, so
    every serving path (DLC, MLflow PyFunc, local_serve) sees the same
    decision boundary.
    """
    config = bundle.load_config(model_dir)
    fmt = config.get("weights_format", "joblib")
    if fmt not in WEIGHTS_FORMATS:
        raise ValueError(f"unknown weights_format {fmt!r}; expected one of "
                         f"{sorted(WEIGHTS_FORMATS)}")
    _, _, load_weights = WEIGHTS_FORMATS[fmt]
    raw = load_weights(os.path.join(model_dir, config["weights_file"]))

    threshold = config.get("prediction_threshold")
    task = config.get("task", "classification")
    if (threshold is not None and float(threshold) != 0.5
            and task == "classification" and hasattr(raw, "predict_proba")):
        return _ThresholdedModel(raw, threshold)
    return raw


if __name__ == "__main__":
    main()
