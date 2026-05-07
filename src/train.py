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
import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import bundle
import tracking

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
    args, _ = parser.parse_known_args()
    weights_file, save_weights, _ = WEIGHTS_FORMATS[args.weights_format]

    os.makedirs(args.model_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.train, "*.csv"))
                   + glob.glob(os.path.join(args.train, "*.parquet")))
    if not files:
        raise SystemExit(f"no .csv or .parquet file found in {args.train}")
    path = files[0]
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    print(f"loaded {path}: {len(df)} rows, {len(df.columns)} columns")
    # drop bookkeeping columns Feast adds (entity + timestamp) if present
    feature_cols = [c for c in df.columns if c not in {"target", "signal_id", "event_timestamp"}]
    X = df[feature_cols]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    run_params = {"n-estimators": args.n_estimators, "max-depth": args.max_depth,
                  "dataset_file": os.path.basename(path)}
    with tracking.mlflow_run(run_name="sklearn-rf", params=run_params,
                             tags={"framework": "sklearn"}):
        clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                     max_depth=args.max_depth, random_state=42)
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"validation_accuracy={acc:.4f}")
        tracking.log_metrics({"validation_accuracy": acc})

        # --- write the standard model bundle -----------------------------
        # config.json: how to rebuild the model. For sklearn the "code" is
        # the sklearn library itself, so we just need the estimator class
        # name, its init params, and the feature schema.
        bundle.save_config(args.model_dir, {
            "framework": "sklearn",
            "framework_version": sklearn.__version__,
            "estimator": type(clf).__name__,
            "estimator_module": type(clf).__module__,
            "params": clf.get_params(),
            "weights_file": weights_file,
            "weights_format": args.weights_format,
            "feature_names": list(X.columns),
            "classes": [int(c) if hasattr(c, "item") else c for c in clf.classes_.tolist()],
        })

        # weights: dispatched by --weights-format. joblib (pickle) is
        # canonical; skops is a safer-pickle drop-in (allowlist of
        # trusted classes, no arbitrary code execution on load).
        save_weights(clf, os.path.join(args.model_dir, weights_file))

        # metadata: provenance + metrics. Augments, never gates loading.
        bundle.save_metadata(args.model_dir, extras={
            "validation_accuracy": acc,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "dataset_file": os.path.basename(path),
        })

        # log the bundle as opaque MLflow artifacts (no-op if disabled).
        tracking.log_bundle(args.model_dir)

        # also register a PyFunc wrapper so the model appears in the
        # MLflow Models tab / Registry. The wrapper calls model_fn at
        # load time — MLflow never pickles RandomForestClassifier itself.
        tracking.register_bundle_as_pyfunc(
            model_dir=args.model_dir,
            model_fn=model_fn,
            registered_name=os.environ.get("MLFLOW_REGISTERED_MODEL", "sage-baker-sklearn"),
        )


def model_fn(model_dir):
    """Inference contract: read the bundle, return a model object.

    SageMaker's SKLearn inference container calls this. The same function
    works for any caller (a test, a local script, a custom MLflow PyFunc)
    because it knows nothing about SageMaker — it just loads the bundle.

    Dispatch on `weights_format` recorded in config.json so swapping the
    format at training time doesn't require updating any caller.
    """
    config = bundle.load_config(model_dir)
    fmt = config.get("weights_format", "joblib")
    if fmt not in WEIGHTS_FORMATS:
        raise ValueError(f"unknown weights_format {fmt!r}; expected one of "
                         f"{sorted(WEIGHTS_FORMATS)}")
    _, _, load_weights = WEIGHTS_FORMATS[fmt]
    return load_weights(os.path.join(model_dir, config["weights_file"]))


if __name__ == "__main__":
    main()
