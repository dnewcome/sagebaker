"""LightGBM trainer — demonstrates a fully pickle-free serialization path.

LightGBM has its own native text-based model format: `booster.save_model(path)`
writes the trees as a human-readable text file (you can `cat` it). Loading
is `lgb.Booster(model_file=path)`. No pickle, no class coupling, no
version-migration drama. It's the boring-good answer to the sklearn
pickle problem when you're willing to use LightGBM-the-framework.

Same bundle layout as src/train.py — config.json, metadata.json, plus
`model.txt` instead of `model.joblib`. model_fn dispatches on the
`framework` field in config.json.

Run standalone (no SageMaker):
    python src/train_lightgbm.py --train ./data --model-dir ./model_lgb
"""
import argparse
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import bundle
import tracking

HP_PATH = "/opt/ml/input/config/hyperparameters.json"
WEIGHTS_FILE = "model.txt"


def load_hyperparameters():
    if not os.path.exists(HP_PATH):
        return {}
    with open(HP_PATH) as f:
        return json.load(f)


class LightGBMPredictor:
    """Adapter that gives a Booster the sklearn `.predict(X) -> labels` shape.

    The Booster itself returns class probabilities; we threshold (binary)
    or argmax (multiclass) so callers can use it interchangeably with
    sklearn estimators.
    """
    def __init__(self, booster):
        self.booster = booster
        self.num_class = booster.num_model_per_iteration() or 1

    def predict(self, X):
        out = self.booster.predict(X)
        if out.ndim == 1:
            return (out > 0.5).astype(int)
        return out.argmax(axis=1)

    def predict_proba(self, X):
        return self.booster.predict(X)


def main():
    hp = load_hyperparameters()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-leaves", type=int, default=int(hp.get("num-leaves", 31)))
    parser.add_argument("--learning-rate", type=float, default=float(hp.get("learning-rate", 0.1)))
    parser.add_argument("--num-iterations", type=int, default=int(hp.get("num-iterations", 100)))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args, _ = parser.parse_known_args()

    os.makedirs(args.model_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.train, "*.csv"))
                   + glob.glob(os.path.join(args.train, "*.parquet")))
    if not files:
        raise SystemExit(f"no .csv or .parquet file found in {args.train}")
    path = files[0]
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    print(f"loaded {path}: {len(df)} rows, {len(df.columns)} columns")

    feature_cols = [c for c in df.columns if c not in {"target", "signal_id", "event_timestamp"}]
    X = df[feature_cols]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_classes = int(np.unique(y).size)
    objective = "binary" if num_classes == 2 else "multiclass"
    lgb_params = {
        "objective": objective,
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "verbose": -1,
    }
    if objective == "multiclass":
        lgb_params["num_class"] = num_classes

    run_params = {**lgb_params, "num_iterations": args.num_iterations,
                  "dataset_file": os.path.basename(path)}
    with tracking.mlflow_run(run_name="lightgbm", params=run_params,
                             tags={"framework": "lightgbm"}):
        train_ds = lgb.Dataset(X_train, label=y_train)
        booster = lgb.train(lgb_params, train_ds, num_boost_round=args.num_iterations)

        adapter = LightGBMPredictor(booster)
        acc = accuracy_score(y_test, adapter.predict(X_test))
        print(f"validation_accuracy={acc:.4f}")
        tracking.log_metrics({"validation_accuracy": acc})

        # config.json: same envelope as the sklearn trainer, just a
        # different framework + native text-format weights file.
        bundle.save_config(args.model_dir, {
            "framework": "lightgbm",
            "framework_version": lgb.__version__,
            "estimator": "Booster",
            "params": lgb_params,
            "weights_file": WEIGHTS_FILE,
            "weights_format": "lightgbm",
            "feature_names": list(X.columns),
            "num_classes": num_classes,
            "objective": objective,
        })

        # native text format — pickle-free, human-readable, completely
        # decoupled from this Python process's class definitions.
        booster.save_model(os.path.join(args.model_dir, WEIGHTS_FILE))

        bundle.save_metadata(args.model_dir, extras={
            "validation_accuracy": acc,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "dataset_file": os.path.basename(path),
        })

        tracking.log_bundle(args.model_dir)
        tracking.register_bundle_as_pyfunc(
            model_dir=args.model_dir,
            model_fn=model_fn,
            registered_name=os.environ.get("MLFLOW_REGISTERED_MODEL", "sage-baker-lightgbm"),
        )


def model_fn(model_dir):
    """Load a LightGBM bundle and return a predictor with sklearn-style API.

    `lgb.Booster(model_file=...)` reads the text format directly — no
    pickle, no class lookup. We wrap in `LightGBMPredictor` so callers
    that already use the sklearn `.predict(X) -> labels` shape work
    unchanged.
    """
    config = bundle.load_config(model_dir)
    booster = lgb.Booster(model_file=os.path.join(model_dir, config["weights_file"]))
    return LightGBMPredictor(booster)


if __name__ == "__main__":
    main()
