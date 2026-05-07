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
WEIGHTS_FILE = "model.joblib"


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
    args, _ = parser.parse_known_args()

    csvs = sorted(glob.glob(os.path.join(args.train, "*.csv")))
    if not csvs:
        raise SystemExit(f"no CSV file found in {args.train}")
    df = pd.read_csv(csvs[0])
    print(f"loaded {csvs[0]}: {len(df)} rows, {len(df.columns)} columns")
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    run_params = {"n-estimators": args.n_estimators, "max-depth": args.max_depth,
                  "dataset_file": os.path.basename(csvs[0])}
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
            "weights_file": WEIGHTS_FILE,
            "feature_names": list(X.columns),
            "classes": [int(c) if hasattr(c, "item") else c for c in clf.classes_.tolist()],
        })

        # weights: framework-specific blob. For sklearn, joblib (pickle) is
        # the canonical format — pin framework_version above to make safe.
        joblib.dump(clf, os.path.join(args.model_dir, WEIGHTS_FILE))

        # metadata: provenance + metrics. Augments, never gates loading.
        bundle.save_metadata(args.model_dir, extras={
            "validation_accuracy": acc,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "dataset_file": os.path.basename(csvs[0]),
        })

        # log the bundle as opaque MLflow artifacts (no-op if disabled).
        tracking.log_bundle(args.model_dir)


def model_fn(model_dir):
    """Inference contract: read the bundle, return a model object.

    SageMaker's SKLearn inference container calls this. The same function
    works for any caller (a test, a local script, a custom MLflow PyFunc)
    because it knows nothing about SageMaker — it just loads the bundle.
    """
    config = bundle.load_config(model_dir)
    return joblib.load(os.path.join(model_dir, config["weights_file"]))


if __name__ == "__main__":
    main()
