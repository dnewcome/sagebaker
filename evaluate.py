"""Evaluate a trained model bundle against a held-out test set.

Runs as the evaluation step in the SageMaker Pipeline (pipeline.py),
producing metrics.json that the metric-gate ConditionStep consumes.
Also runnable locally for testing.

SageMaker Processing inputs/outputs (defaults match pipeline.py):
  /opt/ml/processing/model/model.tar.gz   — TrainingStep output
  /opt/ml/processing/test/*.parquet       — held-out parquet (or train data, for sketch)
  /opt/ml/processing/evaluation/metrics.json   — written here

metrics.json schema:
  {
    "validation_accuracy": 0.83,        # what the ConditionStep reads
    "precision": ..., "recall": ..., "f1": ...,
    "averaging": "binary" | "macro",
    "n_test": 42,
    "classes": [0, 1]
  }

Local testing:
  python evaluate.py --model ./model_sklearn --test ./data --output ./eval
  python evaluate.py --model .sm-scratch/tmp.../compressed_artifacts/model.tar.gz \\
                    --test ./data --output ./eval
"""
import argparse
import glob
import json
import os
import tarfile
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)


def resolve_bundle_dir(model_path):
    """Accept a directory or a .tar.gz; return a path to the extracted bundle."""
    if os.path.isdir(model_path):
        return model_path
    if model_path.endswith((".tar.gz", ".tgz")):
        tmp = tempfile.mkdtemp(prefix="bundle_")
        with tarfile.open(model_path) as tf:
            tf.extractall(tmp)
        return tmp
    raise SystemExit(f"--model must be a directory or .tar.gz, got: {model_path}")


def load_model(model_dir):
    """Mirror src/train.py's model_fn — dispatch by weights_format in config.json.

    Inlined here (rather than importing from train) so this script runs in
    a SageMaker Processing container without needing src/ on PYTHONPATH.
    """
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    weights_path = os.path.join(model_dir, cfg["weights_file"])
    fmt = cfg.get("weights_format", "joblib")
    if fmt == "joblib":
        return joblib.load(weights_path)
    if fmt == "skops":
        import skops.io as sio
        return sio.load(weights_path, trusted=[])
    raise ValueError(f"unsupported weights_format: {fmt!r}")


def find_test_data(test_dir):
    files = sorted(
        glob.glob(os.path.join(test_dir, "*.csv"))
        + glob.glob(os.path.join(test_dir, "*.parquet"))
    )
    if not files:
        raise SystemExit(f"no .csv or .parquet found in {test_dir}")
    return files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/ml/processing/model/model.tar.gz",
                        help="bundle directory or .tar.gz")
    parser.add_argument("--test", default="/opt/ml/processing/test/",
                        help="dir containing the test CSV/parquet")
    parser.add_argument("--output", default="/opt/ml/processing/evaluation/",
                        help="dir to write metrics.json")
    args = parser.parse_args()

    bundle_dir = resolve_bundle_dir(args.model)
    print(f"bundle dir: {bundle_dir}")
    print(f"  contents: {sorted(os.listdir(bundle_dir))}")

    model = load_model(bundle_dir)
    print(f"  loaded: {type(model).__name__}")

    data_path = find_test_data(args.test)
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    print(f"test data: {data_path} ({len(df)} rows)")

    # Drop the same Feast bookkeeping columns the trainer drops
    feature_cols = [c for c in df.columns
                    if c not in {"target", "signal_id", "event_timestamp"}]
    X = df[feature_cols]
    y_true = df["target"].astype(int)
    y_pred = model.predict(X)

    accuracy = float(accuracy_score(y_true, y_pred))
    avg = "binary" if y_true.nunique() == 2 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg, zero_division=0
    )

    metrics = {
        "validation_accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "averaging": avg,
        "n_test": int(len(y_true)),
        "classes": sorted(int(c) for c in y_true.unique()),
    }

    print(f"\nmetrics:\n{json.dumps(metrics, indent=2)}")
    print(f"\nclassification report:\n{classification_report(y_true, y_pred, zero_division=0)}")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.output, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
