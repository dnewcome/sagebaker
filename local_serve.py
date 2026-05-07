"""Test inference against a trained model bundle.

Exercises the contract every inference path uses — `model_fn(model_dir)`
returns a model object, you call .predict on it. SageMaker's inference
container does exactly this internally; so does our MLflow PyFunc
wrapper. Testing model_fn here = testing all of them.

By default, looks at ./model_sklearn/ (the host-side trainer's output).
Pass `--artifact path/to/model.tar.gz` to test a SageMaker artifact
instead — but watch out for sklearn version skew between the trainer's
container and your host venv (this is exactly the pickle-coupling
problem the bundle architecture warns about; the framework_version
field in config.json is the signal).
"""
import argparse
import glob
import os
import sys
import tarfile
import tempfile

import importlib

import pandas as pd
import sklearn

sys.path.insert(0, "src")
import bundle  # type: ignore  # noqa: E402

# Each framework's trainer exposes its own model_fn. Dispatch by the
# `framework` field in config.json so this script handles any bundle
# layout regardless of which trainer produced it.
LOADER_MODULE = {
    "sklearn": "train",
    "torch": "train_torch",
    "lightgbm": "train_lightgbm",
}


def latest_dlc_artifact():
    """Find the most recent SageMaker model.tar.gz under .sm-scratch/."""
    SCRATCH = os.path.abspath(".sm-scratch")
    candidates = sorted(
        glob.glob(os.path.join(SCRATCH, "tmp*/compressed_artifacts/model.tar.gz")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def warn_version_skew(model_dir):
    """Compare the bundle's framework_version against the *importing*
    framework's actual version. Only sklearn really has the load-failure
    risk (joblib + pickle); torch/lightgbm have stable file formats."""
    cfg = bundle.load_config(model_dir)
    trained = cfg.get("framework_version")
    framework = cfg.get("framework")
    if framework == "sklearn":
        current = sklearn.__version__
    else:
        return  # only sklearn pickles, others have stable file formats
    if trained and trained != current:
        print(f"⚠ sklearn version skew: trained with {trained}, "
              f"loading with {current} — pickle load may fail")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./model_sklearn",
                        help="path to a bundle directory (default: ./model_sklearn)")
    parser.add_argument("--artifact", help="path to model.tar.gz; if set, extracted to a temp dir")
    args = parser.parse_args()

    if args.artifact:
        tmp = tempfile.mkdtemp(prefix="bundle_")
        with tarfile.open(args.artifact) as tf:
            tf.extractall(tmp)
        model_dir = tmp
        print(f"extracted {args.artifact} → {model_dir}")
    else:
        if not os.path.isdir(args.model_dir):
            sys.exit(f"no bundle at {args.model_dir} — run "
                     f"`python src/train.py --train ./data --model-dir ./model_sklearn` first")
        model_dir = args.model_dir

    print(f"bundle contents: {sorted(os.listdir(model_dir))}")
    warn_version_skew(model_dir)

    cfg = bundle.load_config(model_dir)
    fw = cfg.get("framework", "sklearn")
    mod_name = LOADER_MODULE.get(fw)
    if not mod_name:
        sys.exit(f"unknown framework {fw!r}; expected one of {sorted(LOADER_MODULE)}")
    print(f"framework: {fw} -> using {mod_name}.model_fn")
    model_fn = importlib.import_module(mod_name).model_fn
    model = model_fn(model_dir)
    print(f"loaded: {type(model).__name__}")

    df = pd.read_csv("data/sonar.csv")
    X = df.drop(columns=["target"]).head(5).values.tolist()
    y = df["target"].head(5).tolist()
    predictions = model.predict(X).tolist()

    print(f"\n  actual:    {y}")
    print(f"  predicted: {predictions}")
    print(f"  matches:   {sum(a == p for a, p in zip(y, predictions))} / {len(y)}")


if __name__ == "__main__":
    main()
