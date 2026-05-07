"""PyTorch training script — demonstrates the bundle layout with safetensors.

Same on-disk contract as src/train.py (config.json + weights file +
metadata.json), just with a different framework and weight format. The
bundle layout is the contract; what each framework writes inside it is
its own business.

Run standalone (no SageMaker needed):
    python src/train_torch.py --train ./data --model-dir ./model_torch

To run inside SageMaker Local Mode, build a torch BYOC image (or use the
AWS PyTorch DLC) following the same pattern as local_train.py /
local_train_dlc.py.
"""
import argparse
import glob
import json
import os

import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from sklearn.model_selection import train_test_split

import bundle
import tracking

HP_PATH = "/opt/ml/input/config/hyperparameters.json"
WEIGHTS_FILE = "model.safetensors"


# --- model class lives in code, versioned in git, hot-editable ----------
class SonarMLP(nn.Module):
    """Small MLP for binary classification.

    The class definition is the "code" half of the bundle. config.json
    only needs to record the constructor kwargs to rebuild the network.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_hyperparameters():
    if not os.path.exists(HP_PATH):
        return {}
    with open(HP_PATH) as f:
        return json.load(f)


def main():
    hp = load_hyperparameters()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=int(hp.get("epochs", 50)))
    parser.add_argument("--hidden-dim", type=int, default=int(hp.get("hidden-dim", 64)))
    parser.add_argument("--lr", type=float, default=float(hp.get("lr", 0.01)))
    parser.add_argument("--batch-size", type=int, default=int(hp.get("batch-size", 32)))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args, _ = parser.parse_known_args()

    os.makedirs(args.model_dir, exist_ok=True)

    csvs = sorted(glob.glob(os.path.join(args.train, "*.csv")))
    if not csvs:
        raise SystemExit(f"no CSV file found in {args.train}")
    df = pd.read_csv(csvs[0])
    print(f"loaded {csvs[0]}: {len(df)} rows, {len(df.columns)} columns")

    X = df.drop(columns=["target"]).values.astype("float32")
    y = df["target"].values.astype("int64")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_dim = X.shape[1]
    num_classes = int(y.max()) + 1

    run_params = {"epochs": args.epochs, "hidden_dim": args.hidden_dim,
                  "lr": args.lr, "batch_size": args.batch_size,
                  "dataset_file": os.path.basename(csvs[0])}
    with tracking.mlflow_run(run_name="torch-mlp", params=run_params,
                             tags={"framework": "torch"}):
        model = SonarMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_test_t = torch.from_numpy(X_test)
        y_test_t = torch.from_numpy(y_test)

        model.train()
        for epoch in range(args.epochs):
            perm = torch.randperm(len(X_train_t))
            epoch_loss = 0.0
            for i in range(0, len(X_train_t), args.batch_size):
                idx = perm[i:i + args.batch_size]
                optim.zero_grad()
                logits = model(X_train_t[idx])
                loss = loss_fn(logits, y_train_t[idx])
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
            tracking.log_metrics({"train_loss": epoch_loss}, step=epoch)

        model.eval()
        with torch.no_grad():
            acc = (model(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
        print(f"validation_accuracy={acc:.4f}")
        tracking.log_metrics({"validation_accuracy": acc})

        # --- write the bundle --------------------------------------------
        bundle.save_config(args.model_dir, {
            "framework": "torch",
            "framework_version": torch.__version__,
            "model_class": "SonarMLP",
            "model_module": "train_torch",
            "init_kwargs": {
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "num_classes": num_classes,
            },
            "weights_file": WEIGHTS_FILE,
            "feature_names": list(df.drop(columns=["target"]).columns),
        })

        # weights as safetensors — mmap-able, no pickle, no RCE on load.
        save_file(model.state_dict(), os.path.join(args.model_dir, WEIGHTS_FILE))

        bundle.save_metadata(args.model_dir, extras={
            "validation_accuracy": acc,
            "epochs": args.epochs,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "dataset_file": os.path.basename(csvs[0]),
        })

        tracking.log_bundle(args.model_dir)


def model_fn(model_dir):
    """Same loader contract as src/train.py.

    Reads config.json, instantiates the SonarMLP class with the recorded
    init_kwargs, and applies the safetensors weights. The model_class
    string is a *registry lookup*, not a pickle reference — only classes
    defined in this module are loadable, and you can audit which.
    """
    config = bundle.load_config(model_dir)

    REGISTRY = {"SonarMLP": SonarMLP}
    cls = REGISTRY[config["model_class"]]
    model = cls(**config["init_kwargs"])

    state_dict = load_file(os.path.join(model_dir, config["weights_file"]))
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    main()
