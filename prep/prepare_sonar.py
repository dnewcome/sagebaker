"""Fetch the Connectionist Bench (Sonar Rocks vs Mines) dataset.

Same dataset as the Kaggle "Underwater Sonar Signals" listing, originally from
Gorman & Sejnowski (1988). Pulled from a public mirror so no Kaggle auth is
needed. 208 instances, 60 numeric sonar-frequency features, binary label
(R = rock, M = mine).

Also writes Feast-friendly parquet versions with a synthetic
event_timestamp + signal_id, split into a features file (the 60 sonar
bands) and a labels file (the target). The trainers in src/ that don't
use Feast still read data/sonar.csv.

Drops a `data/lineage.json` recording the source URL, fetch timestamp,
and sha256 of the resulting CSV — the trainers pick this up and embed
it in the model bundle's metadata.json so every trained model carries
an audit trail back to the data it was trained on.
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone

import pandas as pd

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
DATA_DIR = "data"
FEAST_DATA_DIR = os.path.join("feature_repo", "data")

# wipe stale data
for d in (DATA_DIR, FEAST_DATA_DIR):
    if os.path.isdir(d):
        shutil.rmtree(d)
os.makedirs(DATA_DIR)
os.makedirs(FEAST_DATA_DIR)

cols = [f"f{i}" for i in range(60)] + ["target"]
df = pd.read_csv(URL, header=None, names=cols)
df["target"] = df["target"].map({"R": 0, "M": 1})

# 1) plain CSV — used by the non-Feast trainers (src/train.py, src/train_torch.py)
df.to_csv(os.path.join(DATA_DIR, "sonar.csv"), index=False)

# 2) Feast layout — features and labels in separate parquet tables, keyed by
#    a synthetic signal_id + event_timestamp. Real-world feature stores
#    almost always have features and labels in different tables; mirroring
#    that here keeps the prototype faithful to the pattern.
df = df.assign(
    signal_id=range(len(df)),
    event_timestamp=[
        datetime.now(timezone.utc) - timedelta(hours=len(df) - i)
        for i in range(len(df))
    ],
)

features_df = df[["signal_id", "event_timestamp"] + [f"f{i}" for i in range(60)]]
labels_df = df[["signal_id", "event_timestamp", "target"]]

features_df.to_parquet(os.path.join(FEAST_DATA_DIR, "sonar_features.parquet"), index=False)
labels_df.to_parquet(os.path.join(FEAST_DATA_DIR, "sonar_labels.parquet"), index=False)

# lineage manifest — picked up by trainers and embedded in bundle metadata
csv_path = os.path.join(DATA_DIR, "sonar.csv")
data_hash = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
lineage = {
    "source": "url",
    "url": URL,
    "fetched_at": datetime.now(timezone.utc).isoformat(),
    "dataset_sha256": data_hash,
    "dataset_n_rows": len(df),
}
with open(os.path.join(DATA_DIR, "lineage.json"), "w") as f:
    json.dump(lineage, f, indent=2)

print(f"wrote {DATA_DIR}/sonar.csv ({len(df)} rows)")
print(f"wrote {FEAST_DATA_DIR}/sonar_features.parquet (signal_id, event_timestamp, f0..f59)")
print(f"wrote {FEAST_DATA_DIR}/sonar_labels.parquet (signal_id, event_timestamp, target)")
print(f"wrote {DATA_DIR}/lineage.json (sha256: {data_hash[:16]}...)")
print(f"  {(df['target'] == 0).sum()} rocks, {(df['target'] == 1).sum()} mines")
