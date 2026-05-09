"""Fetch MovieLens-100K and write data/movielens.csv for the ALS plugin.

Public dataset (grouplens.org), no auth, ~1.7 MB. 100K ratings on a
1-5 scale from 943 users on 1682 movies. Schema is mapped to what the
ALS plugin expects: user_id, item_id, weight (+ original timestamp).

Usage:
    make data-movielens
    # or
    python prepare_movielens.py
"""
import hashlib
import io
import json
import os
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone

import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "movielens.csv")

if os.path.isdir(DATA_DIR):
    shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR)

print(f"downloading {URL}...")
with urllib.request.urlopen(URL) as resp:
    zip_bytes = resp.read()

with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
    with zf.open("ml-100k/u.data") as f:
        df = pd.read_csv(f, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

# Map to the ALS plugin's expected schema (user_id / item_id / weight).
# Stringify ids so they're stable keys not confused with array indexes
# downstream.
df = df.assign(
    user_id=df["user_id"].astype(str).map("user-{}".format),
    item_id=df["movie_id"].astype(str).map("item-{}".format),
    weight=df["rating"].astype(float),
)
df = df[["user_id", "item_id", "weight", "timestamp"]]
df.to_csv(OUT_FILE, index=False)

data_hash = hashlib.sha256(open(OUT_FILE, "rb").read()).hexdigest()
lineage = {
    "source": "url",
    "url": URL,
    "dataset_name": "MovieLens-100K",
    "fetched_at": datetime.now(timezone.utc).isoformat(),
    "dataset_sha256": data_hash,
    "dataset_n_rows": len(df),
    "n_users": int(df["user_id"].nunique()),
    "n_items": int(df["item_id"].nunique()),
    "rating_range": [float(df["weight"].min()), float(df["weight"].max())],
    "license": "https://files.grouplens.org/datasets/movielens/ml-100k-README.txt",
}
with open(os.path.join(DATA_DIR, "lineage.json"), "w") as f:
    json.dump(lineage, f, indent=2)

print(f"wrote {OUT_FILE}")
print(f"  {df['user_id'].nunique():,} users")
print(f"  {df['item_id'].nunique():,} items")
print(f"  {len(df):,} ratings")
density = len(df) / (df['user_id'].nunique() * df['item_id'].nunique())
print(f"  density: {density:.3f}")
print(f"wrote {DATA_DIR}/lineage.json (sha256: {data_hash[:16]}...)")
