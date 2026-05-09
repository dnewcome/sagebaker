"""Write the iris dataset to data/iris.csv so SageMaker can mount it as a channel."""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import sklearn
from sklearn.datasets import load_iris

if os.path.isdir("data"):
    shutil.rmtree("data")
os.makedirs("data")

df = load_iris(as_frame=True).frame
csv_path = "data/iris.csv"
df.to_csv(csv_path, index=False)

# lineage manifest — trainers embed this in bundle metadata
data_hash = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
lineage = {
    "source": "sklearn.datasets.load_iris",
    "sklearn_version": sklearn.__version__,
    "fetched_at": datetime.now(timezone.utc).isoformat(),
    "dataset_sha256": data_hash,
    "dataset_n_rows": len(df),
}
with open("data/lineage.json", "w") as f:
    json.dump(lineage, f, indent=2)

print(f"wrote {csv_path} ({len(df)} rows)")
print(f"wrote data/lineage.json (sha256: {data_hash[:16]}...)")
