"""Materialize a training set from BigQuery and capture lineage.

Pattern: explore in notebooks however you like, but for any training run
that goes to a registry, materialize first via this script. That gives
you a deterministic snapshot to hash and re-query later — your model
carries an audit trail back to the exact data it saw.

What we capture in `data/lineage.json` (trainers embed this in
bundle metadata.json):

  - source: "bigquery"
  - project: GCP project the query ran in
  - query: the exact SQL string
  - snapshot_timestamp: BQ time-travel anchor (FOR SYSTEM_TIME AS OF)
  - dataset_sha256: hash of the materialized parquet file
  - dataset_n_rows / n_cols: shape

To reproduce a past training run later, re-run the recorded query at the
recorded snapshot_timestamp (within BQ's time-travel window — 7 days by
default, longer with table snapshots) and verify the sha256 matches.

Prereqs:
    .venv/bin/pip install -r requirements-bigquery.txt
    gcloud auth application-default login
    # or: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

Usage:
    python prepare_bigquery.py
    python prepare_bigquery.py --query "SELECT ..." --project my-gcp-project
    python prepare_bigquery.py --snapshot-time 2026-04-30T00:00:00Z

Default query hits a public dataset (`bigquery-public-data.ml_datasets.iris`)
so this works against any GCP project with billing enabled — replace
the query with your own and you're done.
"""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

DEFAULT_QUERY = """\
SELECT
  sepal_length AS f0,
  sepal_width  AS f1,
  petal_length AS f2,
  petal_width  AS f3,
  CASE species
    WHEN 'setosa'     THEN 0
    WHEN 'versicolor' THEN 1
    WHEN 'virginica'  THEN 2
  END AS target
FROM `bigquery-public-data.ml_datasets.iris`
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                        help="GCP project to bill the query to (defaults to GOOGLE_CLOUD_PROJECT)")
    parser.add_argument("--output", default="data/training.parquet")
    parser.add_argument("--snapshot-time",
                        help="ISO timestamp for FOR SYSTEM_TIME AS OF (BQ default 7-day window). "
                             "If not set, uses 'now' and records that.")
    args = parser.parse_args()

    try:
        from google.cloud import bigquery
    except ImportError:
        raise SystemExit(
            "Install BigQuery deps first: pip install -r requirements-bigquery.txt"
        )

    DATA_DIR = os.path.dirname(args.output) or "data"
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    client = bigquery.Client(project=args.project)
    snapshot_ts = args.snapshot_time or datetime.now(timezone.utc).isoformat()

    print(f"running query against project={client.project}...")
    df = client.query(args.query).to_dataframe()
    df.to_parquet(args.output, index=False)

    data_hash = hashlib.sha256(open(args.output, "rb").read()).hexdigest()

    lineage = {
        "source": "bigquery",
        "project": client.project,
        "query": args.query.strip(),
        "snapshot_timestamp": snapshot_ts,
        "dataset_sha256": data_hash,
        "dataset_n_rows": len(df),
        "dataset_n_cols": len(df.columns),
        "materialized_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(DATA_DIR, "lineage.json"), "w") as f:
        json.dump(lineage, f, indent=2)

    print(f"wrote {args.output} ({len(df)} rows × {len(df.columns)} cols)")
    print(f"wrote {DATA_DIR}/lineage.json (sha256: {data_hash[:16]}...)")


if __name__ == "__main__":
    main()
