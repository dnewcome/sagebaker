"""One-time: upload data/sonar.csv to BigQuery as sage_baker.sonar.

Idempotent — recreates the table on each run via WRITE_TRUNCATE. After
this, prep/prepare_bigquery.py can query the table like any other BQ source.

Prereqs:
  GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT set (via .env)
  data/sonar.csv exists — run `make data-sonar` first
  Service account needs roles/bigquery.dataEditor on the project

Usage:
  make bq-upload-sonar
"""
import os

import pandas as pd
from google.cloud import bigquery

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT:
    raise SystemExit(
        "GOOGLE_CLOUD_PROJECT not set. Add it to .env or export it."
    )

DATASET = "sage_baker"
TABLE = "sonar"
TABLE_ID = f"{PROJECT}.{DATASET}.{TABLE}"

if not os.path.exists("data/sonar.csv"):
    raise SystemExit("data/sonar.csv missing — run `make data-sonar` first.")

client = bigquery.Client(project=PROJECT)

# Create dataset if missing.
dataset_ref = bigquery.Dataset(f"{PROJECT}.{DATASET}")
dataset_ref.location = "US"
try:
    client.get_dataset(dataset_ref.reference)
    print(f"dataset {DATASET} already exists")
except Exception:
    client.create_dataset(dataset_ref)
    print(f"created dataset {DATASET}")

df = pd.read_csv("data/sonar.csv")
print(f"loaded {len(df)} rows × {len(df.columns)} cols from data/sonar.csv")

# Explicit schema so floats stay floats and target stays int.
schema = [bigquery.SchemaField(f"f{i}", "FLOAT64") for i in range(60)]
schema.append(bigquery.SchemaField("target", "INT64"))

job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    schema=schema,
)
job = client.load_table_from_dataframe(df, TABLE_ID, job_config=job_config)
job.result()

table = client.get_table(TABLE_ID)
print(f"uploaded → {TABLE_ID} ({table.num_rows} rows, {len(table.schema)} cols)")
print(f"\ntry it:")
print(f"  bq query --nouse_legacy_sql 'SELECT COUNT(*) FROM `{TABLE_ID}`'")
print(f"  make bq-data-sonar    # materialize via prep/prepare_bigquery.py + train")
