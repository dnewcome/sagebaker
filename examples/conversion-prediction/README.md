# Conversion prediction

Predict whether a session converts (binary, ~6% positive rate) from
pre-decision events only. Demonstrates: imbalanced classification,
class-aware thresholding, the autoresearch agent loop, and the
local-iterate → cloud-train workflow end-to-end.

## What it is

A user lands on the site, generates events (`page_view`, `click`),
and either ends up converting (`add_to_cart` → `checkout` →
`conversion`) or doesn't. The model only sees pre-decision events
(page_view, click) — counting funnel events would deterministically
leak the label, since the simulator only appends them to converting
sessions.

Realistic shape: ~5–7% session conversion, cohort-driven
heterogeneity (the `intender` cohort converts ~2.5× the base rate),
**100% anonymous traffic by default** (`identified_user_fraction=0.0`,
matching real-world traffic where most users aren't logged in).

## Quickstart

```bash
make data-fuzzy           # generate ./data/fuzzy/  (8K events, 1.7K sessions, 500 users)
make train-clickstream    # bundle in ./models/clickstream/
                          # → validation_auc≈0.84
```

## Files

| Path | What it is |
| --- | --- |
| [`src/plugins/clickstream.py`](../../src/plugins/clickstream.py) | The plugin: feature engineering + estimator + threshold |
| [`simulate/scenarios/fuzzy_clickstream.py`](../../simulate/scenarios/fuzzy_clickstream.py) | Synthetic clickstream generator |
| [`program_clickstream.md`](../../program_clickstream.md) | Agent-loop constraints (anti-leakage rules, strategy hints) |
| `data/fuzzy/` | Generated training + ground truth (gitignored) |
| `models/clickstream/` | Bundle output: `config.json`, `metadata.json`, `model.joblib` (gitignored) |

## Try it different ways

### Tune the threshold without retraining

`prediction_threshold` is a field in `models/clickstream/config.json`
(default 0.15 for this plugin's ~6% positive class). Edit it directly
and re-load — the wrapper applies it automatically across MLflow
HTTP, SageMaker DLC, and `local_serve.py`. See
[main README → Serving](../../README.md#serving) for how this
propagates.

### Run the autoresearch agent loop

```bash
make agent-clickstream    # needs ANTHROPIC_API_KEY in .env
```

The agent edits `src/plugins/clickstream.py` iteratively, trains,
keeps proposals that beat the current best, reverts otherwise.
Constraints in `program_clickstream.md` block known leak patterns
(post-decision event reconstruction, value-as-feature, etc.). See
[main README → Autoresearch-style agent loop](../../README.md#autoresearch-style-agent-loop).

### Make the data harder

The default scenario has 100% anonymous users, near-unique
fingerprints, stable IP buckets per user. Dial in realistic mess:

```bash
.venv/bin/python prep/prepare_simulate.py --scenario fuzzy_clickstream \
  --output ./data/fuzzy/ \
  --identified-fraction 0.2 \
  --ip-drift 0.3 --fingerprint-drift 0.3 \
  --fingerprint-namespace-factor 0.3
```

This adds ~20% identifiable users, 30% per-session IP/fingerprint
drift (mobile/WiFi hops, device rotation), and fingerprint
collisions across users.

### Curl a served model

```bash
make mlflow-server        # terminal 1
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 make train-clickstream
make mlflow-serve-http    # terminal 2

curl -X POST http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_records": [{"n_page_views": 3, "n_clicks": 1, ...}]}'
# → {"predictions": [0]}   (class label after threshold applied)
```

See [main README → HTTP scoring server](../../README.md#http-scoring-server-curl-able)
for the full payload contract (55 session-level features).

## Scale to production

Same plugin code, different infra. The pieces that change going from
laptop to prod:

| | Local | Staging | Production |
| --- | --- | --- | --- |
| Data | `make data-fuzzy` (synthetic, 8K rows) | sample of real warehouse (10–100K rows) | full warehouse extract |
| Tracking | local SQLite mlflow | staging MLflow + S3 prefix | prod MLflow + S3 prefix |
| Trigger | `make train-clickstream` | CI runner on commit | scheduled / event-driven retrain |
| Compute | host venv | small EC2 / staging SageMaker job | full SageMaker training job (DLC) |
| Serving | `mlflow-serve-http` on laptop | staging endpoint | SageMaker endpoint via `pipeline.py` / `deploy_endpoint.py` |

What you commit to git: `src/plugins/clickstream.py` and any
adjustments to `program_clickstream.md`. What you do not commit: the
bundle artifacts (gitignored), local data, MLflow runs.

The bridge from local to cloud is the
**[Local iteration vs production push](../../README.md#local-iteration-vs-production-push)**
section in the main README. The pattern is "researchers push code,
cloud trains on full data" — sage-baker's design specifically
supports this without code changes between the two environments.

## Where it can go next

- **Threshold sweep tool** — load the bundle + a held-out set,
  sweep thresholds, write a precision/recall/F1 curve to
  `models/clickstream/threshold_sweep.json`. Tells you what
  threshold to set in `config.json` rather than guessing 0.15.
- **Cohort-conditional thresholds** — different threshold per
  cohort (intender vs browser). Currently flat 0.15 across all
  sessions; would need a small wrapper change.
- **Multi-output: convert + which category** — extend to predict
  not just conversion but conversion category. Closer to a real
  recommendation funnel.
