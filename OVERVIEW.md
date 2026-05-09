# sagebaker — overview

A re-orientation doc. When the project feels too big, start here.

## The whole project in one sentence

> A **bundle** is `config.json + metadata.json + weights`; everything
> else is workflow around producing, evaluating, and deploying bundles.

If you remember nothing else, it's that. Every other piece below is
either creating bundles, validating bundles, or shipping bundles.

## What's actually in the repo

```
              ┌──────────────────────────────────────────────────┐
              │  PLUGIN CONTRACT  (prepare / build_model /       │
              │                    evaluate / model_fn)          │
              └──────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌──────────┐          ┌───────────┐         ┌──────────┐
  │Supervised│          │Recommender│         │Retrieval │
  │  (×5)    │          │  (×1)     │         │  (×1)    │
  └──────────┘          └───────────┘         └──────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   train.py              train_              train_retrieval.py
   train_torch.py        recommender.py
   train_lightgbm.py
   train_feast.py
                              │
                              ▼
                    ┌──────────────────┐
                    │ models/<plugin>/ │
                    │   config.json    │  ← schema-of-truth
                    │   metadata.json  │  ← lineage / metric
                    │   weights file   │
                    └──────────────────┘
```

**3 plugin families × 7 plugins × 5 trainers** is the whole modeling
surface.

### The 7 plugins

| Family | Plugin | Task | Notes |
|---|---|---|---|
| Supervised | `default` | classification | reference plugin; sonar dataset |
| Supervised | `housing` | regression | california-housing; R² metric |
| Supervised | `clickstream` | classification | session-conversion |
| Supervised | `clickstream_linkage` | classification | pair-level identity |
| Supervised | `product_matcher` | classification | cross-retailer same-canonical |
| Recommender | `als` | top-K | MovieLens demo |
| Retrieval | `product_search` | semantic search | sentence-transformers + FAISS |

### Three workflows that produce/use bundles

| Workflow | Entry point | What it does |
|---|---|---|
| Manual | `make train` | One bundle from one plugin |
| Agent loop | `make agent` | Autoresearch loop edits a plugin file, keeps better, reverts worse |
| Productionize | `/productionize <plugin>` | Auto-generates a notebook that rebuilds the bundle from config + sanity-checks predictions match the pickle |

### Four integrations (all opt-in)

| Integration | What it adds | Off-switch |
|---|---|---|
| MLflow | Run tracking + model registry | Don't set `MLFLOW_TRACKING_URI` |
| Feast | Feature store; same `model_fn`, different feature source | Use the standard sklearn path |
| BigQuery | Data source + snapshot-based lineage | Use local CSV/parquet |
| SageMaker | Cloud training (DLC or BYOC) + endpoint deploy | Run locally |

### Tooling around the modeling code

- `pyproject.toml` with PEP 735 dependency groups (`base`, `torch`,
  `lightgbm`, `feast`, `bigquery`, `jupyter`, `recommender`,
  `retrieval`, `agent`, `dev`, `all`)
- Editable install (`pip install -e .`) so `import bundle, train,
  plugins` works from anywhere
- 14-test smoke suite: bundle round-trip, plugin contract,
  config-rebuild, lineage capture
- Claude Code skills: `/productionize` (currently the only one)

## What's *not* in the repo (parking lots — ignore until ready)

Scoping artifacts that capture future direction without blocking
present work. Nothing in them is half-built.

- **PLAN.md Phase 5** — LLM fine-tune sandbox (5a classification with
  LoRA, 5b generative deferred).
- **GH issue #2** — model-types extension brainstorm (XGBoost,
  time-series, NLP, vision, anomaly, etc.).
- **GH issue #3** — A/B testing + training on production outcomes
  design space.

## Focusing thought

The repo's job is **the bundle pattern + the plugin contract**.
Everything else (the agent loop, the productionize notebook, the
integrations) exists *because* those two things exist. If you ever
feel lost, ask: "is this code making a bundle, validating a bundle,
or shipping a bundle?" — every line should have an answer.

## See also

- [README.md](README.md) — full reference, deep dives, command
  recipes
- [CLAUDE.md](CLAUDE.md) — conventions for working in this repo
  with Claude Code
- [PLAN.md](PLAN.md) — phased roadmap (Phases 1–3 done, 4+
  candidates, 5 planned)
