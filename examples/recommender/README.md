# Recommender (collaborative filtering, ALS)

Top-K item recommendations from implicit user-item interactions.
"Users who liked X also liked …" — the classic collaborative
filtering problem. Different harness shape from supervised: no
train/test split on rows, no `target` column. Evaluation is leave-N-
out by user, scored with hit-rate@K / recall@K / NDCG@K.

## What it is

Implicit-feedback ALS (alternating least squares) factorizes the
user × item interaction matrix into low-rank user and item
embeddings. At inference time, recommendations for a user are the
top-K items by dot product against the user's vector.

This is the only scenario in sage-baker that uses **real public
data** (MovieLens-100K) rather than a synthetic generator —
collaborative filtering needs real user behavior signal that's hard
to fake convincingly.

## Quickstart

```bash
make install-recommender      # one-time: implicit + scipy + pyarrow
make data-movielens           # ~1.7 MB download from grouplens.org
make train-als                # bundle in ./models/als/
                              # → hit_rate@10≈0.82, recall@10≈0.16, ndcg@10≈0.21
```

These are typical numbers for ALS on MovieLens-100K — within the
published-results range for the dataset.

## Files

| Path | What it is |
| --- | --- |
| [`src/plugins/als.py`](../../src/plugins/als.py) | The ALS plugin |
| [`src/plugins/base_recommender.py`](../../src/plugins/base_recommender.py) | `RecommenderPlugin` base contract — different from `TrainingPlugin` |
| [`src/train_recommender.py`](../../src/train_recommender.py) | Recommender harness (no row-level train/test split; user-leave-N-out instead) |
| [`prep/prepare_movielens.py`](../../prep/prepare_movielens.py) | Fetches MovieLens-100K, maps schema to `user_id`/`item_id`/`weight` |
| `data/movielens.csv` | Real public dataset (gitignored after download) |
| `models/als/` | Bundle: ALS factors + item ID mapping (gitignored) |

## Try it different ways

### Different ALS hyperparameters

[`src/plugins/als.py`](../../src/plugins/als.py) has knobs for
`factors`, `regularization`, `iterations`, `alpha`. Standard
trade-offs: more factors → richer embeddings but more compute and
overfitting risk; higher α weights confidence in observed
interactions vs unobserved.

### Use synthetic data instead

`make data-als` generates a small synthetic interactions dataset for
quick iteration when you don't want to download MovieLens.

### Hit-rate@K is a coarse metric

For agent-loop iteration on this plugin, hit-rate@10 plateaus easily
because it's quantized. Switch to NDCG@10 or MAP@10 for finer-grained
signal — same lesson as the
[coarse-metric trap](../../README.md#autoresearch-style-agent-loop)
that hit DefaultPlugin earlier.

## Scale to production

Production recommenders typically need:

1. **Online serving** — user vector × item matrix dot product per
   request, at scale. Either precompute top-K offline (batch
   inference, write to a key-value store) or serve the matrix and
   compute online (managed approximate-NN service).
2. **Cold start** — new users / items have no interactions. ALS
   alone returns nothing useful. Production layers a content-based
   fallback (item embeddings from titles / categories — see
   [semantic-search](../semantic-search/)) or hybrid models.
3. **Fresh-data updates** — ALS is a batch model; production runs
   it nightly or on-demand. Streaming approaches (online matrix
   factorization, contextual bandits) are a different family.

Sage-baker's bundle covers (1) cleanly: `model_fn` returns the ALS
matrices, you pre-compute or serve as you choose. (2) and (3) are
beyond the current plugin scope.
