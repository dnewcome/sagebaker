# Product matching (cross-retailer dedup)

Given two product offerings from different retailers — possibly with
different SKUs, sometimes-missing GTINs, and noised titles — predict
whether they're the same canonical product. This is the
**deduplication** problem: a real product catalog has the same iPhone
listed on Amazon, Walmart, eBay, etc., and you need to know they're
all the same thing.

## What it is

A binary pair classifier on cross-retailer pairs. Strong signal when
both retailers provide GTIN (deterministic match); harder cases are
eBay-style listings without GTIN where title similarity, brand
match, category, and price ratio have to carry the load.

The synthetic catalog (`simulate/scenarios/product_catalog`) is
deliberately messy: 6 retailers (Amazon, Walmart, Target, Best Buy,
eBay, Sam's Club), per-retailer SKU formats, retailer-specific GTIN
provision rates (Amazon ~95%, eBay ~28%), title-noise transforms
(brand reordering, lowercase, marketing fluff), retailer-specific
markups, member pricing on Sam's Club, and stock state.

## Quickstart

```bash
make data-products            # ./data/products/  (60 canonicals × 6 retailers = 360 rows)
make data-matcher-pairs       # ./data/matcher/   (5K balanced positive/negative pairs)
make train-product-matcher    # bundle in ./models/product_matcher/
                              # → validation_auc≈0.997 (high — see notes)
```

The high AUC is honest: GTIN match is deterministic when both
retailers have one (most pairs). The interesting cases are the
GTIN-missing pairs where the model must lean on title Jaccard,
brand-token match, category match, and price ratio.

## Files

| Path | What it is |
| --- | --- |
| [`src/plugins/product_matcher.py`](../../src/plugins/product_matcher.py) | The matcher plugin (binary HistGB on pair features) |
| [`prep/prepare_matcher_pairs.py`](../../prep/prepare_matcher_pairs.py) | Pair sampler: cross-retailer positives, optional same-category negatives |
| [`simulate/scenarios/product_catalog.py`](../../simulate/scenarios/product_catalog.py) | Synthetic catalog generator |
| [`simulate/products.py`](../../simulate/products.py) | Canonical-product templates + retailer configs + title noiser |
| `data/products/` | Generated catalog + ground truth (gitignored) |
| `data/matcher/` | Pair-feature parquet (gitignored) |
| `models/product_matcher/` | Bundle output (gitignored) |

## Try it different ways

### Make the negatives harder

```bash
.venv/bin/python prep/prepare_matcher_pairs.py \
  --input ./data/products --output ./data/matcher \
  --n-pairs 5000 --same-category-negatives
```

`--same-category-negatives` biases negative sampling toward pairs
inside the same category (e.g. two different smartphones rather
than a smartphone vs a stand mixer). Forces the model to use
finer-grained signals than category match.

### Cascade with semantic retrieval

The realistic shape for production deduplication is **two-stage**:

1. *Recall* with [semantic-search](../semantic-search/) — given an
   offering, retrieve top-K semantic candidates from FAISS.
2. *Rank* with `product_matcher` — for each (query, candidate)
   pair, predict same-canonical.

Both halves exist as separate plugins on the same dataset; combining
them is one make-target away once you wire a small driver.

### Tweak retailer behavior

Edit [`simulate/products.py`](../../simulate/products.py) — `DEFAULT_RETAILERS`
controls per-retailer GTIN rates, title noise, price markups, member
pricing, and stock-state. Drop eBay's GTIN rate to 0.10 to make the
hard subset of pairs harder; raise everyone's title noise rate to
test feature robustness.

## Scale to production

For a real catalog at work, the path is:

1. **Replace the synthetic catalog**: point `prep/prepare_matcher_pairs.py`
   at a real catalog parquet that has the same column shape
   (`catalog_id`, `retailer`, `sku`, `gtin`, `title`, `category`,
   `list_price`, plus a `true_canonical_id` ground-truth column for
   the labeled subset). Sage-baker's column convention isn't
   load-bearing — adapt the prep script to your real schema.
2. **Train at scale**: the same `product_matcher` plugin on millions
   of cross-retailer pairs. Bundle goes to prod MLflow + S3 via the
   [Local iteration vs production push](../../README.md#local-iteration-vs-production-push)
   workflow.
3. **Serve as a pair-scoring API**: client sends `(offering_a,
   offering_b)`, server returns `P(same_canonical)`. Threshold
   chosen offline, baked into `config.json`.
