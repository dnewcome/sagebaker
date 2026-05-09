"""Build a pair-level dataset for product-matching from a product_catalog scenario.

Same shape as prepare_linkage.py, but for cross-retailer offerings of
the same canonical product instead of cross-session events of the same
user.

Reads `<input>/training.parquet` + `<input>/ground_truth.parquet`,
samples positive pairs (different retailer offerings of the same
canonical product) and negative pairs (different canonical products,
optionally same category to make negatives harder), computes pair
features, writes a single training.parquet ready for the standard
supervised harness with a binary `target` column.

Usage::

    python prepare_matcher_pairs.py --input ./data_products --output ./data_matcher \\
        --n-pairs 5000 [--same-category-negatives]

Pair features (all symmetric — order-independent):

  - same_gtin: 1 if both offerings have a GTIN AND they match. The
    deterministic feature when present; null when one or both lack GTIN.
  - both_have_gtin / one_has_gtin / no_gtin: GTIN provision pattern.
    Important because eBay's low GTIN rate makes these features
    informative.
  - same_brand_token: do the titles share their first word? (Brand
    is usually first or last; this is a cheap-and-noisy proxy.)
  - title_jaccard: token-set Jaccard similarity. Resilient to word
    reordering (which is one of the title-noise transforms).
  - title_token_overlap: count of shared lowercase tokens.
  - same_category: same category path string.
  - same_retailer: 1 if from the same retailer. (Negative pairs from
    same retailer are unusual — different SKUs from same retailer
    that map to the same canonical happens but rarely; same-retailer
    pairs in this sample are mostly negatives.)
  - log_price_diff_ratio: |log(price_a/price_b)|. Same canonical
    should have similar prices; widely different prices suggest
    different products.

Limitations / TODO
------------------
- Token-set features are very rough. A real production matcher would
  use a learned embedding similarity (cosine over the same embedder
  the retrieval plugin builds the index from). Adding it as a feature
  is straightforward — call the retrieval plugin's embedder on each
  title, compute cosine — but requires the retrieval bundle to
  exist and adds a dep.
- Brand normalization: "Apple" vs "APPLE INC" vs "apple" — currently
  case-folded but more would help.
- No sampling within candidate windows (same-category negatives is
  the only "harder negative" knob). Production matchers usually
  blocking-key to limit candidate pairs.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(s).lower()))


def _build_pair_features(a: pd.Series, b: pd.Series) -> dict:
    a_tokens = _tokens(a["title"])
    b_tokens = _tokens(b["title"])
    overlap = a_tokens & b_tokens
    union = a_tokens | b_tokens
    jaccard = len(overlap) / len(union) if union else 0.0

    # GTIN match — present on both?
    a_g = a["gtin"] if pd.notna(a["gtin"]) else None
    b_g = b["gtin"] if pd.notna(b["gtin"]) else None
    both_have = int(a_g is not None and b_g is not None)
    no_gtin = int(a_g is None and b_g is None)
    one_has = int(both_have == 0 and no_gtin == 0)
    same_gtin = int(both_have == 1 and a_g == b_g)

    # Brand token: first lowercased word of the title.
    def _brand_tok(s):
        ts = _TOKEN_RE.findall(str(s).lower())
        return ts[0] if ts else ""
    same_brand_token = int(_brand_tok(a["title"]) == _brand_tok(b["title"]))

    # Price ratio — symmetric via abs(log).
    pa = max(0.01, float(a.get("list_price", 0.0) or 0.0))
    pb = max(0.01, float(b.get("list_price", 0.0) or 0.0))
    log_price_diff = abs(math.log(pa) - math.log(pb))

    return {
        "same_gtin": same_gtin,
        "both_have_gtin": both_have,
        "one_has_gtin": one_has,
        "no_gtin": no_gtin,
        "same_brand_token": same_brand_token,
        "title_jaccard": jaccard,
        "title_token_overlap": len(overlap),
        "same_category": int(str(a.get("category", "")) == str(b.get("category", ""))),
        "same_retailer": int(a["retailer"] == b["retailer"]),
        "log_price_diff_ratio": log_price_diff,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="dir with training.parquet + ground_truth.parquet from product_catalog")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--same-category-negatives", action="store_true",
                        help="bias negatives toward same-category pairs (harder)")
    args = parser.parse_args()

    offerings = pd.read_parquet(os.path.join(args.input, "training.parquet"))
    gt = pd.read_parquet(os.path.join(args.input, "ground_truth.parquet"))
    merged = offerings.merge(
        gt[["catalog_id", "true_canonical_id", "true_category_path"]], on="catalog_id"
    )

    rng = np.random.default_rng(args.seed)
    n_half = args.n_pairs // 2

    # Group by canonical id — positives come from the same canonical, but
    # MUST be from different retailers (within-retailer pairs are
    # essentially trivial since they share SKU, GTIN, etc.).
    by_canonical = (
        merged.reset_index().groupby("true_canonical_id")["index"].apply(list).to_dict()
    )
    multi_offering = [c for c, idxs in by_canonical.items() if len(idxs) >= 2]

    positives = []
    while len(positives) < n_half:
        c = multi_offering[rng.integers(0, len(multi_offering))]
        idxs = by_canonical[c]
        i, j = rng.choice(idxs, size=2, replace=False)
        a, b = merged.iloc[i], merged.iloc[j]
        if a["retailer"] == b["retailer"]:
            # Same canonical via same retailer is unusual (rep duplicate
            # in their own catalog) — skip for cleaner positive sampling.
            continue
        positives.append((a, b))

    negatives = []
    n_total = len(merged)
    while len(negatives) < n_half:
        i = int(rng.integers(0, n_total))
        j = int(rng.integers(0, n_total))
        if i == j:
            continue
        a, b = merged.iloc[i], merged.iloc[j]
        if a["true_canonical_id"] == b["true_canonical_id"]:
            continue
        if args.same_category_negatives:
            if a["true_category_path"] != b["true_category_path"]:
                # Reject ~80% of cross-category negatives to bias toward
                # same-category (harder) pairs.
                if rng.random() < 0.8:
                    continue
        negatives.append((a, b))

    rows = []
    for label, pairs in [(1, positives), (0, negatives)]:
        for a, b in pairs:
            row = _build_pair_features(a, b)
            row["target"] = label
            rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    out_path = os.path.join(args.output, "training.parquet")
    df.to_parquet(out_path, index=False)

    src_lineage_path = os.path.join(args.input, "lineage.json")
    src_lineage = {}
    if os.path.exists(src_lineage_path):
        with open(src_lineage_path) as f:
            src_lineage = json.load(f)
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    lineage = {
        "source": "prepare_matcher_pairs",
        "input_dir": os.path.abspath(args.input),
        "input_lineage": src_lineage,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(df),
        "positive_fraction": float(df["target"].mean()),
        "same_category_negatives": args.same_category_negatives,
        "seed": args.seed,
        "dataset_sha256": sha,
        "feature_names": [c for c in df.columns if c != "target"],
    }
    with open(os.path.join(args.output, "lineage.json"), "w") as f:
        json.dump(lineage, f, indent=2)

    print(f"wrote {out_path} ({len(df)} rows × {len(df.columns)} cols, "
          f"positive_rate={df['target'].mean():.3f})")


if __name__ == "__main__":
    main()
