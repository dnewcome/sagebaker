"""product_catalog: synthetic multi-retailer product catalog with messy IDs.

Generates a corpus where the same canonical product appears on multiple
retailers with different SKUs, sometimes-but-not-always GTINs, slightly
varied titles, retailer-specific pricing (including member pricing on
warehouse-club retailers like Sam's Club), and stock-state fields
(availability, optional reorder date for retailers that expose it).

The training frame is the "what the model sees" view; the ground truth
parquet carries the canonical mapping (`true_canonical_id`,
`true_brand`, `true_base_title`) so evaluation can score how well a
product-matching model recovers identity from messy retailer offerings.

This single dataset supports several distinct ML problems:

  1. **Product matching / canonicalization** — given two retailer
     offerings, are they the same canonical product? GTIN matches when
     both retailers provide one; title-similarity does the rest. The
     `retrieval` plugin family is a natural recall stage; a binary
     pair-classifier is a natural rank stage.
  2. **Category classification** — title → category path.
  3. **Price-anomaly detection** — given (title, category, retailer,
     availability), is this price/member-price unusual? eBay-style
     wide markup + occasional missing GTIN is the natural source of
     anomalies in the simulator.

Realism knobs (CLI flags or scenario params):
  - n_canonical_products      how many distinct "true" products
  - retailers                  defaults to DEFAULT_RETAILERS in
                              simulate/products.py
  - oos_multiplier            scale all retailers' OOS rates (default 1.0)

TODO
----
- Variants per canonical (size/color/capacity) generating multiple
  matched canonicals.
- Time-series prices (one row per snapshot per retailer per product)
  for tracking price changes / promos.
- Coupon / promo metadata as a separate field.
- Description text (longer, useful for embeddings).
- Cross-retailer brand normalization noise ("Apple" vs "APPLE INC").
"""
import random
from datetime import datetime, timezone

import pandas as pd

from ..base import Scenario, SimulationResult
from ..products import DEFAULT_RETAILERS, make_canonical_products, make_offering


class ProductCatalogScenario(Scenario):
    name = "product_catalog"
    description = (
        "Multi-retailer product catalog with retailer-specific SKUs, "
        "sometimes-missing GTINs, title noise, member pricing, and "
        "stock state. Supports product-matching, category-classification, "
        "and price-anomaly problems on the same dataset."
    )
    default_params = {
        "n_canonical_products": 60,
        "oos_multiplier": 1.0,
    }

    def generate(self, seed: int = 42, **params) -> SimulationResult:
        p = {**self.default_params, **params}
        rng = random.Random(seed)
        snapshot_date = datetime(2026, 5, 9, tzinfo=timezone.utc)

        canonicals = make_canonical_products(rng, target_n=int(p["n_canonical_products"]))
        retailers = list(DEFAULT_RETAILERS)
        oos_scale = float(p["oos_multiplier"])

        offerings: list[dict] = []
        ground_truth: list[dict] = []
        catalog_id = 0

        for canonical in canonicals:
            for retailer in retailers:
                # Apply OOS multiplier (clamped to [0, 1]).
                scaled = type(retailer)(
                    name=retailer.name, sku_prefix=retailer.sku_prefix,
                    sku_format=retailer.sku_format,
                    gtin_provision_rate=retailer.gtin_provision_rate,
                    title_noise_rate=retailer.title_noise_rate,
                    price_markup_range=retailer.price_markup_range,
                    member_pricing=retailer.member_pricing,
                    member_discount_range=retailer.member_discount_range,
                    out_of_stock_rate=min(1.0, retailer.out_of_stock_rate * oos_scale),
                    exposes_reorder_date=retailer.exposes_reorder_date,
                )
                offering = make_offering(rng, canonical, scaled, snapshot_date)
                offering["catalog_id"] = catalog_id
                offerings.append(offering)
                ground_truth.append({
                    "catalog_id": catalog_id,
                    "true_canonical_id": canonical.canonical_id,
                    "true_brand": canonical.brand,
                    "true_base_title": canonical.base_title,
                    "true_category_path": canonical.category_path,
                    "true_base_price": canonical.base_price,
                    "true_gtin": canonical.gtin,
                })
                catalog_id += 1

        # Reorder columns so catalog_id is first (consistent with how
        # other scenarios put their primary key first).
        train_cols = ["catalog_id", "retailer", "sku", "gtin", "title",
                      "category", "list_price", "member_price",
                      "availability", "reorder_date"]
        training_df = pd.DataFrame(offerings)[train_cols]
        gt_df = pd.DataFrame(ground_truth)

        # Light shuffle so training doesn't have all-Amazon rows then
        # all-Walmart rows etc. Seeded for reproducibility.
        training_df = training_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        gt_df = gt_df.set_index("catalog_id").loc[training_df["catalog_id"]].reset_index()

        lineage = self.make_lineage(self.name, seed, p, training_df)
        return SimulationResult(training=training_df, ground_truth=gt_df, lineage=lineage)
