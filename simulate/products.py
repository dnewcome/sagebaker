"""Canonical-product population + retailer primitives for product_catalog.

A synthetic product catalog has two layers:

1. **Canonical products** — the "real" thing in the world (e.g. Apple
   iPhone 15 Pro 256GB Space Black). Each has a brand, a base title,
   a category path, a base price. Ground truth.

2. **Retailer offerings** — what each retailer publishes on their
   site. Same canonical product appears multiple times, once per
   retailer, with a different SKU (Amazon ASIN, Walmart WM-id, etc.),
   sometimes-or-sometimes-not a GTIN, a slightly varied title, and
   a price with retailer-specific markup. The model only sees the
   retailer offering view; the canonical mapping is hidden in
   ground truth.

This gives you three distinct ML problems on the same dataset:

  - **Product matching / deduplication**: cross-retailer pair
    classification — given two offerings with different SKUs, are
    they the same canonical product? GTIN match is a strong signal
    when both retailers provide it; semantic title similarity does
    the rest.
  - **Category classification**: title → category path.
  - **Price-anomaly detection**: given (title, category, retailer),
    is this price unusual? (Some retailers' markup is wider than
    others — eBay-style.)

TODO
----
- More categories + product templates (currently a small fixed set
  for sketch purposes; could load from a JSON corpus).
- Variant-level products (size, color, capacity) generating multiple
  canonical SKUs per "model".
- Time-varying prices, stockouts.
- Description text (longer-form, useful for embeddings).
"""
import dataclasses
import random
from typing import Optional


# ---------- canonical product templates ----------------------------

# Small fixed set — enough to demo. Each tuple is
# (brand, base_title_template, variants, base_price_range).
# Variants get appended to the base title to produce the canonical
# title; e.g. ("Apple", "iPhone 15 Pro", ["256GB", "512GB"], (999, 1199))
# generates "Apple iPhone 15 Pro 256GB" with base_price ~999.
PRODUCT_TEMPLATES: dict[str, list[tuple[str, str, list[str], tuple[float, float]]]] = {
    "Electronics > Phones > Smartphones": [
        ("Apple", "iPhone 15 Pro", ["128GB", "256GB", "512GB", "1TB"], (999, 1499)),
        ("Apple", "iPhone 15", ["128GB", "256GB", "512GB"], (799, 999)),
        ("Samsung", "Galaxy S24 Ultra", ["256GB", "512GB", "1TB"], (1199, 1659)),
        ("Samsung", "Galaxy A55", ["128GB", "256GB"], (449, 549)),
        ("Google", "Pixel 8 Pro", ["128GB", "256GB", "512GB"], (999, 1199)),
    ],
    "Electronics > Audio > Headphones": [
        ("Sony", "WH-1000XM5 Wireless Headphones", ["Black", "Silver"], (399, 449)),
        ("Bose", "QuietComfort 45", ["Black", "White Smoke"], (329, 379)),
        ("Apple", "AirPods Pro 2nd Gen", ["USB-C"], (249, 249)),
    ],
    "Apparel > Mens > Shirts": [
        ("Nike", "Dri-FIT Cotton T-Shirt", ["S", "M", "L", "XL"], (25, 35)),
        ("Adidas", "Essentials 3-Stripes Tee", ["S", "M", "L", "XL"], (22, 30)),
        ("Patagonia", "Capilene Cool Daily Shirt", ["S", "M", "L"], (39, 49)),
    ],
    "Home > Kitchen > Appliances": [
        ("KitchenAid", "Artisan Stand Mixer 5Qt", ["Empire Red", "Onyx Black"], (379, 449)),
        ("Vitamix", "5200 Standard Blender", ["Black"], (449, 549)),
        ("Instant Pot", "Duo Plus 6Qt Pressure Cooker", [], (89, 119)),
    ],
}


@dataclasses.dataclass
class CanonicalProduct:
    canonical_id: int
    brand: str
    base_title: str          # "Apple iPhone 15 Pro 256GB" (canonical form)
    category_path: str       # "Electronics > Phones > Smartphones"
    base_price: float
    gtin: str                # 13-digit canonical product code (always exists in reality)


def _fake_gtin(rng: random.Random) -> str:
    """13-digit numeric string — looks like a real GTIN/EAN-13."""
    return "".join(str(rng.randint(0, 9)) for _ in range(13))


def make_canonical_products(rng: random.Random, target_n: int = 50) -> list[CanonicalProduct]:
    """Expand the templates into ~target_n canonical products by sampling
    variants. Repeated calls with the same rng give deterministic output."""
    products: list[CanonicalProduct] = []
    canonical_id = 0
    while len(products) < target_n:
        for category, templates in PRODUCT_TEMPLATES.items():
            for brand, base, variants, price_range in templates:
                if len(products) >= target_n:
                    break
                variant = rng.choice(variants) if variants else ""
                title = f"{brand} {base} {variant}".strip()
                price = round(rng.uniform(*price_range), 2)
                products.append(CanonicalProduct(
                    canonical_id=canonical_id,
                    brand=brand,
                    base_title=title,
                    category_path=category,
                    base_price=price,
                    gtin=_fake_gtin(rng),
                ))
                canonical_id += 1
    return products


# ---------- retailer configs ----------------------------------------

@dataclasses.dataclass
class Retailer:
    name: str
    sku_prefix: str
    sku_format: str               # "alphanumeric" or "numeric"
    gtin_provision_rate: float    # P(retailer publishes GTIN for a given product)
    title_noise_rate: float       # P(this retailer's title differs from canonical)
    price_markup_range: tuple[float, float]
    # Loyalty / member pricing (Sam's Club, Costco, BJ's, Amazon Prime
    # promo). Nonzero discount applied to a fraction of items if member.
    member_pricing: bool = False
    member_discount_range: tuple[float, float] = (0.05, 0.15)
    # Out-of-stock / availability semantics. Different retailers expose
    # this differently — Amazon has "reorder date", eBay has "limited
    # quantity," Best Buy has "ship to home" vs "in store only" etc.
    # We pick a single per-retailer rate for sketch purposes.
    out_of_stock_rate: float = 0.05
    # Whether this retailer's API exposes a forward-looking "reorder
    # date" (Amazon-style) when out of stock — vs just "out of stock".
    exposes_reorder_date: bool = False


# Tuned to roughly match real-world behavior the user described.
DEFAULT_RETAILERS: list[Retailer] = [
    # Amazon: nearly always has GTIN, ASIN-shaped SKUs, modest title rewriting.
    # Exposes reorder dates when OOS.
    Retailer("amazon", "B0", "alphanumeric", 0.95, 0.40, (0.95, 1.05),
             out_of_stock_rate=0.06, exposes_reorder_date=True),
    # Walmart: usually GTIN, numeric SKU, lighter title rewriting.
    Retailer("walmart", "WM-", "numeric", 0.85, 0.25, (0.93, 1.10),
             out_of_stock_rate=0.04),
    # Target: similar to Walmart.
    Retailer("target", "TGT-", "numeric", 0.80, 0.25, (0.95, 1.08),
             out_of_stock_rate=0.05),
    # Best Buy: high GTIN coverage on electronics, low rewriting.
    Retailer("bestbuy", "BBY-", "numeric", 0.90, 0.20, (0.97, 1.05),
             out_of_stock_rate=0.07),
    # eBay: third-party sellers — frequently no GTIN, heavy title noise,
    # wide price variance. The hard case for product matching.
    Retailer("ebay", "EB", "alphanumeric", 0.40, 0.65, (0.80, 1.30),
             out_of_stock_rate=0.10),
    # Sam's Club: member pricing, mid GTIN coverage, conservative titles.
    # Member discount typically 5–15% off list.
    Retailer("samsclub", "SC-", "numeric", 0.75, 0.20, (0.95, 1.10),
             member_pricing=True, member_discount_range=(0.05, 0.15),
             out_of_stock_rate=0.05),
]


def _make_sku(rng: random.Random, retailer: Retailer) -> str:
    if retailer.sku_format == "alphanumeric":
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZ123456789"
        body = "".join(rng.choice(chars) for _ in range(8))
    else:
        body = "".join(str(rng.randint(0, 9)) for _ in range(8))
    return retailer.sku_prefix + body


# ---------- title noise --------------------------------------------

_MARKETING_FLUFF = [
    "Brand New", "Latest Model", "Free Shipping", "Authentic", "Sealed",
    "Genuine", "Original", "OEM", "Bulk", "Factory Refurbished",
]


def _noise_title(rng: random.Random, canonical_title: str) -> str:
    """Apply 1-3 noise transformations to make this retailer's listing
    visually different from the canonical title without changing meaning.
    Tests of a similarity model should reward catching these as duplicates."""
    title = canonical_title
    n_transforms = rng.randint(1, 3)
    transforms = list(range(5))
    rng.shuffle(transforms)

    for t in transforms[:n_transforms]:
        if t == 0:
            # Reorder: put brand at the end with a separator.
            parts = title.split(" ", 1)
            if len(parts) == 2:
                title = f"{parts[1]} - {parts[0]}"
        elif t == 1:
            # Add marketing fluff suffix.
            title = f"{title}, {rng.choice(_MARKETING_FLUFF)}"
        elif t == 2:
            # Spacing on capacity (e.g. "256GB" → "256 GB", or vice versa).
            title = title.replace("GB", " GB") if "GB " not in title else title.replace(" GB", "GB")
        elif t == 3:
            # Title-case → mostly lowercase (eBay-style).
            title = title.lower()
        elif t == 4:
            # Add a generic descriptor suffix.
            title = f"{title} - Unlocked"
    return title


def make_offering(rng: random.Random, canonical: CanonicalProduct,
                  retailer: Retailer, snapshot_date) -> dict:
    """One row in the catalog: this retailer's offering of the canonical product.

    Realistic-mess fields the user flagged as worth simulating:
      - member_price: Sam's Club / Costco-style loyalty pricing. Null
        when the retailer doesn't have a member program; null with
        some probability even when they do (this product not on
        member promo). Otherwise list_price * (1 - discount).
      - availability: in_stock / out_of_stock. Per-retailer base rate.
      - reorder_date: Amazon-style "back in stock by …" date. Only
        populated for retailers that expose it AND when out_of_stock.
        Other OOS retailers just show out_of_stock with no date.
    """
    from datetime import timedelta
    title = (
        _noise_title(rng, canonical.base_title)
        if rng.random() < retailer.title_noise_rate
        else canonical.base_title
    )
    has_gtin = rng.random() < retailer.gtin_provision_rate
    markup = rng.uniform(*retailer.price_markup_range)
    list_price = round(canonical.base_price * markup, 2)

    # Member pricing: only some retailers, only some products on promo.
    member_price = None
    if retailer.member_pricing and rng.random() < 0.7:
        discount = rng.uniform(*retailer.member_discount_range)
        member_price = round(list_price * (1.0 - discount), 2)

    # Availability + reorder date.
    is_oos = rng.random() < retailer.out_of_stock_rate
    availability = "out_of_stock" if is_oos else "in_stock"
    reorder_date = None
    if is_oos and retailer.exposes_reorder_date:
        # Amazon-style: 3 to 30 days out, randomly.
        reorder_date = (snapshot_date + timedelta(days=rng.randint(3, 30))).date().isoformat()

    return {
        "retailer": retailer.name,
        "sku": _make_sku(rng, retailer),
        "gtin": canonical.gtin if has_gtin else None,
        "title": title,
        "category": canonical.category_path,
        "list_price": list_price,
        "member_price": member_price,
        "availability": availability,
        "reorder_date": reorder_date,
    }
