"""product_matcher plugin: predict same-canonical from cross-retailer pairs.

Trains on the pair-level dataset emitted by `prep/prepare_matcher_pairs.py`,
where each row is a (offering_a, offering_b) pair from the synthetic
product_catalog scenario, with binary target = 1 if both offerings are
listings of the same canonical product.

Why this is a useful problem to learn on
----------------------------------------
- **GTIN-as-feature trap.** Pairs where both retailers provide a GTIN
  AND the GTINs match are deterministically positive — a model that
  learned only `same_gtin` would hit very high accuracy on those rows
  but fail on the eBay-style pairs where one or both lack GTIN. Watch
  feature importances: a healthy model uses GTIN AND title-similarity.
- **Title-noise resilience.** title_jaccard / title_token_overlap
  catch the case where the same canonical product has its words
  reordered, brand position swapped, or title lowercased per retailer.
- **Same-retailer negatives.** Most negatives are cross-retailer
  (different products from different sources). Same-retailer negatives
  do appear (different products in the same retailer's catalog) and
  are interesting because retailer is then non-informative.

Bundle / config notes
---------------------
- Records `prediction_threshold` for downstream serving (0.5 default
  is fine for this balanced 1:1 sampled problem; tune per deployment
  if the production candidate-pair distribution is more skewed).

Limitations
-----------
- The current pair-feature set is shallow (token overlap, GTIN match,
  category match, price ratio). Adding cosine similarity between
  embedded titles (using the retrieval plugin's embedder) would likely
  push AUC up — TODO once the integration is worth the dep.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .base import TrainingPlugin


class ProductMatcherPlugin(TrainingPlugin):
    name = "product_matcher"
    task = "classification"

    def prepare(self, df: pd.DataFrame):
        y = df["target"].astype(int)
        X = df.drop(columns=["target"])
        return X, y

    def evaluate(self, model, X_test, y_true):
        proba = model.predict_proba(X_test)[:, 1]
        return "validation_auc", float(roc_auc_score(y_true, proba))

    def build_model(self, params: dict):
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 200)),
            max_depth=int(params.get("max_depth", 5)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            random_state=42,
        )
