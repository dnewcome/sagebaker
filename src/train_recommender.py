"""Training harness for RecommenderPlugin — collaborative filtering models.

SageMaker path conventions are identical to train.py:

    /opt/ml/input/data/train/   input CSV or parquet
    /opt/ml/input/config/hyperparameters.json
    /opt/ml/model/              bundle written here

Bundle layout
-------------
    model/
    ├── config.json             plugin, framework, entity_col, item_col, K
    ├── model.npz               numpy archive: user_factors, item_factors
    ├── entity_index.parquet    entity_id → entity_idx  (needed at inference)
    ├── item_index.parquet      item_idx  → item_id     (needed at inference)
    └── metadata.json           git sha, ranking metrics, data lineage

Inference contract
------------------
``model_fn(model_dir)`` returns a ``RecommenderBundle`` with:

    bundle.recommend(entity_id: str, K: int = 10) -> list[str]

Inference has no dependency on the training library (implicit, etc.).
The factor matrices are plain numpy arrays; recommendation is a dot product.

Hyperparameters (all string-valued in SageMaker hp.json)
---------------------------------------------------------
    plugin              recommender plugin name (default: clh)
    K                   number of recommendations to report (default: 10)
    als_factors         ALS latent factors (default: 32)
    als_iterations      ALS training iterations (default: 10)
    als_regularization  ALS L2 regularisation (default: 0.01)
    als_alpha           confidence scaling for implicit feedback (default: 20.0)
    max_eval_entities   cap on evaluation sample size (default: 2000)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import bundle
import tracking

sys.path.insert(0, os.path.dirname(__file__))
from plugins import get_recommender_plugin, list_recommender_plugins

HP_PATH = "/opt/ml/input/config/hyperparameters.json"


def load_hyperparameters():
    if not os.path.exists(HP_PATH):
        return {}
    with open(HP_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ranking evaluation  (Recall@K, Precision@K, NDCG@K, MAP@K, HitRate@K)
# ---------------------------------------------------------------------------

def _dcg_at_k(rel, k):
    rel = np.asarray(rel, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    return float((rel / np.log2(np.arange(2, rel.size + 2))).sum())


def evaluate(model, data, K, max_entities=2000, seed=0):
    """Compute standard ranking metrics over the held-out test set.

    Uses the plugin model's ``recommend(entity_idx, K, exclude)`` method,
    which the harness can call without knowing which library backs the model.

    Parameters
    ----------
    model       fitted model wrapper (ALSWrapper or similar)
    data        InteractionData with test_items and train_items
    K           cut-off for all metrics
    max_entities cap on evaluation sample (full set can be slow)
    seed        random seed for the evaluation subsample
    """
    entity_ids = list(data.test_items.keys())
    if len(entity_ids) > max_entities:
        rng = np.random.RandomState(seed)
        entity_ids = list(rng.choice(entity_ids, size=max_entities, replace=False))

    recalls, precisions, hits, ndcgs, maps = [], [], [], [], []

    for eid in entity_ids:
        truth = data.test_items[eid]
        seen = data.train_items.get(eid, set())
        recs = model.recommend(eid, K, exclude=seen)

        if not recs:
            recalls.append(0.0); precisions.append(0.0); hits.append(0.0)
            ndcgs.append(0.0); maps.append(0.0)
            continue

        rel = [1 if r in truth else 0 for r in recs]
        n_rel = len(truth)

        recalls.append(sum(rel) / n_rel if n_rel else 0.0)
        precisions.append(sum(rel) / K)
        hits.append(1.0 if sum(rel) > 0 else 0.0)

        ideal = _dcg_at_k([1] * min(n_rel, K), K)
        ndcgs.append(_dcg_at_k(rel, K) / ideal if ideal else 0.0)

        n_hits, ap = 0, 0.0
        for i, r in enumerate(rel, 1):
            if r:
                n_hits += 1
                ap += n_hits / i
        maps.append(ap / min(n_rel, K) if n_rel else 0.0)

    return {
        f"recall_at_{K}":    float(np.mean(recalls)),
        f"precision_at_{K}": float(np.mean(precisions)),
        f"hit_rate_at_{K}":  float(np.mean(hits)),
        f"ndcg_at_{K}":      float(np.mean(ndcgs)),
        f"map_at_{K}":       float(np.mean(maps)),
        "n_entities_evaluated": len(entity_ids),
    }


# ---------------------------------------------------------------------------
# Inference contract
# ---------------------------------------------------------------------------

class RecommenderBundle:
    """Loaded bundle ready for inference. Returned by model_fn(model_dir).

    Pure numpy — no dependency on implicit or any training library.
    Recommendation is a dot product between a stored entity embedding and
    all item embeddings, followed by argsort.

    This is the object an inference service (or any caller) holds after
    loading a recommender bundle. It owns the entity→idx lookup, the
    item←idx lookup, and the score computation.
    """

    def __init__(self, user_factors, item_factors, entity_map, item_map):
        self._uf = user_factors   # np.ndarray (n_entities, n_factors)
        self._if = item_factors   # np.ndarray (n_items,    n_factors)
        self._entity_map = entity_map  # str → int
        self._item_map = item_map      # int → str

    def recommend(self, entity_id: str, K: int = 10) -> list:
        """Return up to K item IDs for entity_id.

        Returns an empty list if entity_id was not seen at training time
        (cold-start entities are not handled here; the caller should decide
        on a fallback such as popularity-based recommendations).
        """
        idx = self._entity_map.get(str(entity_id))
        if idx is None:
            return []
        scores = self._if @ self._uf[idx]
        top_k = np.argsort(-scores)[:K]
        return [self._item_map[int(i)] for i in top_k if int(i) in self._item_map]

    @property
    def n_entities(self) -> int:
        return len(self._entity_map)

    @property
    def n_items(self) -> int:
        return len(self._item_map)


def model_fn(model_dir: str) -> RecommenderBundle:
    """Load a recommender bundle. SageMaker and local inference both call this.

    Returns a RecommenderBundle with .recommend(entity_id, K) -> list[item_id].
    Only numpy and pandas are required — no training library.
    """
    cfg = bundle.load_config(model_dir)

    factors = np.load(os.path.join(model_dir, cfg.get("weights_file", "model.npz")))
    user_factors = factors["user_factors"]
    item_factors = factors["item_factors"]

    entity_df = pd.read_parquet(
        os.path.join(model_dir, cfg.get("entity_index_file", "entity_index.parquet"))
    )
    item_df = pd.read_parquet(
        os.path.join(model_dir, cfg.get("item_index_file", "item_index.parquet"))
    )

    entity_map = dict(zip(
        entity_df["entity_id"].astype(str),
        entity_df["entity_idx"].astype(int),
    ))
    item_map = dict(zip(
        item_df["item_idx"].astype(int),
        item_df["item_id"].astype(str),
    ))

    return RecommenderBundle(user_factors, item_factors, entity_map, item_map)


# ---------------------------------------------------------------------------
# Training harness
# ---------------------------------------------------------------------------

def main():
    hp = load_hyperparameters()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--plugin", type=str,
                        default=hp.get("plugin", "clh"),
                        help=f"recommender plugin. available: {list_recommender_plugins()}")
    parser.add_argument("--K", type=int,
                        default=int(hp.get("K", 10)),
                        help="recommendation cut-off for evaluation metrics")
    parser.add_argument("--max-eval-entities", type=int,
                        default=int(hp.get("max_eval_entities", 2000)),
                        help="max entities to evaluate (capped for speed)")
    args, _ = parser.parse_known_args()

    plugin = get_recommender_plugin(args.plugin)
    print(f"plugin:  {plugin.name}")
    print(f"K:       {args.K}")

    # Unified params dict passed to plugin.build_model().
    params = {k.replace("-", "_"): str(v) for k, v in hp.items()}
    params.setdefault("K", str(args.K))

    os.makedirs(args.model_dir, exist_ok=True)

    # --- load data ----------------------------------------------------------
    files = sorted(
        glob.glob(os.path.join(args.train, "*.csv"))
        + glob.glob(os.path.join(args.train, "*.parquet"))
    )
    if not files:
        raise SystemExit(f"no .csv or .parquet found in {args.train}")
    path = files[0]
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    print(f"loaded {path}: {len(df):,} rows, {len(df.columns)} columns")

    # --- plugin: build interaction matrix + split ---------------------------
    print("preparing interaction matrix...")
    data = plugin.prepare(df)

    # --- build + fit model --------------------------------------------------
    model = plugin.build_model(params)

    run_params = {
        **params,
        "plugin": plugin.name,
        "K": str(args.K),
        "dataset_file": os.path.basename(path),
        "n_entities": str(data.n_entities),
        "n_items": str(data.n_items),
    }

    with tracking.mlflow_run(
        run_name=f"{plugin.name}-train",
        params=run_params,
        tags={"framework": "implicit-als", "plugin": plugin.name},
    ):
        print("fitting model...")
        model.fit(data.train_mat)

        # --- evaluate -------------------------------------------------------
        print(f"evaluating at K={args.K} "
              f"(max {args.max_eval_entities} entities)...")
        metrics = evaluate(model, data, K=args.K,
                           max_entities=args.max_eval_entities)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        tracking.log_metrics(metrics)

        # --- save bundle ----------------------------------------------------
        # 1. Factor matrices — raw numpy, no training lib needed at inference.
        weights_file = "model.npz"
        np.savez(
            os.path.join(args.model_dir, weights_file),
            user_factors=model.user_factors,
            item_factors=model.item_factors,
        )

        # 2. Index tables — entity_id↔idx and item_idx↔item_id.
        entity_index_file = "entity_index.parquet"
        item_index_file = "item_index.parquet"
        data.entity_index.to_parquet(
            os.path.join(args.model_dir, entity_index_file), index=False
        )
        data.item_index.to_parquet(
            os.path.join(args.model_dir, item_index_file), index=False
        )

        # 3. config.json
        cfg = {
            "plugin": plugin.name,
            "framework": "implicit-als",
            "weights_file": weights_file,
            "entity_index_file": entity_index_file,
            "item_index_file": item_index_file,
            "entity_col": plugin.entity_col,
            "item_col": plugin.item_col,
            "K": args.K,
        }
        cfg.update(plugin.extra_config(model, data))
        bundle.save_config(args.model_dir, cfg)

        # 4. metadata.json
        extras = {**metrics, "dataset_file": os.path.basename(path)}
        lineage = bundle.load_lineage(args.train)
        if lineage:
            extras["data_lineage"] = lineage
        bundle.save_metadata(args.model_dir, extras=extras)

        tracking.log_bundle(args.model_dir)


if __name__ == "__main__":
    main()
