"""Generic ALS (Alternating Least Squares) collaborative filtering plugin.

Algorithm
---------
Matrix factorization via the ``implicit`` library. The sparse user×item
interaction matrix is factorized into two embedding tables; recommendations
are the K highest dot-product scores between the user embedding and all item
embeddings.

Inference is pure numpy — no ``implicit`` dependency at serving time.

Expected training schema
------------------------
user_id     string user identifier
item_id     string item identifier
weight      numeric interaction strength (e.g. rating, play count, clicks)

For local development, ``prepare.py --plugin als`` generates synthetic data
with latent cluster structure so the model has real signal to learn.

Hyperparameters
---------------
als_factors         latent factor dimension  (default: 32)
als_iterations      training iterations      (default: 10)
als_regularization  L2 regularisation        (default: 0.01)
als_alpha           confidence scaling       (default: 20.0)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base_recommender import InteractionData, RecommenderPlugin

try:
    from implicit.als import AlternatingLeastSquares
    _HAS_IMPLICIT = True
except ImportError:
    _HAS_IMPLICIT = False

_NEEDED = ["user_id", "item_id", "weight"]

MIN_ITEM_COUNT = 3    # minimum unique users per item
MIN_ENTITY_SIZE = 3   # minimum distinct items per user
HOLDOUT_FRAC = 0.20
MIN_HOLDOUT_SIZE = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_interactions(df, min_item_count, min_entity_size, max_passes=5):
    for _ in range(max_passes):
        n_before = len(df)
        item_ok = df.groupby("item_id")["user_id"].nunique()
        df = df[df["item_id"].isin(item_ok[item_ok >= min_item_count].index)]
        user_ok = df.groupby("user_id")["item_id"].nunique()
        df = df[df["user_id"].isin(user_ok[user_ok >= min_entity_size].index)]
        if len(df) == n_before:
            break
    return df.copy()


def _leave_k_out_split(df, holdout_frac=0.20, min_size=3, seed=42):
    rng = np.random.RandomState(seed)
    train_rows, test_rows = [], []
    for _, g in df.groupby("user_idx"):
        n = len(g)
        if n < min_size:
            train_rows.append(g)
            continue
        n_test = max(1, int(round(n * holdout_frac)))
        perm = rng.permutation(n)
        train_rows.append(g.iloc[perm[n_test:]])
        test_rows.append(g.iloc[perm[:n_test]])
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = (
        pd.concat(test_rows, ignore_index=True)
        if test_rows
        else pd.DataFrame(columns=df.columns)
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model wrapper (also imported by private plugins that extend ALS)
# ---------------------------------------------------------------------------

class ALSWrapper:
    """Wraps implicit's ALS with the interface the harness expects.

    Recommendation uses a raw numpy dot product so inference has no
    dependency on the ``implicit`` package. The factor arrays stored in
    model.npz are the complete model state.
    """

    def __init__(self, factors=32, regularization=0.01, iterations=10, alpha=20.0):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None

    def fit(self, train_mat) -> "ALSWrapper":
        if not _HAS_IMPLICIT:
            raise ImportError(
                "implicit is required for ALS training. "
                "Run: pip install --group recommender"
            )
        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False,
        )
        model.fit((train_mat * self.alpha).astype(np.float32))
        self._user_factors = np.array(model.user_factors)
        self._item_factors = np.array(model.item_factors)
        return self

    def recommend(self, entity_idx: int, K: int, exclude: set) -> list:
        scores = self._item_factors @ self._user_factors[entity_idx]
        for idx in exclude:
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf
        top = np.argsort(-scores)
        return [int(i) for i in top if int(i) not in exclude][:K]

    @property
    def user_factors(self) -> np.ndarray:
        return self._user_factors

    @property
    def item_factors(self) -> np.ndarray:
        return self._item_factors


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class ALSPlugin(RecommenderPlugin):
    name = "als"
    entity_col = "user_id"
    item_col = "item_id"

    def prepare(self, df: pd.DataFrame) -> InteractionData:
        missing = [c for c in _NEEDED if c not in df.columns]
        if missing:
            raise ValueError(f"ALS training data is missing columns: {missing}")

        df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last").copy()
        df = _filter_interactions(df, MIN_ITEM_COUNT, MIN_ENTITY_SIZE)
        if len(df) == 0:
            raise ValueError(
                "No interactions remain after filtering. "
                "Lower MIN_ITEM_COUNT / MIN_ENTITY_SIZE or generate more data."
            )

        users = df["user_id"].unique()
        items = df["item_id"].unique()
        user2idx = {u: i for i, u in enumerate(users)}
        item2idx = {a: i for i, a in enumerate(items)}

        entity_index = pd.DataFrame({
            "entity_id": users,
            "entity_idx": np.arange(len(users), dtype=np.int32),
        })
        item_index = pd.DataFrame({
            "item_idx": np.arange(len(items), dtype=np.int32),
            "item_id": items,
        })

        df = df.assign(
            user_idx=df["user_id"].map(user2idx),
            item_idx=df["item_id"].map(item2idx),
        )

        train_df, test_df = _leave_k_out_split(
            df, holdout_frac=HOLDOUT_FRAC, min_size=MIN_HOLDOUT_SIZE
        )

        values = train_df["weight"].values.astype(np.float32)
        n_entities, n_items = len(users), len(items)
        train_mat = sp.csr_matrix(
            (values, (train_df["user_idx"].values, train_df["item_idx"].values)),
            shape=(n_entities, n_items),
        )
        train_mat.eliminate_zeros()

        test_items = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

        print(f"  {n_entities:,} users × {n_items:,} items")
        print(f"  train: {len(train_df):,} interactions")
        print(f"  test:  {len(test_df):,} interactions over {len(test_items):,} users")
        print(f"  matrix density: {train_mat.nnz / (n_entities * n_items):.2e}")

        return InteractionData(
            train_mat=train_mat,
            test_items=test_items,
            train_items=train_items,
            entity_index=entity_index,
            item_index=item_index,
            n_entities=n_entities,
            n_items=n_items,
        )

    def build_model(self, params: dict) -> ALSWrapper:
        return ALSWrapper(
            factors=int(params.get("als_factors", 32)),
            regularization=float(params.get("als_regularization", 0.01)),
            iterations=int(params.get("als_iterations", 10)),
            alpha=float(params.get("als_alpha", 20.0)),
        )

    def extra_config(self, model: ALSWrapper, data: InteractionData) -> dict:
        return {
            "entity_col": self.entity_col,
            "item_col": self.item_col,
            "n_entities": data.n_entities,
            "n_items": data.n_items,
            "als_factors": model.factors,
            "als_iterations": model.iterations,
        }

    # ------------------------------------------------------------------
    # Synthetic data generation (local dev / CI)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate(
        n_users: int = 500,
        n_items: int = 2000,
        n_clusters: int = 8,
        avg_interactions: int = 25,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        user_cluster = rng.integers(0, n_clusters, size=n_users)
        item_cluster = rng.integers(0, n_clusters, size=n_items)
        user_ids = [f"user-{i:06d}" for i in range(n_users)]
        item_ids = [f"item-{i:06d}" for i in range(n_items)]

        rows = []
        for ui, user in enumerate(user_ids):
            n_interactions = rng.integers(
                max(1, avg_interactions // 2), avg_interactions * 2
            )
            own = [ii for ii, ic in enumerate(item_cluster) if ic == user_cluster[ui]]
            n_own = max(1, int(n_interactions * 0.7))
            n_rand = n_interactions - n_own
            chosen = []
            if own:
                chosen += list(rng.choice(own, size=min(n_own, len(own)), replace=False))
            rand = [ii for ii in range(n_items) if ii not in set(chosen)]
            if rand and n_rand > 0:
                chosen += list(rng.choice(rand, size=min(n_rand, len(rand)), replace=False))
            for ii in set(chosen):
                rows.append({
                    "user_id": user,
                    "item_id": item_ids[ii],
                    "weight": float(rng.integers(1, 10)),
                })
        return pd.DataFrame(rows)

    def prepare_data(self, output_dir: str, seed: int = 42, extra_args: list = None) -> None:
        parser = argparse.ArgumentParser(prog="prepare.py --plugin als")
        parser.add_argument("--users",    type=int, default=500,  help="number of synthetic users")
        parser.add_argument("--items",    type=int, default=2000, help="number of synthetic items")
        parser.add_argument("--clusters", type=int, default=8,    help="number of latent clusters")
        args = parser.parse_args(extra_args or [])

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        df = self._generate(
            n_users=args.users, n_items=args.items,
            n_clusters=args.clusters, seed=seed,
        )
        out_path = os.path.join(output_dir, "als.csv")
        df.to_csv(out_path, index=False)

        sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
        lineage = {
            "source": "synthetic",
            "generator": f"prepare.py --plugin {self.name}",
            "dataset_sha256": sha,
            "dataset_n_rows": len(df),
            "n_cols": len(df.columns),
            "schema": list(df.columns),
            "n_users": df["user_id"].nunique(),
            "n_items": df["item_id"].nunique(),
            "seed": seed,
        }
        with open(os.path.join(output_dir, "lineage.json"), "w") as f:
            json.dump(lineage, f, indent=2)

        print(f"wrote {out_path}")
        print(f"  {df['user_id'].nunique():,} users")
        print(f"  {df['item_id'].nunique():,} items")
        print(f"  {len(df):,} interactions")
        print(f"  density: {len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.3f}")
        print(f"wrote {output_dir}/lineage.json (sha256: {sha[:16]}...)")
