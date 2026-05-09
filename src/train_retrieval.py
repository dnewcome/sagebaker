"""Training harness for retrieval plugins (semantic search / FAISS).

Different shape from the supervised trainer:
  - No train/test split (we're indexing a corpus, not learning a function).
  - "Training" = compute corpus embeddings, build the FAISS index,
    persist alongside the metadata.
  - Bundle layout extends the standard one:

        config.json        framework=faiss, plugin=<name>, embedder_model=…
        index.faiss        binary FAISS index (IndexFlatL2 by default)
        corpus.parquet     metadata for every indexed item, in the same
                           order the embeddings were added to the index
        metadata.json      saved_at, n_items, dataset_file, lineage

`model_fn(model_dir)` returns a `_RetrievalModel` that exposes
`.predict(queries)` (list of texts → list of top-K dicts) so the
existing scoring server / DLC / MLflow pyfunc paths keep working
without per-framework adapters.

This is a sketch — not yet wired into plugins/__init__.py or the
Makefile. Verify the shape, then register and add `make train-search`.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import bundle
import tracking

sys.path.insert(0, os.path.dirname(__file__))

# Retrieval plugins live in their own registry — they don't share the
# supervised TrainingPlugin contract, so the existing get_plugin()
# wouldn't work for them. The simplest registry: one dict, one lookup.
from plugins.product_search import ProductSearchPlugin  # noqa: E402

_RETRIEVAL_REGISTRY = {
    ProductSearchPlugin.name: ProductSearchPlugin,
}


def get_retrieval_plugin(name: str):
    if name not in _RETRIEVAL_REGISTRY:
        raise ValueError(f"unknown retrieval plugin {name!r}; "
                         f"available: {sorted(_RETRIEVAL_REGISTRY)}")
    return _RETRIEVAL_REGISTRY[name]()


# ---------- training (= corpus indexing) ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True,
                        help="dir containing the corpus CSV/parquet")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--plugin", default="product_search")
    parser.add_argument("--top-k", type=int, default=5,
                        help="default k recorded in config.json — callers can override")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.train, "*.csv")) +
                   glob.glob(os.path.join(args.train, "*.parquet")))
    preferred = [f for f in files if os.path.basename(f).startswith("training.")]
    path = preferred[0] if preferred else files[0]
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    print(f"loaded {path}: {len(df)} rows")

    plugin = get_retrieval_plugin(args.plugin)
    items = plugin.prepare_corpus(df)
    texts = [it.text for it in items]
    metadata_rows = [it.metadata for it in items]
    print(f"corpus: {len(items)} items")

    # 1. Build the embedder (one-time cost) and embed all items in one
    # batch. This is the part the user's prior implementation got
    # wrong — embed-per-request is what made it slow.
    embedder = plugin.build_embedder()
    print(f"embedder: {plugin.embedder_model}")

    run_params = {"plugin": plugin.name, "embedder_model": plugin.embedder_model,
                  "n_items": len(items)}
    with tracking.mlflow_run(run_name=f"{plugin.name}-index", params=run_params,
                             tags={"framework": "faiss", "plugin": plugin.name}):
        embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
        print(f"embeddings: {embeddings.shape}")

        # 2. Build the index — IndexFlatL2 by default; plugin can override
        # for larger corpora (IVF, HNSW, etc.).
        index = plugin.build_index(embeddings)
        print(f"index: {type(index).__name__} with {index.ntotal} vectors")

        # --- write the bundle ---
        os.makedirs(args.model_dir, exist_ok=True)
        import faiss
        faiss.write_index(index, os.path.join(args.model_dir, "index.faiss"))
        pd.DataFrame(metadata_rows).to_parquet(
            os.path.join(args.model_dir, "corpus.parquet"), index=False
        )

        config = {
            "plugin": plugin.name,
            "framework": "faiss",
            "embedder_model": plugin.embedder_model,
            "embedding_dim": int(embeddings.shape[1]),
            "n_items": len(items),
            "default_top_k": args.top_k,
            "task": "retrieval",
        }
        bundle.save_config(args.model_dir, config)

        extras = {"n_items": len(items), "dataset_file": os.path.basename(path)}
        lineage = bundle.load_lineage(args.train)
        if lineage:
            extras["data_lineage"] = lineage
        bundle.save_metadata(args.model_dir, extras=extras)

        tracking.log_metrics({"n_indexed": len(items)})
        tracking.log_bundle(args.model_dir)


# ---------- inference ----------------------------------------------

class _RetrievalModel:
    """Inference wrapper. Exposes `.predict(queries)` so it slots into
    the existing serving paths (DLC, MLflow PyFunc) without adapters."""
    def __init__(self, embedder, index, metadata_rows, plugin, default_k):
        self._embedder = embedder
        self._index = index
        self._metadata = metadata_rows
        self._plugin = plugin
        self._default_k = default_k

    def predict(self, model_input, k: int | None = None):
        # Accept a list[str], a 1-element list (single query), or a
        # DataFrame with a `text` or `query` column (MLflow scoring
        # server batches things this way).
        if isinstance(model_input, pd.DataFrame):
            col = "query" if "query" in model_input.columns else "text"
            texts = model_input[col].tolist()
        elif isinstance(model_input, str):
            texts = [model_input]
        else:
            texts = list(model_input)
        return self._plugin.query(
            self._embedder, self._index, self._metadata,
            texts, k=k or self._default_k,
        )


def model_fn(model_dir):
    """Load the bundle: embedder + index + corpus metadata. Called once
    at container startup (DLC, MLflow scoring server). Per-request cost
    is just one embedder forward pass + FAISS lookup."""
    import faiss

    config = bundle.load_config(model_dir)
    plugin = get_retrieval_plugin(config["plugin"])
    plugin.embedder_model = config.get("embedder_model", plugin.embedder_model)

    embedder = plugin.build_embedder()
    index = faiss.read_index(os.path.join(model_dir, "index.faiss"))
    metadata = pd.read_parquet(os.path.join(model_dir, "corpus.parquet")).to_dict(orient="records")

    return _RetrievalModel(
        embedder=embedder, index=index, metadata_rows=metadata,
        plugin=plugin, default_k=config.get("default_top_k", 5),
    )


if __name__ == "__main__":
    main()
