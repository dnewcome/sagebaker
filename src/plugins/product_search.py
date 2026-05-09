"""Product-search retrieval plugin: semantic similarity over product titles.

Demonstrates the RetrievalPlugin contract using a HuggingFace
sentence-transformer for embeddings and FAISS for ANN search. Target
shape for the user's day-job problem (titles → embeddings → top-K
similar products), without the "one product at a time" anti-pattern.

Input data: parquet/csv with at least `title` and `product_id` columns;
optional `category`, `price`, etc. get carried through into query
results via metadata.

Embedder: sentence-transformers/all-MiniLM-L6-v2 by default
(80MB, 384-dim, fast on CPU). Tunable via the
`embedder_model` config. For a tighter latency budget swap to a
distilled MiniLM-3 layer; for higher quality swap to mpnet-base.
"""
from .base_retrieval import CorpusItem, RetrievalPlugin


_DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


class ProductSearchPlugin(RetrievalPlugin):
    name = "product_search"
    embedder_model: str = _DEFAULT_EMBEDDER

    def prepare_corpus(self, df) -> list[CorpusItem]:
        items = []
        for _, row in df.iterrows():
            text = str(row.get("title", "")).strip()
            if "description" in row and row["description"] is not None:
                text = f"{text}. {row['description']}"
            metadata = {
                "product_id": row["product_id"],
                "title": row.get("title"),
                "category": row.get("category"),
            }
            # Drop None/NaN values so JSON output is clean at query time.
            metadata = {k: v for k, v in metadata.items() if v is not None}
            items.append(CorpusItem(text=text, metadata=metadata))
        return items

    def build_embedder(self):
        # SentenceTransformer caches the model under ~/.cache/huggingface
        # after first download. For offline / production-portable
        # bundles we'd vendor the model weights into the bundle dir
        # via safetensors instead of relying on HF download — that's
        # a follow-on once this works.
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.embedder_model)
