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
import pandas as pd

from .base_retrieval import CorpusItem, RetrievalPlugin


_DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


class ProductSearchPlugin(RetrievalPlugin):
    name = "product_search"
    embedder_model: str = _DEFAULT_EMBEDDER

    # Columns whose values get concatenated into the embedded text (in order).
    # Override in a subclass if you have a richer text column.
    text_columns: tuple[str, ...] = ("title", "description")

    # Columns NOT to pass through as metadata (typically because they ARE
    # the embedded text, or are too large to include in every query result).
    skip_metadata_columns: tuple[str, ...] = ("description",)

    # Columns that MUST be in the metadata if present — the primary key
    # for the catalog. Tries each in order, picks the first that exists.
    id_columns: tuple[str, ...] = ("product_id", "catalog_id", "sku")

    def prepare_corpus(self, df) -> list[CorpusItem]:
        # Build the text from configured columns, joined by ". ".
        text_cols = [c for c in self.text_columns if c in df.columns]
        items = []
        for _, row in df.iterrows():
            text_parts = [str(row[c]).strip() for c in text_cols
                          if pd.notna(row[c]) and str(row[c]).strip()]
            text = ". ".join(text_parts)
            # Pass through every column except text-only ones and NaNs as metadata.
            metadata = {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in row.items()
                if k not in self.skip_metadata_columns and pd.notna(v)
            }
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
