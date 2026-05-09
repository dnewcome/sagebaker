"""Retrieval plugin contract — semantic similarity / nearest-neighbor search.

Different shape from supervised (TrainingPlugin) and collaborative
filtering (RecommenderPlugin):

  - Input: a corpus of texts (or other items to embed) + a free-text
    query at inference time.
  - Output: top-K most similar items from the corpus.
  - "Training" is really *indexing*: compute embeddings for the
    corpus, build a FAISS index, persist both alongside metadata.

Two layers in the artifact:

  1. **Embedder** — model that turns text → vector. Could be a
     pretrained HuggingFace sentence-transformer, a fine-tuned one,
     or a custom torch model. Loaded fresh in `model_fn` at container
     startup, so the FAISS-style "embedding model loaded per request"
     anti-pattern doesn't happen.
  2. **Index** — FAISS (or any ANN library) holding the corpus
     embeddings, queried by vector similarity. Built once during
     "training," persisted as a binary file in the bundle, loaded
     once at startup.

Adding a new retrieval plugin:
  1. Create src/plugins/<name>.py with a RetrievalPlugin subclass.
  2. Register in `_RETRIEVAL_REGISTRY` (or drop in private/).
  3. Add Makefile targets: data-<name>, train-<name>.
"""
import dataclasses
from typing import Any

import pandas as pd


@dataclasses.dataclass
class CorpusItem:
    """One indexable thing. text gets embedded; metadata is returned at query
    time."""
    text: str
    metadata: dict[str, Any]


class RetrievalPlugin:
    name: str = "base_retrieval"

    def prepare_corpus(self, df: pd.DataFrame) -> list[CorpusItem]:
        """Turn the raw dataframe into items ready to embed.

        For products: text might be `title + " " + description`,
        metadata might be `{"product_id": ..., "category": ...}`.
        Whatever's in metadata is what `query()` returns.
        """
        raise NotImplementedError

    def build_embedder(self):
        """Return an embedder with `.encode(list[str], **kwargs) -> np.ndarray`.

        Typically a HuggingFace sentence-transformer; could also be
        a custom torch model with the same interface. Called once at
        startup, so model loading cost is amortized across requests.
        """
        raise NotImplementedError

    def build_index(self, embeddings):
        """Build an ANN index from corpus embeddings. Default: flat L2.

        Override for IVF / HNSW / product-quantized variants when the
        corpus is large enough to need them (typically >100K items
        before flat-L2 starts to hurt at query time).
        """
        import faiss
        import numpy as np
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def query(self, embedder, index, metadata: list[dict],
              text: str | list[str], k: int = 5) -> list[list[dict]]:
        """Single or batched query → list of top-K metadata rows per input.

        Default implementation embeds the query text(s), runs the FAISS
        search, and returns metadata for the matched corpus items.
        Override if you need re-ranking, MMR diversity, hybrid lexical
        + semantic, etc.
        """
        import numpy as np
        texts = [text] if isinstance(text, str) else list(text)
        query_vecs = np.ascontiguousarray(
            embedder.encode(texts), dtype=np.float32
        )
        distances, indices = index.search(query_vecs, k)
        return [
            [{**metadata[idx], "_distance": float(distances[i, j])}
             for j, idx in enumerate(row) if idx != -1]
            for i, row in enumerate(indices)
        ]
