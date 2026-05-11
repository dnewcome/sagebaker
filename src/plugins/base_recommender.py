"""Base class and shared data structures for recommender training plugins.

Adding a new recommender
------------------------
1. Create ``src/plugins/<name>.py`` with a RecommenderPlugin subclass.
2. Register it in ``src/plugins/__init__.py``.
3. Add ``data-<name>`` and ``train-<name>`` Makefile targets.

Model wrapper contract
----------------------
``build_model()`` must return an object with:

    .fit(train_mat: csr_matrix) -> self
    .recommend(entity_idx: int, K: int, exclude: set[int]) -> list[int]
    .user_factors: np.ndarray   shape (n_entities, n_factors)
    .item_factors: np.ndarray   shape (n_items,    n_factors)

The harness saves ``user_factors`` and ``item_factors`` as a numpy ``.npz``
archive. ``model_fn`` in ``train_recommender.py`` loads them and does
inference as a pure numpy dot product — no training library required at
serving time.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class InteractionData:
    """Everything the harness needs after ``plugin.prepare(df)``.

    ``train_mat``  — scipy CSR matrix, shape (n_entities, n_items).
    ``test_items`` — entity_idx → set of held-out item_idxs (for evaluation).
    ``train_items``— entity_idx → set of training item_idxs (for filtering).
    ``entity_index``— DataFrame with columns ``entity_id``, ``entity_idx``.
    ``item_index`` — DataFrame with columns ``item_idx``,  ``item_id``.
    """
    train_mat: object           # scipy.sparse.csr_matrix
    test_items: dict            # int → set[int]
    train_items: dict           # int → set[int]
    entity_index: pd.DataFrame  # entity_id (str), entity_idx (int)
    item_index: pd.DataFrame    # item_idx (int),  item_id (str)
    n_entities: int
    n_items: int


class RecommenderPlugin:
    """Base class for collaborative-filtering training plugins.

    The plugin owns:
      - Interaction matrix construction from raw data, including any
        filtering and the train/test split (all inside ``prepare``).
      - The model class, its wrapper, and hyperparameter defaults.
      - Optional extra fields for ``config.json`` via ``extra_config``.

    The harness (``train_recommender.py``) owns everything else:
      - Data loading from the SageMaker train channel.
      - Calling ``model.fit(data.train_mat)``.
      - Ranking evaluation (Recall@K, NDCG@K, MAP@K, HitRate@K).
      - Bundle serialization and MLflow tracking.
    """
    name: str = "base_recommender"
    entity_col: str = "entity_id"   # column name in the raw DataFrame
    item_col: str = "item_id"       # column name in the raw DataFrame
    dependencies: list = []

    def prepare(self, df: pd.DataFrame) -> InteractionData:
        """Raw DataFrame → InteractionData (matrix + indices + train/test split).

        All preprocessing — column normalization, interaction filtering,
        integer index construction, sparse matrix build, and train/test
        split — belongs here. The harness calls this once and then fits
        the model on ``data.train_mat``.
        """
        raise NotImplementedError

    def build_model(self, params: dict):
        """Instantiate an unfitted model wrapper.

        ``params`` is a flat dict of strings (SageMaker hyperparameter
        format). Parse what the model needs; provide defaults for anything
        that may be absent.

        The returned object must satisfy the model wrapper contract
        described in this module's docstring.
        """
        raise NotImplementedError

    def extra_config(self, model, data: InteractionData) -> dict:
        """Extra fields to merge into config.json. Optional."""
        return {}

    def load_bundle(self, model_dir: str):
        """Load a recommender bundle from ``model_dir``.

        Default: calls ``model_fn`` from ``train_recommender`` which returns
        a ``RecommenderBundle`` (pure numpy, no training library required).
        Override if the bundle layout differs.
        """
        import train_recommender
        return train_recommender.model_fn(model_dir)

    def serve(self, model, raw_input: list, config: dict) -> dict:
        """End-to-end inference for HTTP serving.

        Recommender plugins must override this — there is no generic
        implementation because output format varies by use case
        (ranked lists, scored pairs, etc.).
        """
        raise NotImplementedError(
            f"Plugin '{self.name}' has no serve() implementation."
        )

    def prepare_data(self, output_dir: str, seed: int = 42, extra_args: list = None) -> None:
        """Generate synthetic training data into ``output_dir``.

        Override in plugins that can generate their own synthetic data for
        local development and CI. The root-level ``prepare.py`` dispatches
        here so all prepare logic lives alongside the plugin it belongs to.
        """
        raise NotImplementedError(
            f"Plugin '{self.name}' has no prepare_data() implementation. "
            "Provide a real dataset via the --train argument instead."
        )
