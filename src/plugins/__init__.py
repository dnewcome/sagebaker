"""Plugin registry for metric-specific training modules.

Two plugin families:

TrainingPlugin      — supervised (X/y): train.py harness
                      Add to _SUPERVISED_PLUGINS + make targets.

RecommenderPlugin   — collaborative filtering: train_recommender.py harness
                      Add to _RECOMMENDER_PLUGINS + make targets.

To add a new supervised model:
  1. Create ``src/plugins/<metric>.py`` with a TrainingPlugin subclass.
  2. Add it to ``_SUPERVISED_PLUGINS`` below.
  3. Add ``data-<metric>`` and ``train-<metric>`` Makefile targets.

To add a new recommender:
  1. Create ``src/plugins/<metric>.py`` with a RecommenderPlugin subclass.
  2. Add it to ``_RECOMMENDER_PLUGINS`` below.
  3. Add ``data-<metric>`` and ``train-<metric>`` Makefile targets.

Private plugins
---------------
Drop any ``.py`` file into ``src/plugins/private/`` and it will be
auto-discovered at import time. That directory is gitignored so private
plugins never appear in the public repository. Imports inside private
plugins should use absolute names (e.g. ``from plugins.base import ...``).
"""
import importlib.util
import os

from .als import ALSPlugin
from .base import TrainingPlugin
from .base_recommender import InteractionData, RecommenderPlugin
from .clickstream import ClickstreamPlugin
from .default import DefaultPlugin
from .housing import HousingPlugin

_SUPERVISED_PLUGINS: list[type[TrainingPlugin]] = [
    DefaultPlugin,
    HousingPlugin,
    ClickstreamPlugin,
]

_RECOMMENDER_PLUGINS: list[type[RecommenderPlugin]] = [
    ALSPlugin,
]

_SUPERVISED_REGISTRY: dict[str, type[TrainingPlugin]] = {
    cls.name: cls for cls in _SUPERVISED_PLUGINS
}
_RECOMMENDER_REGISTRY: dict[str, type[RecommenderPlugin]] = {
    cls.name: cls for cls in _RECOMMENDER_PLUGINS
}


def _discover_private_plugins() -> None:
    """Auto-load any .py files found in src/plugins/private/."""
    private_dir = os.path.join(os.path.dirname(__file__), "private")
    if not os.path.isdir(private_dir):
        return
    for entry in sorted(os.listdir(private_dir)):
        if not entry.endswith(".py") or entry.startswith("_"):
            continue
        module_name = entry[:-3]
        spec = importlib.util.spec_from_file_location(
            f"plugins.private.{module_name}",
            os.path.join(private_dir, entry),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in vars(mod).values():
            if not isinstance(attr, type) or attr.__module__ != mod.__name__:
                continue
            if issubclass(attr, TrainingPlugin) and attr is not TrainingPlugin:
                _SUPERVISED_REGISTRY[attr.name] = attr
            elif issubclass(attr, RecommenderPlugin) and attr is not RecommenderPlugin:
                _RECOMMENDER_REGISTRY[attr.name] = attr


_discover_private_plugins()


def get_plugin(name: str) -> TrainingPlugin:
    if name not in _SUPERVISED_REGISTRY:
        raise ValueError(
            f"Unknown supervised plugin {name!r}. "
            f"Available: {sorted(_SUPERVISED_REGISTRY)}"
        )
    return _SUPERVISED_REGISTRY[name]()


def list_plugins() -> list[str]:
    return sorted(_SUPERVISED_REGISTRY)


def get_recommender_plugin(name: str) -> RecommenderPlugin:
    if name not in _RECOMMENDER_REGISTRY:
        raise ValueError(
            f"Unknown recommender plugin {name!r}. "
            f"Available: {sorted(_RECOMMENDER_REGISTRY)}"
        )
    return _RECOMMENDER_REGISTRY[name]()


def list_recommender_plugins() -> list[str]:
    return sorted(_RECOMMENDER_REGISTRY)
