"""Scenario registry for synthetic-data generators.

Mirrors src/plugins/__init__.py: public scenarios are listed below;
anything dropped into simulate/scenarios/private/ is auto-discovered at
import time. The private/ directory is gitignored so work-internal
scenarios (replicas of real-data shapes) never reach the public repo.

To add a public scenario:
  1. Create simulate/scenarios/<name>.py with a Scenario subclass.
  2. Add it to _SCENARIOS below.
  3. Optionally add a Makefile shortcut: data-<name>.

To add a private scenario:
  1. Create simulate/scenarios/private/<name>.py with a Scenario subclass.
  2. That's it — auto-discovered at import time, gitignored.
"""
import importlib.util
import os

from .base import Scenario, SimulationResult
from .scenarios.fuzzy_clickstream import FuzzyClickstreamScenario
from .scenarios.product_catalog import ProductCatalogScenario

_SCENARIOS: list[type[Scenario]] = [
    FuzzyClickstreamScenario,
    ProductCatalogScenario,
]

_REGISTRY: dict[str, type[Scenario]] = {cls.name: cls for cls in _SCENARIOS}


def _discover_private_scenarios() -> None:
    base_dir = os.path.dirname(__file__)
    private_dir = os.path.join(base_dir, "scenarios", "private")
    if not os.path.isdir(private_dir):
        return
    for entry in sorted(os.listdir(private_dir)):
        if not entry.endswith(".py") or entry.startswith("_"):
            continue
        module_name = entry[:-3]
        spec = importlib.util.spec_from_file_location(
            f"simulate.scenarios.private.{module_name}",
            os.path.join(private_dir, entry),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in vars(mod).values():
            if not isinstance(attr, type) or attr.__module__ != mod.__name__:
                continue
            if issubclass(attr, Scenario) and attr is not Scenario:
                _REGISTRY[attr.name] = attr


_discover_private_scenarios()


def get_scenario(name: str) -> Scenario:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown scenario {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def list_scenarios() -> list[str]:
    return sorted(_REGISTRY)


def describe_scenarios() -> dict[str, str]:
    return {name: cls.description for name, cls in _REGISTRY.items()}
