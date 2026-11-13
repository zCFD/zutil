"""Mapping rules, one module per block of the v1 deck.

Each rule module defines:

- ``OWNS``: the v1 top-level keys it claims, as :func:`fnmatch` patterns over
  normalised keys (``"time marching"``, ``"ic *"``, ``"bc *"``);
- ``ORDER``: an int; rules run in ascending order;
- ``convert(c: Conversion) -> None``: claims every leaf under the keys it owns.

Modules are found by name, so adding a rule touches no shared file.
"""

from __future__ import annotations

import fnmatch
import importlib
import pkgutil
from types import ModuleType


def rule_modules() -> list[ModuleType]:
    modules = [
        importlib.import_module(f"{__name__}.{info.name}")
        for info in pkgutil.iter_modules(__path__)
        if not info.name.startswith("_")
    ]
    return sorted(modules, key=lambda m: (m.ORDER, m.__name__))


def owner(top_level_key: str) -> list[ModuleType]:
    """The rule modules whose ``OWNS`` matches a normalised top-level v1 key."""
    return [
        m
        for m in rule_modules()
        if any(fnmatch.fnmatchcase(top_level_key, pat) for pat in m.OWNS)
    ]
