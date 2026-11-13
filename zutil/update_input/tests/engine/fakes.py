"""Stand-ins for the source layer and the rule modules, so the engine is tested alone.

The real ``source.py`` and rule modules are written in parallel with the
engine. These fakes honour the same contracts: ``read_deck`` returns a
``V1Deck`` and an opaque deck-file object, ``render_deck`` returns the whole
new file text, and a rule module has ``ORDER``, ``OWNS`` and ``convert``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Callable

import zutil.update_input as update_input_pkg
from zutil.update_input import rules as rules_pkg
from zutil.update_input import source as real_source
from zutil.update_input.model import (
    Expr,
    Leaf,
    Placeholder,
    Source,
    V1Deck,
    Value,
)

SHOCKTUBE_PRELUDE = """xmin = [12]
xmax = [13]
sym = [14, 15, 16, 17]
R = 1.0
"""

SHOCKTUBE_V1 = {
    "units": "SI",
    "reference": "IC_1",
    "restart": False,
    "partitioner": "metis",
    "time marching": {
        "unsteady": {"total time": 0.2, "time step": 1.0e-5},
        "scheme": {
            "name": "runge kutta",
            "stage": "rk third order tvd",
            "kind": "global timestepping",
        },
    },
    "equations": "euler",
    "euler": {"order": "first", "entropy fix": 0.05},
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": 1.0,
        "Sutherlands const": 110.4,
        "Prandtl No": 0.71,
        "Turbulent Prandtl No": 0.9,
    },
    "IC_1": {
        "temperature": 1.0,
        "pressure": 1.0,
        "V": {"vector": [0.75, 0.0, 0.0]},
        "Reynolds No": 1.0,
        "Reference Length": 1.0,
    },
    "IC_2": {
        "temperature": 0.8,
        "pressure": 0.1,
        "V": {"vector": [0.0, 0.0, 0.0]},
        "Reynolds No": 1.0,
        "Reference Length": 1.0,
    },
    "BC_1": {"zone": [14, 15, 16, 17], "type": "symmetry"},
    "BC_2": {"zone": [12], "type": "farfield", "kind": "riemann", "condition": "IC_1"},
    "BC_3": {"zone": [13], "type": "farfield", "kind": "riemann", "condition": "IC_2"},
    "write output": {
        "format": "vtk",
        "surface variables": ["V", "p", "T", "rho"],
        "volume variables": ["V", "p", "T", "rho"],
    },
    "report": {"frequency": 1000},
}

SHOCKTUBE_SOURCES = {
    ("air", "gas constant"): "R",
    ("BC_1", "zone"): "sym",
    ("BC_2", "zone"): "xmin",
    ("BC_3", "zone"): "xmax",
}


def shocktube_v2(label: str) -> dict:
    """A v2 deck that validates, apart from ``config version``, which the engine sets."""
    return {
        "solver": {
            "equations": {"type": "euler"},
            "numerical scheme": {"order": "first", "entropy fix": 0.05},
            "fluid properties": {
                "material": "air",
                "gamma": 1.4,
                "gas constant": 1.0,
                "sutherlands const": 110.4,
                "prandtl no": 0.71,
            },
            "reference conditions": {
                "IC_1": {
                    "temperature": 1.0,
                    "pressure": 1.0,
                    "v": {"vector": [0.75, 0.0, 0.0]},
                    "reference length": 1.0,
                },
                "IC_2": {
                    "temperature": 0.8,
                    "pressure": 0.1,
                    "v": {"vector": [0.0, 0.0, 0.0]},
                    "reference length": 1.0,
                },
            },
            "output settings": {
                "solution": {
                    "format": "vtk",
                    "surface variables": ["V", "p", "T", "rho"],
                    "volume variables": ["V", "p", "T", "rho"],
                },
                "report": {"frequency": 1000},
            },
            "solver settings": {"type": "compressible", "partitioner": "metis"},
            "initialisation": {"initial conditions": "IC_1", "restart": False},
            "time settings": {
                "type": "unsteady",
                "kind": "global timestepping",
                "total time": 0.2,
                "time step": 1.0e-5,
                "scheme": {"name": "runge kutta", "stage": 3},
            },
        },
        "model": {
            label: {
                "boundary conditions": {
                    "BC_1": {"zones": [14, 15, 16, 17], "type": "symmetry"},
                    "BC_2": {
                        "zones": [12],
                        "type": "farfield",
                        "kind": "riemann",
                        "condition": "IC_1",
                    },
                    "BC_3": {
                        "zones": [13],
                        "type": "farfield",
                        "kind": "riemann",
                        "condition": "IC_2",
                    },
                }
            }
        },
    }


def flatten(tree: dict, sources: dict | None = None, prefix: tuple = ()) -> list[Leaf]:
    """One leaf per non-dict value, with the source text the deck would have written."""
    sources = sources or {}
    leaves = []
    for key, value in tree.items():
        keys = prefix + (key,)
        if isinstance(value, dict):
            leaves.extend(flatten(value, sources, keys))
        else:
            leaves.append(Leaf(keys, value, sources.get(keys, repr(value))))
    return leaves


def paths_of(tree: dict, prefix: tuple = ()) -> list[tuple[tuple, Any]]:
    out = []
    for key, value in tree.items():
        if isinstance(value, dict) and value:
            out.extend(paths_of(value, prefix + (key,)))
        else:
            out.append((prefix + (key,), value))
    return out


def shocktube_deck() -> V1Deck:
    import copy

    evaluated = copy.deepcopy(SHOCKTUBE_V1)
    return V1Deck(flatten(evaluated, SHOCKTUBE_SOURCES), evaluated)


def _text(value) -> str:
    if isinstance(value, Source):
        if value.leaf.source is not None:
            return value.leaf.source
        return repr(value.leaf.value)
    if isinstance(value, Value):
        return repr(value.value)
    if isinstance(value, Expr):
        return value.text
    if isinstance(value, Placeholder):
        return repr(value.text)
    raise TypeError(value)


def _nest(v2: dict) -> dict:
    root: dict = {}
    for path, value in v2.items():
        node = root
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return root


def _literal(node) -> str:
    if isinstance(node, dict):
        return "{" + ", ".join(f"{k!r}: {_literal(v)}" for k, v in node.items()) + "}"
    return _text(node)


class FakeDeckFile:
    def __init__(self, path: Path, prelude: str):
        self.path = path
        self.prelude = prelude


class FakeSource(types.ModuleType):
    """A source layer that hands back a prepared deck and renders a flat literal."""

    def __init__(self, deck: V1Deck, prelude: str = SHOCKTUBE_PRELUDE):
        super().__init__("zutil.update_input.source")
        self.deck = deck
        self.prelude = prelude
        self.read_paths: list[Path] = []
        self.rendered: dict | None = None

    def read_deck(self, path, namespace=None):
        self.read_paths.append(Path(path))
        return self.deck, FakeDeckFile(Path(path), self.prelude)

    def render_deck(self, deck_file, v2, base_v2=None):
        self.rendered = dict(v2)
        return deck_file.prelude + "parameters = " + _literal(_nest(v2)) + "\n"

    def related_decks(self, deck_file):
        return []

    def read_related(self, deck_file, path):
        raise ValueError(f"{deck_file.path.name} has no related deck {path}")

    v2_name = staticmethod(real_source.v2_name)
    DeckReadError = real_source.DeckReadError
    RenderError = real_source.RenderError


def install_source(monkeypatch, fake: FakeSource) -> FakeSource:
    monkeypatch.setitem(sys.modules, "zutil.update_input.source", fake)
    monkeypatch.setattr(update_input_pkg, "source", fake, raising=False)
    return fake


def rule(
    name: str, order: int, owns: tuple[str, ...], convert: Callable
) -> types.ModuleType:
    module = types.ModuleType(name)
    module.ORDER = order
    module.OWNS = owns
    module.convert = convert
    return module


def install_rules(monkeypatch, *modules: types.ModuleType) -> None:
    monkeypatch.setattr(rules_pkg, "rule_modules", lambda: list(modules))


def complete_shocktube_rule(label: str) -> types.ModuleType:
    """Claims every leaf and writes a v2 deck that validates."""

    def convert(c):
        for leaf in c.v1:
            if leaf.path == ("air", "gas constant"):
                continue
            c.drop(leaf, "covered by the test's v2 tree")
        gas = c.v1.leaf("air", "gas constant")
        for path, value in paths_of(shocktube_v2(label)):
            if path == ("solver", "fluid properties", "gas constant"):
                c.move(gas, path)
            else:
                c.set(path, Value(value))

    return rule("fake_complete", 10, ("*",), convert)
