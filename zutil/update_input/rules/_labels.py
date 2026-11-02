"""Deck labels, and values that name a reference condition, shared by the rules."""

from __future__ import annotations

import fnmatch
from typing import Any, Optional

from ..model import Conversion, Leaf, V2Path, Value, norm_key


def bc_labels(c: Conversion) -> list[str]:
    """The deck's boundary-condition labels, as it spelt them."""
    return [k for k in c.v1.children() if fnmatch.fnmatchcase(norm_key(k), "bc *")]


def listed(names: list[str]) -> str:
    """``names`` as an English list: ``a, b and c``."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def ic_labels(c: Conversion) -> list[str]:
    """The deck's reference-condition labels, as it spelt them."""
    return [k for k in c.v1.children() if fnmatch.fnmatchcase(norm_key(k), "ic *")]


def ic_label(c: Conversion, name: Any) -> Optional[str]:
    """The label ``name`` refers to, matched as the v1 solver matched it."""
    if not isinstance(name, str):
        return None
    for label in ic_labels(c):
        if norm_key(name) == norm_key(label):
            return label
    return None


def move_ic_name(c: Conversion, leaf: Leaf, path: V2Path) -> None:
    """Move a value that names an IC, re-spelt to the label when only its case differs."""
    label = ic_label(c, leaf.value)
    if label is not None and leaf.value != label:
        c.move(
            leaf,
            path,
            value=Value(label),
            respelled=True,
            reason=f"the reference condition is spelt {label} in this deck",
        )
        return
    c.move(leaf, path)
