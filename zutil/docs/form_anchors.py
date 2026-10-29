"""
Reference anchors in the form JSON Schema.

Every property the reference documents carries ``zcfd.anchor``, the
``route#fragment`` of its entry, so a form links a field to the reference
without deriving the anchor itself.

A model reached from more than one place in the deck can document one
property under several anchors (a wall's ``WallScalarField`` under
``temperature`` and under ``heat_flux``).
Such a property carries ``zcfd.anchors`` instead: a map from the property's
deck-path pattern to its anchor.  A pattern is the dotted schema-key path with
each instance name written ``<name>`` (``<mesh>`` for the mesh, ``<n>`` for a
list index) and a boundary-condition or fluid-zone name written
``<name>[<type>]``, for example
``model.<mesh>.boundary_conditions.<name>[wall].fsi.zone``.
"""

import functools
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

from .anchors import kebab, locate


def annotate(schema: dict, root: str) -> dict:
    """Write each documented property's anchor into ``schema`` in place and return it."""
    for (model, key), by_pattern in field_anchors().items():
        holder = schema if model == root else schema.get("$defs", {}).get(model)
        prop = (holder or {}).get("properties", {}).get(key)
        if not isinstance(prop, dict):
            continue
        extra = prop.setdefault("zcfd", {})
        values = set(by_pattern.values())
        if len(values) == 1:
            extra["anchor"] = values.pop()
        else:
            extra["anchors"] = dict(sorted(by_pattern.items()))
    return schema


@functools.lru_cache(maxsize=1)
def field_anchors() -> Dict[Tuple[str, str], Dict[str, str]]:
    """(model name, key) -> {deck-path pattern: anchor}, for every entry the reference emits."""
    from .tree import Walker

    out: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for page in Walker().build():
        if page.kind == "output-variables":
            continue
        tag = page.type_value
        for deck, pattern in emitted_paths(page):
            where = locate(deck, tag)
            for owner in where.owners:
                out[(owner.__name__, deck[-1])][".".join(pattern)] = where.anchor
    return dict(out)


def emitted_paths(page) -> Iterator[Tuple[List[str], List[str]]]:
    """(deck key path with sample instance names, deck-path pattern) for each entry on ``page``."""
    tag = page.type_value
    for root in [page.deck_path] + page.extra_deck_paths:
        deck = [_SAMPLE.get(k, k) for k in root]
        pattern = [_placeholder(k, tag) for k in root]
        yield from _entries(page.entries, deck, pattern)


_SAMPLE = {"<mesh>": "m1", "<name>": "n1", "<n>": "0"}
_CONTAINERS = {"named": "<name>", "list": "<n>"}


def _placeholder(key: str, tag: Optional[str]) -> str:
    if key == "<name>" and tag is not None:
        return f"<name>[{kebab(tag)}]"
    return key


def _entries(entries: list, deck: list, pattern: list):
    for entry in entries:
        key = entry["key"]
        yield deck + [key], pattern + [key]
        container = _CONTAINERS.get(entry["facts"].get("container"))
        if container is None:
            inner_deck, inner_pattern = deck + [key], pattern + [key]
        else:
            inner_deck, inner_pattern = deck + [key, "0"], pattern + [key, container]
        yield from _entries(entry.get("fields") or [], inner_deck, inner_pattern)
