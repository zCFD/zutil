"""
Examples derived from the published decks.

The corpus is every ``controlfile`` whose entry in a ``*.zcfd-test.yml``
carries ``publish: true``.  For each dict key the example is the most
common non-default value across the corpus; a settings block is taken
whole from one deck, one example per union kind; values are verbatim and
ties are broken by case name.  A key no published deck sets falls back to
its default.
"""

import contextlib
import io
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

from .anchors import normalise_key
from .introspect import json_safe


@dataclass
class Deck:
    case: str
    controlfile: str
    label: Optional[str]
    parameters: Optional[dict] = None
    error: Optional[str] = None

    @property
    def sort_key(self) -> Tuple[str, str]:
        return (self.case, self.controlfile)


def published_entries(testcases: Path) -> Iterator[Tuple[Path, dict]]:
    """(case directory, yml entry) for every entry marked ``publish: true``."""
    for yml in sorted(testcases.rglob("*.zcfd-test.yml")):
        try:
            data = yaml.safe_load(yml.read_text()) or {}
        except yaml.YAMLError:
            continue
        stack = list(data.get("zcfd", []) or []) if isinstance(data, dict) else []
        while stack:
            entry = stack.pop(0)
            if not isinstance(entry, dict):
                continue
            stack.extend(entry.get("zcfd", []) or [])
            if entry.get("publish") is True and entry.get("controlfile"):
                yield yml.parent, entry


def load_corpus(testcases: Path) -> List[Deck]:
    decks: Dict[Tuple[str, str], Deck] = {}
    for case_dir, entry in published_entries(testcases):
        controlfile = str(entry["controlfile"])
        case = case_dir.relative_to(testcases).as_posix()
        if (case, controlfile) in decks:
            continue
        deck = Deck(case, controlfile, entry.get("label"))
        try:
            deck.parameters = _execute(case_dir / controlfile)
        except BaseException as exc:
            deck.error = f"{type(exc).__name__}: {exc}"
        decks[(case, controlfile)] = deck
    return sorted(decks.values(), key=lambda d: d.sort_key)


def _execute(path: Path) -> dict:
    from zutil.fileutils import get_parameters_from_file

    cwd = os.getcwd()
    saved_path = list(sys.path)
    os.chdir(path.parent)
    sys.path.insert(0, str(path.parent))
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            parameters = get_parameters_from_file(
                path.resolve(), namespace={"__file__": str(path.resolve())}
            )
    finally:
        os.chdir(cwd)
        sys.path[:] = saved_path
    if not isinstance(parameters, dict):
        raise TypeError("parameters is not a dict")
    return parameters


def lookup(block: Any, key: str) -> Tuple[bool, Any]:
    """A key in a deck dict, matched case- and space-folded."""
    if not isinstance(block, dict):
        return False, None
    for raw, value in block.items():
        if isinstance(raw, str) and normalise_key(raw) == key:
            return True, value
    return False, None


def instances(deck: Deck, pattern: List[str]) -> List[Any]:
    """Every block in the deck at ``pattern``; ``<mesh>``, ``<name>`` and ``<n>`` match any instance."""
    found = [deck.parameters]
    for segment in pattern:
        nxt = []
        for block in found:
            if segment.startswith("<"):
                values = block.values() if isinstance(block, dict) else block
                if isinstance(block, (dict, list)):
                    nxt.extend(v for v in values if isinstance(v, dict) or callable(v))
            else:
                ok, value = lookup(block, segment)
                if ok:
                    nxt.append(value)
        found = nxt
    return found


def _normalised(value: Any) -> Any:
    safe = json_safe(value)
    return _fold(safe)


def _fold(value: Any) -> Any:
    if isinstance(value, dict):
        return {normalise_key(k): _fold(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fold(v) for v in value]
    return value


def _canon(value: Any) -> str:
    return json.dumps(_normalised(value), sort_keys=True)


def _tag(value: Any, discriminator: Optional[str]) -> Optional[str]:
    if discriminator is None or not isinstance(value, dict):
        return None
    ok, tag = lookup(value, discriminator)
    return str(tag).strip().lower() if ok and tag is not None else None


def select(values: List[Tuple[Deck, Any]], default: Any) -> Optional[dict]:
    """The most common non-default value, ties broken by case name; the default only if nothing else is set."""
    if not values:
        return None
    default_canon = json.dumps(_fold(default), sort_keys=True)
    counts: Counter = Counter()
    first: Dict[str, Tuple[Deck, Any]] = {}
    for deck, value in sorted(values, key=lambda dv: dv[0].sort_key):
        canon = _canon(value)
        counts[canon] += 1
        first.setdefault(canon, (deck, value))
    ranked = sorted(
        counts, key=lambda c: (c == default_canon, -counts[c], first[c][0].sort_key)
    )
    best = ranked[0]
    deck, value = first[best]
    return {
        "value": json_safe(value),
        "source": {"case": deck.case, "controlfile": deck.controlfile},
        "count": counts[best],
    }


WIDELY_USED = 0.5


def derive(
    entries: List[dict],
    blocks: List[Tuple[Deck, Any]],
    gaps: List[dict],
    location: str,
    corpus_size: int = 0,
) -> None:
    """Fill ``examples`` on each entry (recursively) from the blocks the corpus has at this level.

    ``widely_used`` marks a key set by more than ``WIDELY_USED`` of the
    ``corpus_size`` published decks, where naming the cases tells the reader nothing.
    """
    for entry in entries:
        if entry.get("link"):
            entry["examples"] = []
            continue
        values = [
            (d, v) for d, b in blocks for ok, v in [lookup(b, entry["key"])] if ok
        ]
        users = {d.sort_key for d, _ in values}
        entry["widely_used"] = (
            bool(corpus_size) and len(users) > WIDELY_USED * corpus_size
        )
        facts = entry["facts"]
        default = facts["default"]["value"]
        union = entry.get("union") or {}
        disc = union.get("discriminator")
        examples: List[dict] = []
        if entry["fields"] and disc:
            items = items_of(values, facts["container"])
            by_tag: Dict[Optional[str], list] = {}
            for deck, value in items:
                by_tag.setdefault(_tag(value, disc), []).append((deck, value))
            order = [str(v["value"]).lower() for v in union["variants"]]
            for tag in sorted(
                by_tag, key=lambda t: order.index(t) if t in order else len(order)
            ):
                chosen = select(by_tag[tag], None)
                if chosen:
                    chosen["variant"] = tag
                    examples.append(chosen)
        else:
            chosen = select(values, default)
            if chosen:
                chosen["variant"] = None
                examples.append(chosen)
        if not examples:
            if facts["default"]["show"]:
                examples.append(
                    {"value": default, "variant": None, "source": "default", "count": 0}
                )
            elif facts["required"] and not facts.get("discriminator"):
                gaps.append(
                    {
                        "model": entry["declared_in"]["model"],
                        "module": entry["declared_in"]["module"],
                        "key": entry["key"],
                        "anchors": [f"{location}#{entry['anchor']}"],
                    }
                )
        entry["examples"] = examples
        if entry["fields"]:
            derive(
                entry["fields"],
                items_of(values, facts["container"]),
                gaps,
                location,
                corpus_size,
            )


def items_of(
    values: List[Tuple[Deck, Any]], container: Optional[str]
) -> List[Tuple[Deck, Any]]:
    out = []
    for deck, value in values:
        if container == "list" and isinstance(value, list):
            out.extend((deck, v) for v in value)
        elif container == "named" and isinstance(value, dict):
            out.extend((deck, v) for v in value.values())
        else:
            out.append((deck, value))
    return out


def page_blocks(
    corpus: List[Deck], page: dict, extra_paths: List[List[str]]
) -> List[Tuple[Deck, Any]]:
    """The blocks each published deck has at this page's root (and its per-model override path)."""
    blocks = []
    for deck in corpus:
        if deck.parameters is None:
            continue
        for pattern in [page["deck_path"]] + extra_paths:
            for block in instances(deck, pattern):
                if (
                    page.get("type")
                    and _tag(block, page["type"]["discriminator"])
                    != str(page["type"]["value"]).lower()
                ):
                    continue
                blocks.append((deck, block))
    return blocks
