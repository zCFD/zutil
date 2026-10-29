"""
Deck outlines: each reference page's block drawn in place from ``parameters``.

An outline is a tree of nodes.  A node is ``{"key", "link", "value",
"muted", "children", "more"}``: a leaf carries ``value``, the Python text
shown after the key, and a block carries ``children``; ``more`` marks a
block whose other keys are elided, and ``list`` a block written as a
Python list of its items.  ``link`` is the route, with an optional
fragment, of the reference entry the key opens.  The root node has no key
and stands for ``parameters``.
"""

import json
from typing import Dict, List, Optional

from zutil.validate.schemas.solver_schemas import MESH_OVERRIDES

from .anchors import deck_key

_TYPED = {
    "float": "<float>",
    "int": "<int>",
    "str": "<str>",
    "bool": "<bool>",
    "vector (3 floats)": "<[x, y, z]>",
    "unit vector (3 floats)": "<[x, y, z]>",
    "3 ints": "<[i, j, k]>",
    "choice": "<choice>",
    "Python function": "<function>",
}
_BLOCKS = {"settings block", "choice of settings blocks", "named settings blocks"}


def py_literal(value) -> str:
    """A JSON-safe value as a deck writes it in Python."""
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(py_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        if set(value) == {"python"} and isinstance(value["python"], str):
            return value["python"]
        return (
            "{"
            + ", ".join(f"{py_literal(k)}: {py_literal(v)}" for k, v in value.items())
            + "}"
        )
    return str(value)


def node(
    key: Optional[str],
    link: Optional[str] = None,
    value: Optional[str] = None,
    muted: bool = False,
    children: Optional[List[dict]] = None,
    more: bool = False,
) -> dict:
    return {
        "key": key,
        "link": link,
        "value": value,
        "muted": muted,
        "children": children,
        "more": more,
    }


def _is_block(entry: dict) -> bool:
    facts = entry.get("facts") or {}
    return facts.get("type") in _BLOCKS or facts.get("container") == "named"


def placeholder(entry: dict, page_type: Optional[dict] = None) -> tuple:
    """(text, muted) for an entry's value: its fixed value, its default or its type."""
    facts = entry.get("facts") or {}
    options = entry.get("options") or []
    if page_type and entry["key"] == page_type["discriminator"]:
        return py_literal(page_type["value"]), False
    if facts.get("discriminator") and len(options) == 1:
        return py_literal(options[0]["value"]), False
    if _is_block(entry):
        return "{...}", True
    default = facts.get("default") or {}
    if default.get("show"):
        return py_literal(default.get("value")), False
    kind = facts.get("type") or ""
    if facts.get("container") == "list" or kind.startswith("list"):
        return "[...]", True
    return _TYPED.get(kind, f"<{kind}>" if kind else "..."), True


def _entry_link(page: dict, entry: dict) -> str:
    if entry.get("link"):
        return entry["link"]
    page_type = page.get("type")
    if page_type and entry["key"] == page_type["discriminator"] and page.get("parent"):
        return page["parent"]
    return f"{page['route']}#{entry['anchor']}"


def entry_node(page: dict, entry: dict) -> dict:
    text, muted = placeholder(entry, page.get("type"))
    return node(deck_key(entry["key"]), _entry_link(page, entry), text, muted)


def _strip(path: List[str]) -> tuple:
    out = list(path)
    while out and out[-1].startswith("<"):
        out.pop()
    return tuple(out)


def block_routes(pages: List[dict]) -> Dict[tuple, str]:
    """The route that documents each deck path, typed pages yielding to their index."""
    routes: Dict[tuple, str] = {}
    for page in pages:
        if "deck_path" not in page or page.get("kind") == "type":
            continue
        routes.setdefault(_strip(page["deck_path"]), page["route"])
    return routes


def nest(
    path: List[str], routes: Dict[tuple, str], inner: List[dict], more: bool
) -> dict:
    """The ``parameters`` root with ``inner`` placed at ``path``; a ``<n>`` segment is one item of a list."""
    children, tail_more, listed = inner, more, False
    for depth in range(len(path), 0, -1):
        key = path[depth - 1]
        if key == "<n>":
            children = [node(None, children=children, more=tail_more)]
            tail_more, listed = False, True
            continue
        named = key.startswith("<")
        link = None if named else routes.get(tuple(path[:depth]))
        block = node(
            key if named else deck_key(key),
            link,
            children=children,
            more=tail_more,
        )
        if listed:
            block["list"] = True
        children, tail_more, listed = [block], not named, False
    return node(None, routes.get(()), children=children, more=tail_more and bool(path))


def page_outline(page: dict, routes: Dict[tuple, str]) -> Optional[dict]:
    """The page's block in place, every key it lists shown; None for pages without keys."""
    if page.get("kind") in ("output-variables", "type-index"):
        return None
    entries = page.get("entries") or []
    if not entries:
        return None
    inner = [entry_node(page, e) for e in entries]
    page_type = page.get("type")
    if page_type and not any(e["key"] == page_type["discriminator"] for e in entries):
        inner.insert(
            0,
            node(
                page_type["discriminator"],
                page.get("parent"),
                py_literal(page_type["value"]),
            ),
        )
    path = page["deck_path"]
    if path and not path[-1].startswith("<"):
        routes = {**routes, tuple(path): page["route"]}
    return nest(path, routes, inner, False)


def _required_block(page: dict) -> dict:
    """A named instance of the page's type holding only its fixed and required keys."""
    entries = page.get("entries") or []
    kept = [
        entry_node(page, e)
        for e in entries
        if e["key"] == page["type"]["discriminator"]
        or (e.get("facts") or {}).get("required")
    ]
    return node(
        "<name>",
        page["route"],
        children=kept,
        more=len(kept) < len(entries),
    )


def walkthrough(pages: List[dict]) -> Optional[dict]:
    """A complete minimal deck from the root down, with one sentence per top-level key."""
    by_route = {p["route"]: p for p in pages}
    root = by_route.get("reference/input-deck")
    solver = by_route.get("reference/solver")
    model = by_route.get("reference/model")
    if not root or not solver or not model:
        return None

    def required(page: dict) -> List[dict]:
        return [e for e in page["entries"] if (e.get("facts") or {}).get("required")]

    solver_keys = required(solver)
    mesh_children = []
    for entry in model["entries"]:
        link = entry.get("link") or ""
        typed = [
            p for p in pages if p.get("kind") == "type" and p.get("parent") == link
        ]
        if typed and (entry["facts"].get("required") or entry["key"] == "fluid_zones"):
            mesh_children.append(
                node(
                    deck_key(entry["key"]),
                    link,
                    children=[_required_block(typed[0])],
                )
            )
        elif entry["facts"].get("required"):
            mesh_children.append(entry_node(model, entry))
    deck = node(
        None,
        root["route"],
        children=[
            *(entry_node(root, e) for e in root["entries"] if not e.get("link")),
            node(
                "solver",
                solver["route"],
                children=[entry_node(solver, e) for e in solver_keys],
                more=len(solver_keys) < len(solver["entries"]),
            ),
            node(
                "model",
                model["route"],
                children=[
                    node(
                        "<mesh>",
                        None,
                        children=mesh_children,
                        more=len(mesh_children) < len(model["entries"]),
                    )
                ],
            ),
        ],
    )
    return {"deck": deck, "blocks": _sentences(root, solver, model)}


def _names(entries: List[dict]) -> str:
    keys = [f"`{deck_key(e['key'])}`" for e in entries]
    if len(keys) < 2:
        return "".join(keys)
    return ", ".join(keys[:-1]) + " and " + keys[-1]


def _sentences(root: dict, solver: dict, model: dict) -> List[dict]:
    out = []
    for entry in root["entries"]:
        if entry.get("link"):
            continue
        values = [o["value"] for o in entry.get("options") or []]
        if len(values) == 1:
            text = f"The deck format version. The only value accepted is `{py_literal(values[0])}`."
        else:
            text = entry.get("description") or ""
        out.append(
            {
                "key": deck_key(entry["key"]),
                "link": f"{root['route']}#{entry['anchor']}",
                "text": text,
            }
        )
    must = [e for e in solver["entries"] if e["facts"].get("required")]
    rest = [e for e in solver["entries"] if not e["facts"].get("required")]
    text = f"Settings shared by every mesh. It must set {_names(must)}."
    if rest:
        text += f" {_names(rest)} {'is' if len(rest) == 1 else 'are'} optional."
    out.append({"key": "solver", "link": solver["route"], "text": text})
    own = [e for e in model["entries"] if e["key"] not in MESH_OVERRIDES]
    overrides = [e for e in model["entries"] if e["key"] in MESH_OVERRIDES]
    must = [e for e in own if e["facts"].get("required")]
    may = [e for e in own if not e["facts"].get("required")]
    text = f"One block per mesh, under a name of your choosing. Each must set {_names(must)}"
    text += f", and may set {_names(may)}." if may else "."
    if overrides:
        text += f" Its {_names(overrides)} override the solver's blocks of the same name for that mesh."
    out.append({"key": "model", "link": model["route"], "text": text})
    return out
