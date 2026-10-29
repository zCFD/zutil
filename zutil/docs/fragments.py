"""
Deck fragments in the guide, checked against the generated reference.

A fragment is a ``json`` or ``python`` fence whose content is a dict, a run
of dict entries, or a script that assigns to ``parameters``.  A fence whose
info string names a file, such as ``json file=amgx.json``, shows some other
file and is not a fragment.  It may elide
with ``...``, ``..`` or ``{...}`` and may use deck variables and function
calls as values; those parts are not checked.  Every key it does name must
exist at that point in the reference, and every option value it gives must
be one the reference lists.
"""

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from .anchors import normalise_key

INSTANCE = "<name>"
_PLACEHOLDERS = {"<name>", "<mesh>"}
_FENCE = re.compile(r"^```(json|python)\b([^\n]*)\n(.*?)^```", re.M | re.S)
_ELISION_LINE = re.compile(r"^\s*\.{2,4},?\s*$", re.M)
_DOTS = re.compile(r"\.{2,4}")


class _Unchecked:
    pass


UNCHECKED = _Unchecked()


@dataclass
class Node:
    children: Dict[str, "Node"] = field(default_factory=dict)
    named: bool = False
    free: bool = False
    options: Optional[list] = None
    options_by_type: Dict[str, list] = field(default_factory=dict)
    by_type: Dict[str, Set[str]] = field(default_factory=dict)

    def child(self, key: str) -> "Node":
        return self.children.setdefault(key, Node())


@dataclass
class Fragment:
    page: Path
    line: int
    value: dict
    root: Optional[List[str]]


@dataclass
class Problem:
    page: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.page}:{self.line}: {self.message}"


def _segment(key: str) -> str:
    return INSTANCE if key in _PLACEHOLDERS else normalise_key(key)


def load_index(reference_dir: Path) -> Node:
    """The key tree the reference documents, with options and named maps."""
    root = Node()
    for path in sorted((reference_dir / "reference").rglob("*.json")):
        page = json.loads(path.read_text())
        if "deck_path" not in page:
            continue
        node = root
        for key in page["deck_path"]:
            if key == "<n>":
                continue
            seg = _segment(key)
            if seg == INSTANCE:
                node.named = True
            node = node.child(seg)
        type_name = (
            page["route"].rsplit("/", 1)[-1] if page.get("kind") == "type" else None
        )
        tops = _add_entries(node, page.get("entries", []), type_name)
        if type_name:
            node.by_type.setdefault(type_name, set()).update(tops)
    _settle(root)
    _share_overrides(root)
    return root


def _share_overrides(root: Node) -> None:
    from zutil.validate.schemas.solver_schemas import MESH_OVERRIDES

    solver = root.children.get("solver")
    mesh = root.children.get("model", Node()).children.get(INSTANCE)
    if not solver or not mesh:
        return
    for key in MESH_OVERRIDES:
        if key in mesh.children and key in solver.children:
            mesh.children[key] = solver.children[key]


def _settle(node: Node) -> None:
    if node.children:
        node.free = False
    for child in node.children.values():
        _settle(child)


def _flatten(entries: list) -> Iterator[dict]:
    for entry in entries:
        yield entry
        yield from _flatten(entry.get("fields") or [])


def _add_entries(
    node: Node, entries: list, type_name: Optional[str] = None
) -> Set[str]:
    flat = list(_flatten(entries))
    named = {
        tuple(e["path"])
        for e in flat
        if (e.get("facts") or {}).get("type", "") == "named settings blocks"
    }
    tops = set()
    for entry in flat:
        path = entry["path"]
        tops.add(normalise_key(path[0]))
        target = node
        for i, key in enumerate(path):
            target = target.child(normalise_key(key))
            if tuple(path[: i + 1]) in named:
                target.named = True
                if i < len(path) - 1:
                    target = target.child(INSTANCE)
        facts = entry.get("facts") or {}
        if entry.get("options"):
            values = [o["value"] for o in entry["options"]]
            known = target.options or []
            target.options = known + [v for v in values if v not in known]
            if type_name:
                target.options_by_type[type_name] = values
        if facts.get("type") == "settings block" and not entry.get("fields"):
            target.free = True
    return tops


def _literal(node: ast.AST):
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if k is None:
                continue
            key = _literal(k)
            if isinstance(key, str):
                out[key] = _literal(v)
        return out
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Constant) and node.value is not Ellipsis:
        if isinstance(node.value, str) and _DOTS.fullmatch(node.value):
            return UNCHECKED
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal(node.operand)
        return -inner if isinstance(inner, (int, float)) else UNCHECKED
    return UNCHECKED


def _subscript_root(target: ast.AST) -> Optional[List[str]]:
    keys = []
    while isinstance(target, ast.Subscript):
        sl = target.slice
        if not (isinstance(sl, ast.Constant) and isinstance(sl.value, str)):
            return None
        keys.append(sl.value)
        target = target.value
    if isinstance(target, ast.Name) and target.id == "parameters":
        return list(reversed(keys))
    return None


def _clean(src: str) -> str:
    src = _ELISION_LINE.sub("", src)
    src = _DOTS.sub("...", src)
    return re.sub(r",[ \t]*\.\.\.[ \t]*(?=[}\]])", "", src)


def _parse(lang: str, src: str) -> Tuple[List[Tuple[dict, Optional[List[str]]]], bool]:
    """Dicts in a fence, each with its explicit root, and whether it read."""
    if lang == "json":
        try:
            value = json.loads(src)
        except ValueError:
            return [], False
        return ([(value, None)] if isinstance(value, dict) else []), True
    src = _clean(src).strip()
    for text in (src, "{" + src.rstrip(",") + "}"):
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError:
            continue
        if isinstance(tree.body, ast.Dict):
            return [(_literal(tree.body), None)], True
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], False
    found = []
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Dict):
            for target in stmt.targets:
                root = _subscript_root(target)
                if root is not None:
                    found.append((_literal(stmt.value), root))
    return found, True


def fragments(
    pages_dir: Path,
) -> Iterator[Tuple[Optional[Fragment], Optional[Problem]]]:
    for page in sorted(pages_dir.rglob("*.md")):
        text = page.read_text()
        for match in _FENCE.finditer(text):
            if "file=" in match.group(2):
                continue
            line = text.count("\n", 0, match.start()) + 1
            rel = page.relative_to(pages_dir)
            found, read = _parse(match.group(1), match.group(3))
            if not read:
                yield (
                    None,
                    Problem(rel, line, "the fragment cannot be read as a deck dict"),
                )
                continue
            for value, root in found:
                yield Fragment(rel, line, value, root), None


def _descend(node: Node, path: List[str]) -> Optional[Node]:
    for key in path:
        seg = _segment(key)
        if seg in node.children:
            node = node.children[seg]
        elif node.named and INSTANCE in node.children:
            node = node.children[INSTANCE]
        else:
            return None
    return node


def _walk(
    node: Node,
    value,
    path: List[str],
    errors: List[str],
    block_type: Optional[str] = None,
) -> None:
    if value is UNCHECKED or node.free:
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                _walk(node, item, path, errors)
        return
    if not isinstance(value, dict):
        options = node.options_by_type.get(block_type, node.options)
        if (
            options is not None
            and isinstance(value, (str, int, float))
            and not isinstance(value, bool)
        ):
            if value not in options:
                shown = ", ".join(repr(o) for o in options)
                errors.append(
                    f"'{' › '.join(path)}' is {value!r}; the reference lists {shown}"
                )
        return
    allowed = None
    type_value = value.get("type")
    route_name = (
        type_value.replace(" ", "-").replace("_", "-")
        if isinstance(type_value, str)
        else None
    )
    if route_name and node.by_type:
        allowed = node.by_type.get(route_name)
    if allowed is None:
        route_name = None
    for key, sub in value.items():
        seg = normalise_key(key)
        where = path + [key]
        if seg in node.children and (allowed is None or seg in allowed):
            _walk(node.children[seg], sub, where, errors, route_name)
        elif node.named and INSTANCE in node.children:
            _walk(node.children[INSTANCE], sub, where, errors)
        elif seg in node.children:
            errors.append(
                f"'{' › '.join(where)}' is not a setting of a {type_value!r} block"
            )
        else:
            errors.append(f"'{' › '.join(where)}' is not in the reference")


def _candidates(index: Node, keys: Set[str]) -> List[List[str]]:
    out = []

    def visit(node: Node, path: List[str]):
        if keys <= set(node.children):
            out.append(path)
        for seg, child in node.children.items():
            visit(child, path + [seg])

    visit(index, [])
    return out


def check(fragment: Fragment, index: Node) -> List[str]:
    """Errors for one fragment; empty when it matches the reference."""
    value = fragment.value
    if fragment.root is not None:
        start = _descend(index, fragment.root)
        if start is None:
            return [f"'{' › '.join(fragment.root)}' is not in the reference"]
        errors: List[str] = []
        _walk(start, value, list(fragment.root), errors)
        return errors
    keys = {normalise_key(k) for k in value}
    roots = _candidates(index, keys)
    if not roots and all(isinstance(v, dict) for v in value.values()):
        inner = set()
        for v in value.values():
            inner |= {normalise_key(k) for k in v}
        roots = [r[:-1] for r in _candidates(index, inner) if r and r[-1] == INSTANCE]
        typed = all(isinstance(v.get("type"), str) for v in value.values())
        roots = [
            r
            for r in roots
            if (n := _descend(index, r)) is not None
            and n.named
            and (typed or not n.children[INSTANCE].by_type)
        ]
    if not roots:
        return [f"no reference block has all of {', '.join(sorted(value))}"]
    best: Optional[List[str]] = None
    for root in sorted(roots, key=len):
        errors = []
        _walk(_descend(index, root), value, [k for k in root if k != INSTANCE], errors)
        if not errors:
            return []
        if best is None or len(errors) < len(best):
            best = errors
    return best or []


def check_pages(pages_dir: Path, reference_dir: Path) -> List[Problem]:
    index = load_index(reference_dir)
    problems = []
    for fragment, problem in fragments(pages_dir):
        if problem:
            problems.append(problem)
            continue
        for message in check(fragment, index):
            problems.append(Problem(fragment.page, fragment.line, message))
    return problems
