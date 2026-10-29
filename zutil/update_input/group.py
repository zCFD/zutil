"""Convert a v1 overset group to one config version 2 deck.

A v1 overset run is an ``override`` deck whose ``mesh_case_pair`` lists
``(mesh, deck)`` pairs; the solver ran each deck on its own mesh. Each deck is
converted by the single-deck rules under its own model label, and the trees
are then merged:

- a ``model`` block stays with its model;
- a ``solver`` setting every model gives the same value is written once;
- a block v2 lets a model override (:data:`OVERRIDABLE`) is compared whole:
  one block every model shares goes under ``solver`` alone; otherwise every
  model carries its own block complete, because the C++ reads a model's block
  in place of the solver's, and ``solver`` holds the keys of the most common
  block, ties to the first pair, that every model's block sets, because the
  Python solver merges a model's output settings over the solver's; the
  incompressible coupling scheme
  (:data:`INCOMPRESSIBLE_SOLVER_ONLY`) is left out of that block and placed
  as a solver-only setting;
- in any other ``solver`` block, models that disagree, or a value some models
  leave unset, is a decision, except a named entry such as a reference
  condition that only some models define;
- values are written as evaluated, because each deck's names belong to that
  deck; a function is copied into the new deck when it reads nothing from its
  deck but modules, and is a decision otherwise.
"""

from __future__ import annotations

import ast
import builtins
import dis
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from . import report
from .convert import deck_chain, relative
from .mesh import MeshReadError, read_mesh
from .model import (
    UNCONVERTED,
    Conversion,
    ConversionError,
    Disposition,
    Expr,
    Leaf,
    MeshInfo,
    Note,
    Placeholder,
    Source,
    V1Deck,
    V1Path,
    V2Path,
    Value,
    norm_key,
)
from .rules._labels import bc_labels, listed

GROUP = "override"
PAIRS = "mesh_case_pair"
OVERRIDABLE = (
    "initialisation",
    "convergence control",
    "numerical scheme",
    "output settings",
)
NAMED = ("reference conditions",)
INCOMPRESSIBLE_SOLVER_ONLY = (("solver", "convergence control", "scheme"),)
PER_MESH = (
    (
        ("mapping", "number of additional overset active layers"),
        "overset",
        "overset interface",
        "the layers",
    ),
    (("sponge layer damping",), "farfield", "farfield boundary", "the sponge layer"),
)
_STAND_IN = "parameters = None"


@dataclass(frozen=True)
class Pair:
    """One ``(mesh, deck)`` entry of the group, and the model it becomes."""

    mesh: str
    deck: Path
    label: str


@dataclass
class Group:
    """An ``override`` deck read for conversion.

    ``target`` is the span of the ``override`` statement in ``text``, which
    the new ``parameters`` statement replaces. ``extra`` are the group's
    other keys, such as ``device``.
    """

    path: Path
    text: str
    target: tuple[int, int]
    pairs: list[Pair]
    extra: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def read_group(path: Union[str, Path]) -> Optional[Group]:
    """The group at ``path``, or None when the deck is not an overset group.

    Raises:
        DeckReadError: the deck sets ``override`` but it is not a valid group.
    """
    from .convert import in_deck_dir
    from .source import DeckReadError

    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None
    statement = _override_statement(tree)
    if statement is None:
        return None
    namespace: dict = {"__file__": str(path.resolve())}
    with in_deck_dir(path):
        try:
            exec(compile(tree, str(path), "exec"), namespace)
        except Exception as e:
            raise DeckReadError(
                f"{path.name} does not run: {type(e).__name__}: {e}"
            ) from e
    if "parameters" in namespace:
        return None
    override = namespace.get(GROUP)
    if not isinstance(override, dict) or PAIRS not in override:
        raise DeckReadError(
            f"{path.name} sets override but no override > {PAIRS}, so it is not an overset group"
        )
    pairs, notes = _pairs(path, override[PAIRS])
    return Group(
        path=path,
        text=text,
        target=_span(text, statement),
        pairs=pairs,
        extra={k: v for k, v in override.items() if k != PAIRS},
        notes=notes,
    )


def _override_statement(tree: ast.Module) -> Optional[ast.stmt]:
    found = None
    for stmt in tree.body:
        targets = []
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if "parameters" in names:
            return None
        if GROUP in names:
            found = stmt
    return found


def _span(text: str, stmt: ast.stmt) -> tuple[int, int]:
    from .source import _line_starts, _offset

    starts = _line_starts(text)
    return _offset(text, starts, stmt.lineno, stmt.col_offset), _offset(
        text, starts, stmt.end_lineno, stmt.end_col_offset
    )


def _pairs(path: Path, raw: Any) -> tuple[list[Pair], list[str]]:
    from .source import DeckReadError

    if not isinstance(raw, (list, tuple)):
        raise DeckReadError(
            f"{path.name}: override > {PAIRS} must be a list of (mesh, deck) pairs"
        )
    if not raw:
        raise DeckReadError(
            f"{path.name}: override > {PAIRS} names no (mesh, deck) pairs"
        )
    entries = []
    for n, entry in enumerate(raw, start=1):
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
            or not all(isinstance(e, str) for e in entry)
        ):
            raise DeckReadError(
                f"{path.name}: override > {PAIRS} pair {n} is {entry!r}; "
                "each pair is (mesh file name, deck file name)"
            )
        mesh, deck = entry
        mesh = mesh if mesh.endswith(".h5") else mesh + ".h5"
        deck = deck if deck.endswith(".py") else deck + ".py"
        deck_path = path.parent / deck
        if not deck_path.is_file():
            raise DeckReadError(
                f"{path.name}: override > {PAIRS} pair {n} names {deck}, which does not exist"
            )
        entries.append((mesh, deck_path))

    stems = [d.stem for _, d in entries]
    repeated = sorted({s for s in stems if stems.count(s) > 1})
    count: dict[str, int] = {}
    pairs = []
    for mesh, deck_path in entries:
        stem = deck_path.stem
        label = stem
        if stem in repeated:
            count[stem] = count.get(stem, 0) + 1
            label = f"{stem}_{count[stem]}"
        pairs.append(Pair(mesh, deck_path, label))
    notes = []
    for stem in repeated:
        labels = ", ".join(p.label for p in pairs if p.deck.stem == stem)
        notes.append(
            f"{stem}.py is listed more than once, so its models are numbered: {labels}."
        )
    return pairs, notes


@dataclass
class _Member:
    pair: Pair
    c: Conversion
    decks: list[Path]
    flat: dict[V2Path, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.pair.label


@dataclass(frozen=True)
class _Pending:
    """A member's placeholder, standing for that member's decision ``id``."""

    label: str
    id: int
    text: str


@dataclass
class _Ask:
    """A decision of the group deck, made before it is numbered."""

    path: V2Path
    question: str
    options: tuple[str, ...] = ()
    details: str = ""
    v1_paths: list[V1Path] = field(default_factory=list)
    value: Any = None


@dataclass
class _Plain:
    value: Any
    original: Any
    label: str


class _NotPlain(Exception):
    pass


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    if isinstance(value, dict):
        return {_plain(k): _plain(v) for k, v in value.items()}
    raise _NotPlain(type(value).__name__)


def _evaluated(value: Any) -> Any:
    if isinstance(value, Source):
        return value.leaf.value
    if isinstance(value, (Value, Expr)):
        return value.value
    return value


def _flatten(member: _Member) -> None:
    for path, value in member.c.v2.items():
        if isinstance(value, Placeholder):
            member.flat[path] = _Pending(member.label, value.decision_id, value.text)
            continue
        _flatten_value(member, path, _evaluated(value), value)


def _flatten_value(member: _Member, path: V2Path, value: Any, original: Any) -> None:
    if (
        path[0] != UNCONVERTED
        and isinstance(value, dict)
        and value
        and all(isinstance(k, str) for k in value)
    ):
        for key, sub in value.items():
            _flatten_value(member, path + (key,), sub, Value(sub))
        return
    member.flat[path] = _Plain(value, original, member.label)


def _same(a: Any, b: Any) -> bool:
    from .source import same_value, without_item

    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, _Pending) or isinstance(b, _Pending):
        return (
            isinstance(a, _Pending)
            and isinstance(b, _Pending)
            and without_item(a.text) == without_item(b.text)
        )
    if isinstance(a, _Plain) and isinstance(b, _Plain):
        if inspect.isfunction(a.value) or inspect.isfunction(b.value):
            return _same_behaviour(a.value, b.value)
        return same_value(a.value, b.value)
    return False


def _unit_text(unit: Any) -> str:
    if isinstance(unit, dict):
        return "{" + ", ".join(f"{k!r}: {_unit_text(v)}" for k, v in unit.items()) + "}"
    if isinstance(unit, _Pending):
        return "an open decision"
    try:
        return repr(_plain(unit.value))
    except _NotPlain:
        return _original_text(unit)


def _original_text(item: _Plain) -> str:
    original = item.original
    if isinstance(original, Source) and original.leaf.source is not None:
        return original.leaf.source
    if isinstance(original, Expr):
        return original.text
    if callable(item.value) and hasattr(item.value, "__name__"):
        return item.value.__name__
    return repr(item.value)


class _Merge:
    """Merges the members' flat v2 trees into the group's tree."""

    def __init__(self, members: list[_Member]) -> None:
        self.members = members
        self.tree: dict[V2Path, Any] = {}
        self.where: dict[tuple[str, V2Path], V2Path] = {}
        self.asks: list[_Ask] = []
        self.absorbed: dict[tuple[str, int], _Ask] = {}
        self.notes: list[Note] = []
        self.solver_only: tuple[V2Path, ...] = ()

    def _within(self, path: V2Path, unit: V2Path) -> bool:
        if path[: len(unit)] != unit:
            return False
        return not any(
            path[: len(p)] == p and unit[: len(p)] != p for p in self.solver_only
        )

    def run(self) -> None:
        if any(_solver_type(m) == "incompressible" for m in self.members):
            self.solver_only = INCOMPRESSIBLE_SOLVER_ONLY
        units = self._units()
        for unit in units:
            self._place(unit)

    def _units(self) -> list[V2Path]:
        every: list[V2Path] = []
        seen: set[V2Path] = set()
        for m in self.members:
            for path in m.flat:
                if path not in seen:
                    seen.add(path)
                    every.append(path)
        prefixes = {p[:n] for p in every for n in range(1, len(p))}
        units: list[V2Path] = []
        taken: set[V2Path] = set()
        for path in every:
            only = next((p for p in self.solver_only if path[: len(p)] == p), None)
            if only is not None:
                unit = only
            elif path[0] == "solver" and len(path) > 2 and path[1] in OVERRIDABLE:
                unit = path[:2]
            else:
                unit = next(
                    (
                        path[:n]
                        for n in range(1, len(path) + 1)
                        if path[:n] in seen and path[:n] in prefixes
                    ),
                    path,
                )
            if unit not in taken:
                taken.add(unit)
                units.append(unit)
        return units

    def _value(self, m: _Member, unit: V2Path) -> Any:
        if unit in m.flat:
            return m.flat[unit]
        nested: dict = {}
        for path, item in m.flat.items():
            if self._within(path, unit) and len(path) > len(unit):
                node = nested
                for key in path[len(unit) : -1]:
                    node = node.setdefault(key, {})
                node[path[-1]] = item
        return nested or None

    def _put(self, m: _Member, unit: V2Path, target: V2Path) -> None:
        for path, item in m.flat.items():
            if self._within(path, unit):
                group_path = target + path[len(unit) :]
                self.where[(m.label, path)] = group_path
                self.tree.setdefault(group_path, item)

    def _place(self, unit: V2Path) -> None:
        holders = [
            (m, v) for m in self.members if (v := self._value(m, unit)) is not None
        ]
        head = unit[0]
        if head == "model" or head == "config version":
            for m, _ in holders:
                self._put(m, unit, unit)
            return
        if head == UNCONVERTED:
            if len(holders) == len(self.members) and _all_same(holders):
                for m, _ in holders:
                    self._put(m, unit, unit)
                return
            for m, _ in holders:
                named = (UNCONVERTED, f"{m.label} > {unit[1]}") + unit[2:]
                self._put(m, unit, named)
            return
        block = unit[1] if len(unit) > 1 else ""
        if head == "solver" and block in OVERRIDABLE and unit not in self.solver_only:
            self._place_overridable(unit, holders)
            return
        missing = [m for m in self.members if all(m is not h for h, _ in holders)]
        named_entry = block in NAMED and len(unit) > 2
        absent = all(not any(p[:3] == unit[:3] for p in m.flat) for m in missing)
        if _all_same(holders) and (not missing or (named_entry and absent)):
            for m, _ in holders:
                self._put(m, unit, unit)
            return
        self._conflict(unit, holders, missing)

    def _place_overridable(self, unit: V2Path, holders: list) -> None:
        classes: list[list] = []
        for m, value in holders:
            for cls in classes:
                if _same(cls[0][1], value):
                    cls.append((m, value))
                    break
            else:
                classes.append([(m, value)])
        winner = max(classes, key=len)
        if len(classes) == 1:
            for m, _ in winner:
                self._put(m, unit, unit)
        else:
            first = winner[0][0]
            shared = [
                (path, item)
                for path, item in first.flat.items()
                if self._within(path, unit) and all(path in m.flat for m, _ in holders)
            ]
            for path, item in shared or [(unit, _Plain({}, Value({}), first.label))]:
                self.tree.setdefault(path, item)
            for m, _ in holders:
                self._put(m, unit, ("model", m.label) + unit[1:])
        for m in self.members:
            if all(m is not h for h, _ in holders):
                self.notes.append(
                    Note(f"{m.label} sets no {unit[1]}, so it takes the solver's.")
                )

    def _conflict(self, unit: V2Path, holders: list, missing: list) -> None:
        options: list[str] = []
        said = []
        for m, value in holders:
            text = _unit_text(value)
            said.append(f"{m.label} sets {text}")
            if not isinstance(value, _Pending) and text not in options:
                options.append(text)
        said += [f"{m.label} does not set it" for m in missing]
        where = " > ".join(unit[1:])
        ask = _Ask(
            path=unit,
            question=(
                f"The models set {where} differently, and config version 2 sets it "
                "once for the whole run. Which value holds?"
            ),
            options=tuple(options),
            details="; ".join(said) + ".",
        )
        for m, value in holders:
            self._absorb(m, unit, ask)
        self.tree[unit] = ask
        self.asks.append(ask)

    def _absorb(self, m: _Member, unit: V2Path, ask: _Ask) -> None:
        for path, item in m.flat.items():
            if path[: len(unit)] == unit:
                self.where[(m.label, path)] = path
                if isinstance(item, _Pending):
                    self.absorbed[(m.label, item.id)] = ask

    def group_path(self, label: str, v2_path: Optional[V2Path]) -> Optional[V2Path]:
        """Where a member's v2 path landed in the group tree."""
        if v2_path is None:
            return None
        if (label, v2_path) in self.where:
            return self.where[(label, v2_path)]
        for (owner, path), target in self.where.items():
            if owner != label:
                continue
            if path[: len(v2_path)] == v2_path:
                return target[: len(target) - (len(path) - len(v2_path))]
            if v2_path[: len(path)] == path:
                return target + v2_path[len(path) :]
        return None


def _function_at(m: _Member, d: Any) -> Any:
    for path in d.v1_paths:
        leaf = m.c.v1.leaf(*path)
        if leaf is not None and inspect.isfunction(leaf.value):
            return leaf.value
    return None


def _solver_type(m: _Member) -> Any:
    item = m.flat.get(("solver", "solver settings", "type"))
    return item.value if isinstance(item, _Plain) else None


def _same_behaviour(a: Any, b: Any) -> bool:
    """Whether two deck values are one function: same code, same deck values read."""
    from .source import same_value

    if a is b:
        return True
    if not (inspect.isfunction(a) and inspect.isfunction(b)):
        return False
    if a.__code__ != b.__code__ or a.__code__.co_filename != b.__code__.co_filename:
        return False
    for name in _global_names(a.__code__):
        if (name in a.__globals__) != (name in b.__globals__):
            return False
        if name in a.__globals__ and not same_value(
            a.__globals__[name], b.__globals__[name]
        ):
            return False
    return True


def _all_same(holders: list) -> bool:
    return all(_same(holders[0][1], v) for _, v in holders[1:])


def _global_names(code: Any) -> set[str]:
    names = set()
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            names.add(str(instruction.argval).removeprefix("NULL + "))
    for const in code.co_consts:
        if inspect.iscode(const):
            names |= _global_names(const)
    return names


class _Uncopyable(Exception):
    pass


def _function_text(func: Any) -> tuple[str, list[str]]:
    """The source of a deck function and the imports it needs.

    Raises:
        _Uncopyable: with the reason, when the function is not self-contained.
    """
    from .rules._functions import top_level_def

    found = top_level_def(func)
    if isinstance(found, str):
        raise _Uncopyable(found)
    _, text, _, node = found
    names = _global_names(func.__code__)
    for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
        names |= {n.id for n in ast.walk(default) if isinstance(n, ast.Name)}
    imports = []
    reads = []
    for name in sorted(names):
        if name in func.__globals__:
            value = func.__globals__[name]
            if value is func:
                continue
            if inspect.ismodule(value):
                module = value.__name__
                imports.append(
                    f"import {module}"
                    if module == name
                    else f"import {module} as {name}"
                )
                continue
            reads.append(name)
        elif not hasattr(builtins, name):
            reads.append(name)
    if reads:
        raise _Uncopyable(f"reads {', '.join(reads)} from its deck")
    return ast.get_source_segment(text, node), imports


class _Functions:
    """The functions copied into the group deck, by name."""

    def __init__(self, rewrites: dict[tuple[Path, int], str]) -> None:
        self.by_name: dict[str, tuple[Any, str]] = {}
        self.imports: list[str] = []
        self.rewrites = rewrites

    def add(self, func: Any) -> str:
        text, imports = _function_text(func)
        code = func.__code__
        text = self.rewrites.get(
            (Path(code.co_filename).resolve(), code.co_firstlineno), text
        )
        name = func.__name__
        if name in self.by_name:
            if self.by_name[name][0].__code__ != func.__code__:
                raise _Uncopyable(
                    "has the name of a different function another model's deck defines"
                )
            return name
        self.by_name[name] = (func, text)
        for line in imports:
            if line not in self.imports:
                self.imports.append(line)
        return name

    def preamble(self) -> str:
        parts = []
        if self.imports:
            parts.append("\n".join(sorted(self.imports)))
        parts += [text for _, text in self.by_name.values()]
        return "\n\n\n".join(parts) + "\n\n\n" if parts else ""


def _group_v1(group: Group, members: list[_Member]) -> V1Deck:
    leaves = []
    for m in members:
        for leaf in m.c.v1:
            leaves.append(
                Leaf((m.label,) + leaf.keys, leaf.value, leaf.source, leaf.line)
            )
    for key, value in group.extra.items():
        leaves.append(Leaf((GROUP, key), value, None, None))
    return V1Deck(leaves, {m.label: m.c.v1.evaluated for m in members})


def _member_paths(m: _Member, v1_path: V1Path) -> V1Path:
    return (norm_key(m.label),) + v1_path


def _read_members(
    group: Group, ignore_legacy_errors: bool, output_path: Path
) -> list[_Member]:
    from . import source
    from .convert import (
        _config_version,
        _convert_one,
        _Run,
        default_output,
        in_deck_dir,
        mesh_value,
    )

    members = []
    for pair in group.pairs:
        with in_deck_dir(pair.deck):
            v1, deck_file = source.read_deck(pair.deck)
        if _config_version(v1.evaluated) == 2:
            raise source.DeckReadError(
                f"{pair.deck.name}, listed in {group.path.name}, is already config version 2"
            )
        mesh, mesh_file = _mesh(group, pair)
        decks = deck_chain(pair.deck, deck_file)
        run = _Run(
            label=pair.label,
            mesh=mesh,
            mesh_file=mesh_file,
            ignore_legacy_errors=ignore_legacy_errors,
            root_dir=group.path.resolve().parent,
            mesh_name=mesh_value(pair.mesh, output_path, group.path.parent),
            standalone=False,
            member_decks=tuple(decks),
        )
        result = _convert_one(run, pair.deck, v1, deck_file, default_output(pair.deck))
        members.append(_Member(pair, result.conversion, decks))
    return members


def _mesh(group: Group, pair: Pair) -> tuple[Optional[MeshInfo], Optional[Path]]:
    path = group.path.parent / pair.mesh
    if not path.is_file():
        return None, None
    try:
        return read_mesh(path), path
    except MeshReadError:
        return None, None


def _bc_types(m: _Member) -> set[str]:
    return {
        str(m.c.v1.value(label, "type", default="")).lower() for label in bc_labels(m.c)
    }


def _resolve_per_mesh(m: _Member) -> dict[int, str]:
    """The decisions a member parked for want of a boundary its own mesh lacks.

    v1 applied each of :data:`PER_MESH` at the boundaries of that type on each
    deck's own mesh, so on a mesh without one it did nothing.
    """
    types = _bc_types(m)
    resolved = {}
    for d in m.c.decisions:
        if not d.v1_paths or d.v2_path[0] != UNCONVERTED:
            continue
        for prefix, bc_type, missing, what in PER_MESH:
            if bc_type not in types and all(
                p[: len(prefix)] == prefix for p in d.v1_paths
            ):
                m.flat.pop(d.v2_path, None)
                resolved[d.id] = (
                    f"{m.label} has no {missing}, and v1 applied {what} at the "
                    f"{bc_type} boundaries of each deck's own mesh"
                )
    return resolved


class _Builder:
    """Builds the group's :class:`Conversion` from the merged tree."""

    def __init__(self, group: Group, members: list[_Member], output: Path) -> None:
        self.group = group
        self.members = members
        self.output = output
        self.merge = _Merge(members)
        self.functions = _Functions(
            {(r.file, r.line): r.text for m in members for r in m.c.rewrites}
        )
        self.resolved: dict[tuple[str, int], str] = {}
        self._ask_for: dict[tuple[str, int], _Ask] = {}

    def build(self) -> Conversion:
        for m in self.members:
            _flatten(m)
            for decision_id, reason in _resolve_per_mesh(m).items():
                self.resolved[(m.label, decision_id)] = reason
        self.merge.run()
        tree = self.merge.tree
        asks = list(self.merge.asks)
        for ask in asks:
            for m in self.members:
                ask.v1_paths += self._leaves_at(m, ask.path)
        asks += self._functions(tree)
        asks += self._member_asks(tree)
        asks += self._extra(tree)

        gv1 = _group_v1(self.group, self.members)
        gc = Conversion(gv1, "")
        for note in self._notes():
            gc.notes.append(note)
        order = self._order(tree)
        asks.sort(key=lambda a: order.get(a.path, len(order)))
        placeholders: dict[int, Placeholder] = {}
        for ask in asks:
            leaves = [gv1.leaf(*p) for p in ask.v1_paths]
            leaves = [lf for lf in leaves if lf is not None and not gc.claimed(lf)]
            value = None if ask.value is None else Value(ask.value)
            placeholders[id(ask)] = gc.decide(
                leaves, ask.path, ask.question, ask.options, ask.details, value=value
            )
        for path, item in tree.items():
            if path in gc.v2:
                continue
            gc.set(path, self._v2_value(item, placeholders))
        gc.v2 = {
            p: gc.v2[p] for p in sorted(gc.v2, key=lambda p: order.get(p, len(order)))
        }
        self._dispositions(gc, gv1)
        return gc

    def _order(self, tree: dict) -> dict[V2Path, int]:
        from .convert import _top_level_rank

        index = {m.label: n for n, m in enumerate(self.members)}
        ranked = sorted(
            enumerate(tree),
            key=lambda e: (
                _top_level_rank(e[1]),
                index.get(e[1][1], 0) if e[1][0] == "model" else 0,
                e[1][0] == "model" and len(e[1]) > 2 and e[1][2] in OVERRIDABLE,
                e[0],
            ),
        )
        return {path: n for n, (_, path) in enumerate(ranked)}

    def _v2_value(self, item: Any, placeholders: dict) -> Any:
        if isinstance(item, _Ask):
            return placeholders[id(item)]
        if isinstance(item, _Pending):
            return placeholders[id(self._ask_for[(item.label, item.id)])]
        if isinstance(item, _Function):
            return Expr(item.name, item.func)
        return Value(_plain(item.value))

    def _functions(self, tree: dict) -> list[_Ask]:
        asks: list[_Ask] = []
        seen: list[tuple[Any, _Ask, list[str]]] = []
        for path, item in tree.items():
            if not isinstance(item, _Plain):
                continue
            try:
                _plain(item.value)
                continue
            except _NotPlain:
                pass
            leaves = [p for m in self.members for p in self._leaves_at(m, path)]
            match = next((e for e in seen if _same_behaviour(e[0], item.value)), None)
            if match is not None:
                match[1].v1_paths += leaves
                if item.label not in match[2]:
                    match[2].append(item.label)
                tree[path] = match[1]
                continue
            text = _original_text(item)
            if callable(item.value):
                try:
                    name = self.functions.add(item.value)
                    tree[path] = _Function(name, item.value)
                    continue
                except _Uncopyable as e:
                    reason = f"the function {text}, which {e}"
            else:
                reason = f"{text}, which cannot be written as a value"
            ask = _Ask(path=path, question=text, details=reason, v1_paths=leaves)
            seen.append((item.value, ask, [item.label]))
            tree[path] = ask
            asks.append(ask)
        for _, ask, labels in seen:
            text, reason = ask.question, ask.details
            deck = self._member(labels[0]).pair.deck.name
            if len(labels) == 1:
                ask.question = f"Which function in {self.output.name} replaces {labels[0]}'s {text}?"
                said = f"{deck} sets it to {reason}."
            else:
                ask.question = f"Which function in {self.output.name} replaces {text}?"
                said = f"{listed(labels)} set it to {reason}, read from {deck} first."
            ask.details = (
                f"{said} The new deck holds every model, so copy the function into "
                f"{self.output.name} and write its name wherever this item is marked."
            )
        return asks

    def _member(self, label: str) -> _Member:
        return next(m for m in self.members if m.label == label)

    def _leaves_at(self, m: _Member, group_path: V2Path) -> list[V1Path]:
        out = []
        for d in m.c.dispositions:
            target = self.merge.group_path(m.label, d.v2_path)
            if target is not None and (
                target[: len(group_path)] == group_path
                or group_path[: len(target)] == target
            ):
                out.append(_member_paths(m, d.v1_path))
        return out

    def _conflict_at(self, path: V2Path) -> Optional[_Ask]:
        return next((a for a in self.merge.asks if path[: len(a.path)] == a.path), None)

    def _member_asks(self, tree: dict) -> list[_Ask]:
        by_key: dict[tuple, _Ask] = {}
        asks = []
        for m in self.members:
            for d in m.c.decisions:
                if (m.label, d.id) in self.resolved:
                    continue
                paths = [_member_paths(m, p) for p in d.v1_paths]
                target = self.merge.group_path(m.label, d.v2_path)
                if target is None:
                    raise ConversionError(
                        f"{m.label}'s decision at {' > '.join(d.v2_path)} has no place in the group deck"
                    )
                absorbed = self.merge.absorbed.get(
                    (m.label, d.id)
                ) or self._conflict_at(target)
                if absorbed is not None:
                    absorbed.v1_paths += paths
                    absorbed.details += f" {m.label} also left it open: {d.question}"
                    self._ask_for[(m.label, d.id)] = absorbed
                    continue
                func = _function_at(m, d)
                key = (
                    (func.__code__, d.question)
                    if func is not None
                    else (target, d.question, d.options)
                )
                if key in by_key:
                    by_key[key].v1_paths += paths
                    self._ask_for[(m.label, d.id)] = by_key[key]
                    continue
                item = tree.get(target)
                value = None
                if isinstance(item, _Plain):
                    value = _plain(item.value)
                details = d.details
                if func is not None:
                    details += (
                        f" The new deck holds every model, so copy the function into "
                        f"{self.output.name} with that def line, and write its name "
                        "wherever this item is marked."
                    )
                ask = _Ask(
                    path=target,
                    question=d.question,
                    options=d.options,
                    details=details,
                    v1_paths=paths,
                    value=value,
                )
                by_key[key] = ask
                self._ask_for[(m.label, d.id)] = ask
                asks.append(ask)
                if isinstance(item, _Pending):
                    tree[target] = ask
        return asks

    def _extra(self, tree: dict) -> list[_Ask]:
        asks = []
        for key, value in self.group.extra.items():
            name = f"{GROUP} > {key}"
            path = (UNCONVERTED, name)
            try:
                kept = _plain(value)
            except _NotPlain:
                kept = repr(value)
            ask = _Ask(
                path=path,
                question=f"update_input has no rule for '{name}'",
                details=(
                    "It is a run option of the v1 overset group. Move it to its "
                    f"config version 2 home, or delete it. Its v1 value is kept under "
                    f"'{UNCONVERTED} > {name}'."
                ),
                v1_paths=[(GROUP, norm_key(key))],
                value=kept,
            )
            tree[path] = ask
            asks.append(ask)
        return asks

    def _notes(self) -> list[Note]:
        root = self.group.path.resolve().parent
        notes = []
        for m in self.members:
            deck = relative(m.pair.deck, root)
            notes.append(Note(f"Model {m.label} is {deck} on {m.pair.mesh}."))
        notes += [Note(text) for text in self.group.notes]
        notes += self.merge.notes
        for m in self.members:
            for note in m.c.notes:
                paths = tuple(_member_paths(m, p) for p in note.v1_paths)
                notes.append(Note(f"{m.label}: {note.text}", paths))
        return notes

    def _dispositions(self, gc: Conversion, gv1: V1Deck) -> None:
        for m in self.members:
            for d in m.c.dispositions:
                path = _member_paths(m, d.v1_path)
                leaf = gv1.leaf(*path)
                if leaf is None or gc.claimed(leaf):
                    continue
                if (m.label, d.decision_id) in self.resolved:
                    gc.drop(leaf, self.resolved[(m.label, d.decision_id)])
                    continue
                gc.dispositions.append(
                    Disposition(
                        path,
                        d.kind,
                        self.merge.group_path(m.label, d.v2_path),
                        d.reason,
                        None,
                    )
                )


@dataclass(frozen=True)
class _Function:
    name: str
    func: Any


def convert_group(group: Group, ignore_legacy_errors: bool, output_path: Path) -> Any:
    """Convert every deck the group names, and merge them into one deck.

    Returns a :class:`~zutil.update_input.convert.ConversionResult` with no
    bases; ``members`` lists the decks that were read.
    """
    from . import source
    from .convert import (
        ConversionResult,
        ConverterBugError,
        _check_v2,
        report_path_for,
    )

    members = _read_members(group, ignore_legacy_errors, output_path)
    builder = _Builder(group, members, output_path)
    gc = builder.build()

    start, end = group.target
    preamble = builder.functions.preamble()
    text = group.text[:start] + preamble + _STAND_IN + group.text[end:]
    target = (start + len(preamble), start + len(preamble) + len(_STAND_IN))
    deck = source.DeckFile(path=group.path, text=text, style="literal", target=target)
    rendered = source.render_deck(deck, gc.v2)

    errors = any(m.c.legacy_errors for m in members)
    validated = False
    unvalidated = ""
    if not gc.decisions:
        try:
            _check_v2(rendered, group.path, output_path, {})
            validated = True
        except ConverterBugError as e:
            if not errors:
                raise
            gc.note(f"The new deck fails config version 2 validation: {e.errors}")
            unvalidated = (
                "not valid config version 2, as a v1 deck had legacy errors "
                "(see the report)"
            )

    root = group.path.resolve().parent
    report_path = report_path_for(output_path)
    names = report.Names(
        deck=relative(group.path, root),
        output=relative(output_path, root),
        report=relative(report_path, root),
        unvalidated=unvalidated,
    )
    summary = report.summary(gc, names, validated)
    decks: list[Path] = []
    for m in members:
        for path in m.decks:
            if all(path.resolve() != d.resolve() for d in decks):
                decks.append(path)
    return ConversionResult(
        text=rendered,
        report=report.markdown(gc, names, validated),
        summary=summary,
        conversion=gc,
        validated=validated,
        output_path=output_path,
        report_path=report_path,
        deck_path=group.path,
        own_summary=summary,
        members=decks,
    )


__all__ = ["Group", "Pair", "convert_group", "read_group"]
