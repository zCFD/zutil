"""Shared types for converting a v1 control deck to config version 2.

A v1 deck is read into a :class:`V1Deck`: one :class:`Leaf` per non-dict value,
keyed by its normalised key path. Rules read the deck and record, through a
:class:`Conversion`, where every leaf goes in the v2 tree and why.

v1 paths are tuples of normalised keys (see :func:`norm_key`). v2 paths are
tuples of the keys written into the output deck: lower case with spaces for
schema keys (``("solver", "time settings", "type")``), and user labels such as
``"IC_1"`` or ``"BC_3"`` exactly as the v1 deck spelt them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Union

V1Path = tuple[str, ...]
V2Path = tuple[str, ...]

UNCONVERTED = "unconverted v1 keys"

INCOMPRESSIBLE_EQUATIONS = ("incomp viscous", "incomp rans", "incomp les")


def norm_key(key: Any) -> str:
    """Fold a v1 key to lower case with single spaces in place of underscores."""
    return " ".join(str(key).replace("_", " ").lower().split())


def norm_path(*keys: Any) -> V1Path:
    return tuple(norm_key(k) for k in keys)


@dataclass(frozen=True)
class Leaf:
    """One non-dict value of the evaluated v1 ``parameters`` dict.

    ``keys`` are the keys as the deck spelt them. ``source`` is the value's
    source text when the deck wrote it as a literal entry of the parameters
    dict, and ``None`` when the value has no literal source (it came from a
    ``**`` merge, a function, a loop). ``line`` is the source line, if known.
    """

    keys: tuple[str, ...]
    value: Any
    source: Optional[str] = None
    line: Optional[int] = None

    @property
    def path(self) -> V1Path:
        return norm_path(*self.keys)


@dataclass(frozen=True)
class Source:
    """Write the v1 leaf's own source text, or its value when it has none."""

    leaf: Leaf


@dataclass(frozen=True)
class Value:
    """Write a Python value through ``repr``."""

    value: Any


@dataclass(frozen=True)
class Expr:
    """Write ``text`` as a Python expression that evaluates to ``value``."""

    text: str
    value: Any


@dataclass(frozen=True)
class Placeholder:
    """A DECIDE marker: a string starting ``DECIDE:`` that names an open decision.

    Most keys fail v2 validation on it, but a free-text key such as ``mesh``
    does not, so a draft is finished only when no ``DECIDE`` string is left.
    """

    decision_id: int
    text: str


V2Value = Union[Source, Value, Expr, Placeholder]

DispositionKind = Literal["moved", "respelled", "dropped", "decide"]


@dataclass(frozen=True)
class Disposition:
    """What happened to one v1 leaf."""

    v1_path: V1Path
    kind: DispositionKind
    v2_path: Optional[V2Path] = None
    reason: str = ""
    decision_id: Optional[int] = None


@dataclass(frozen=True)
class Decision:
    """A choice the engineer must make before the deck can run."""

    id: int
    v2_path: V2Path
    question: str
    options: tuple[str, ...] = ()
    details: str = ""
    v1_paths: tuple[V1Path, ...] = ()


@dataclass(frozen=True)
class FunctionRewrite:
    """New source for a function a deck defines, so v2 can call it.

    ``start`` and ``end`` are character offsets of the ``def`` statement in the
    text of ``file``, and ``text`` replaces that span. ``old_def`` and
    ``new_def`` are the ``def`` lines before and after, as one line each.
    """

    file: Path
    line: int
    name: str
    start: int
    end: int
    text: str
    old_def: str
    new_def: str


@dataclass(frozen=True)
class Note:
    """A fact about the conversion that changes no key, shown in the report."""

    text: str
    v1_paths: tuple[V1Path, ...] = ()


@dataclass
class MeshInfo:
    """What ``--mesh`` tells the converter.

    ``zones_by_type`` maps a v1 boundary ``ref`` code to the zone ids of that
    type in the mesh. ``cell_zones`` are the mesh's cell zone ids, or None
    when they were not read.
    """

    path: str
    label: str
    zones_by_type: dict[int, list[int]] = field(default_factory=dict)
    cell_zones: Optional[list[int]] = None


class V1Deck:
    """The evaluated v1 ``parameters`` dict, flattened to leaves."""

    def __init__(
        self, leaves: list[Leaf], evaluated: dict, notes: Optional[list[Note]] = None
    ) -> None:
        self.evaluated = evaluated
        self.notes: list[Note] = list(notes or [])
        self._leaves: dict[V1Path, Leaf] = {}
        for leaf in leaves:
            if leaf.path in self._leaves:
                raise ValueError(f"two v1 keys fold to {' > '.join(leaf.path)}")
            self._leaves[leaf.path] = leaf

    def __iter__(self) -> Iterator[Leaf]:
        return iter(self._leaves.values())

    def leaf(self, *keys: Any) -> Optional[Leaf]:
        return self._leaves.get(norm_path(*keys))

    def has(self, *keys: Any) -> bool:
        prefix = norm_path(*keys)
        return any(p[: len(prefix)] == prefix for p in self._leaves)

    def leaves_under(self, *keys: Any) -> list[Leaf]:
        prefix = norm_path(*keys)
        return [lf for p, lf in self._leaves.items() if p[: len(prefix)] == prefix]

    def children(self, *keys: Any) -> list[str]:
        """Child keys of a v1 dict, as the deck spelt them, in deck order."""
        prefix = norm_path(*keys)
        seen: dict[str, str] = {}
        for lf in self._leaves.values():
            if lf.path[: len(prefix)] == prefix and len(lf.path) > len(prefix):
                seen.setdefault(lf.path[len(prefix)], lf.keys[len(prefix)])
        return list(seen.values())

    def value(self, *keys: Any, default: Any = None) -> Any:
        """The evaluated value at a v1 path, a dict for a subtree."""
        node: Any = self.evaluated
        for key in norm_path(*keys):
            if not isinstance(node, dict):
                return default
            match = [k for k in node if norm_key(k) == key]
            if not match:
                return default
            node = node[match[0]]
        return node

    @property
    def equations(self) -> str:
        """The normalised top-level ``equations`` value, empty when absent or not a string."""
        value = self.value("equations")
        return norm_key(value) if isinstance(value, str) else ""

    @property
    def incompressible(self) -> bool:
        """Whether ``equations`` names one of the incompressible equation sets."""
        return self.equations in INCOMPRESSIBLE_EQUATIONS

    @property
    def equation_set(self) -> str:
        """The equation set without the ``incomp`` prefix of the incompressible names."""
        if self.incompressible:
            return self.equations.removeprefix("incomp ")
        return self.equations


class ConversionError(Exception):
    """A rule broke the conversion contract; this is a converter bug."""


class Conversion:
    """The state a rule reads and writes.

    Every v1 leaf must be claimed exactly once, through :meth:`move`,
    :meth:`fold`, :meth:`drop`, :meth:`decide` or :meth:`park`. Every v2 path
    is written at most once. :meth:`finish` ends the conversion.
    """

    def __init__(self, v1: V1Deck, model: str, mesh: Optional[MeshInfo] = None) -> None:
        self.v1 = v1
        self.model = model
        self.mesh = mesh
        self.v2: dict[V2Path, V2Value] = {}
        self.dispositions: list[Disposition] = []
        self.decisions: list[Decision] = []
        self.notes: list[Note] = list(v1.notes)
        self._claimed: set[V1Path] = set()
        self._parked: set[int] = set()
        self.deck_files: tuple[Path, ...] = ()
        self.rewrites: list[FunctionRewrite] = []
        self.legacy_errors: tuple[str, ...] = ()

    def claimed(self, leaf: Leaf) -> bool:
        return leaf.path in self._claimed

    def unclaimed(self) -> list[Leaf]:
        return [lf for lf in self.v1 if lf.path not in self._claimed]

    def _claim(self, leaf: Leaf) -> None:
        if leaf.path in self._claimed:
            raise ConversionError(f"v1 key {' > '.join(leaf.path)} claimed twice")
        self._claimed.add(leaf.path)

    def set(self, v2_path: V2Path, value: V2Value) -> None:
        """Write a v2 value that no single v1 leaf supplies."""
        if v2_path in self.v2:
            raise ConversionError(f"v2 key {' > '.join(v2_path)} written twice")
        for other in self.v2:
            if other[: len(v2_path)] == v2_path or v2_path[: len(other)] == other:
                raise ConversionError(
                    f"v2 key {' > '.join(v2_path)} overlaps {' > '.join(other)}"
                )
        self.v2[v2_path] = value

    def move(
        self,
        leaf: Leaf,
        v2_path: V2Path,
        value: Optional[V2Value] = None,
        respelled: bool = False,
        reason: str = "",
    ) -> None:
        """Send a v1 leaf to ``v2_path``, as its own source unless ``value`` is given."""
        self._claim(leaf)
        self.set(v2_path, value if value is not None else Source(leaf))
        self.dispositions.append(
            Disposition(
                leaf.path, "respelled" if respelled else "moved", v2_path, reason
            )
        )

    def move_tree(self, v1_keys: tuple[str, ...], v2_prefix: V2Path) -> None:
        """Move every leaf under ``v1_keys`` to the same relative path under ``v2_prefix``.

        Relative keys are folded with :func:`norm_key`; user labels below the
        prefix are not expected.
        """
        prefix = norm_path(*v1_keys)
        for leaf in self.v1.leaves_under(*v1_keys):
            if not self.claimed(leaf):
                self.move(leaf, v2_prefix + leaf.path[len(prefix) :])

    def fold(
        self,
        leaves: list[Leaf],
        v2_path: V2Path,
        value: V2Value,
        respelled: bool = False,
        reason: str = "",
    ) -> None:
        """Send several v1 leaves into the one v2 value at ``v2_path``."""
        if not leaves:
            raise ConversionError(f"folding into {' > '.join(v2_path)} needs leaves")
        for leaf in leaves:
            self._claim(leaf)
        self.set(v2_path, value)
        for leaf in leaves:
            self.dispositions.append(
                Disposition(
                    leaf.path, "respelled" if respelled else "moved", v2_path, reason
                )
            )

    def also(
        self, leaf: Leaf, v2_path: V2Path, value: Optional[V2Value] = None
    ) -> None:
        """Write an already claimed v1 leaf to one more v2 path."""
        if not self.claimed(leaf):
            raise ConversionError(
                f"v1 key {' > '.join(leaf.path)} must be claimed before it is copied"
            )
        self.set(v2_path, value if value is not None else Source(leaf))
        self.note(
            f"{' > '.join(leaf.keys)} is also written to {' > '.join(v2_path)}",
            leaf.path,
        )

    def rewrite_at(self, file: Path, line: int) -> Optional[FunctionRewrite]:
        """The rewrite already recorded for the function defined at ``file:line``."""
        return next(
            (r for r in self.rewrites if r.file == file and r.line == line), None
        )

    def rewrite(self, rewrite: FunctionRewrite) -> None:
        """Record new source for a deck function, once for each function."""
        existing = self.rewrite_at(rewrite.file, rewrite.line)
        if existing is None:
            self.rewrites.append(rewrite)
        elif existing != rewrite:
            raise ConversionError(
                f"{rewrite.name} in {rewrite.file.name} is rewritten two different ways"
            )

    def drop(self, leaf: Leaf, reason: str) -> None:
        if not reason:
            raise ConversionError(f"dropping {' > '.join(leaf.path)} needs a reason")
        self._claim(leaf)
        self.dispositions.append(Disposition(leaf.path, "dropped", None, reason))

    def note(self, text: str, v1_path: Optional[V1Path] = None) -> None:
        """Add a note, or add ``v1_path`` to the note that already says ``text``."""
        for i, note in enumerate(self.notes):
            if note.text != text:
                continue
            if v1_path is not None and v1_path not in note.v1_paths:
                self.notes[i] = replace(note, v1_paths=note.v1_paths + (v1_path,))
            return
        self.notes.append(Note(text, (v1_path,) if v1_path is not None else ()))

    def decide(
        self,
        leaves: list[Leaf],
        v2_path: V2Path,
        question: str,
        options: tuple[str, ...] = (),
        details: str = "",
        value: Optional[V2Value] = None,
    ) -> Placeholder:
        """Record a decision, claim ``leaves``, and write at ``v2_path``.

        The placeholder is written unless ``value`` is given, for a decision
        whose unresolved state is shown some other way.
        """
        decision = Decision(
            len(self.decisions) + 1,
            v2_path,
            question,
            options,
            details,
            tuple(lf.path for lf in leaves),
        )
        decision_id = decision.id
        placeholder = _placeholder(decision)
        self.decisions.append(decision)
        for leaf in leaves:
            self._claim(leaf)
            self.dispositions.append(
                Disposition(leaf.path, "decide", v2_path, question, decision_id)
            )
        self.set(v2_path, value if value is not None else placeholder)
        return placeholder

    def park(
        self,
        leaves: list[Leaf],
        name: str,
        question: str,
        options: tuple[str, ...] = (),
        details: str = "",
        value: Optional[V2Value] = None,
    ) -> None:
        """Decide on ``leaves`` whose v2 home is unknown, keeping ``value`` under the unconverted keys.

        ``value`` defaults to the single leaf's own source.
        """
        if value is None:
            if len(leaves) != 1:
                raise ConversionError(f"parking {name} needs a value for its leaves")
            value = Source(leaves[0])
        kept = f"Its v1 value is kept under '{UNCONVERTED} > {name}'."
        self.decide(
            leaves,
            (UNCONVERTED, name),
            question,
            options,
            f"{details} {kept}".strip(),
            value=value,
        )
        self._parked.add(self.decisions[-1].id)

    def finish(self, variant: str = "") -> None:
        """Park every unclaimed leaf, then number the decisions from one.

        For a base deck, ``variant`` names the deck that loads it; the parked
        keys are left to that deck, which converts them from its own full
        parameters.
        """
        for leaf in self.unclaimed():
            name = " > ".join(str(k) for k in leaf.keys)
            self.park(
                [leaf],
                name,
                f"update_input has no rule for '{name}'",
                details="Move it to its config version 2 home, or delete it.",
            )
        if variant:
            self._leave_parked_to(variant)
        self._renumber()

    def _leave_parked_to(self, variant: str) -> None:
        reason = f"left to {variant}, which loads this deck and converts it in full"
        self.v2 = {p: v for p, v in self.v2.items() if p[:1] != (UNCONVERTED,)}
        self.decisions = [d for d in self.decisions if d.id not in self._parked]
        self.dispositions = [
            replace(d, kind="dropped", v2_path=None, reason=reason, decision_id=None)
            if d.decision_id in self._parked
            else d
            for d in self.dispositions
        ]
        self._parked = set()

    def _renumber(self) -> None:
        new_id = {d.id: n for n, d in enumerate(self.decisions, start=1)}
        self.decisions = [replace(d, id=new_id[d.id]) for d in self.decisions]
        by_id = {d.id: d for d in self.decisions}
        self.v2 = {
            p: _placeholder(by_id[new_id[v.decision_id]])
            if isinstance(v, Placeholder) and v.decision_id in new_id
            else v
            for p, v in self.v2.items()
        }
        self.dispositions = [
            replace(d, decision_id=new_id[d.decision_id])
            if d.decision_id is not None
            else d
            for d in self.dispositions
        ]
        self._parked = {new_id[i] for i in self._parked}


def _placeholder(decision: Decision) -> Placeholder:
    if decision.options:
        choice = " or ".join(repr(o) for o in decision.options)
    else:
        choice = decision.question
    return Placeholder(
        decision.id, f"DECIDE: {choice} (see conversion report, item {decision.id})"
    )
