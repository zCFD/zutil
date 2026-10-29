"""Reference conditions, initialisation and restart (deck patterns C1 to C14).

A plain IC moves under ``solver > reference conditions`` as written, less the
keys its equation set does not take. A derived IC (one with ``reference``) is
inlined into the inflow and outflow BCs whose ``condition`` names it, because
the solver reads ratios from the BC dict, and is kept in reference conditions
only when a farfield, a sponge, another IC or the initialisation reference
still names it. ``reference`` and ``initial`` become ``solver >
initialisation``; the restart keys follow them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ...validate.schemas import callables
from ..model import Conversion, Leaf, V2Path, norm_key
from ._functions import port_function
from ._labels import bc_labels, ic_labels, listed, move_ic_name

OWNS = [
    "ic *",
    "reference",
    "initial",
    "restart",
    "restart casename",
    "restart ignore history",
    "interpolate restart",
    "restart meshname",
    "solution smooth cycles",
]
ORDER = 10

REFERENCE_CONDITIONS: V2Path = ("solver", "reference conditions")
INITIALISATION: V2Path = ("solver", "initialisation")
SOLVER_SETTINGS: V2Path = ("solver", "solver settings")
INITIAL_CONDITIONS: V2Path = INITIALISATION + ("initial conditions",)

VISCOSITY_KEYS = ("reynolds no", "viscosity")
TURBULENCE_KEYS = (
    "turbulence intensity",
    "eddy viscosity ratio",
    "ambient turbulence intensity",
    "ambient eddy viscosity ratio",
)
INVISCID = "inviscid: no viscosity in the reference condition"
LAMINAR = "laminar: no turbulence state"
VISCOSITY_WINS = "v1 read viscosity whenever it was set, so reynolds no had no effect"

REFERENCE_RATIOS = (
    "total pressure ratio",
    "total temperature ratio",
    "static pressure ratio",
)

BC_KEYS: dict[tuple[str, str], dict[str, str]] = {
    ("inflow", "default"): {
        "total pressure ratio": "total pressure ratio",
        "total temperature ratio": "total temperature ratio",
        "static pressure ratio": "static pressure ratio",
        "vector": "direction",
        "mach": "mach",
    },
    ("inflow", "massflow"): {
        "mass flow rate": "mass flow rate",
        "mass flow ratio": "mass flow ratio",
        "total temperature ratio": "total temperature ratio",
        "vector": "direction",
    },
    ("outflow", "pressure"): {
        "static pressure ratio": "static pressure ratio",
    },
    ("outflow", "massflow"): {
        "mass flow rate": "mass flow rate",
        "mass flow ratio": "mass flow ratio",
    },
    ("outflow", "radial pressure gradient"): {
        "static pressure ratio": "static pressure ratio",
    },
}

BC_UNREAD: dict[tuple[str, str], str] = {
    ("outflow", "total temperature ratio"): (
        "the solver never used a total temperature ratio on an outflow"
    ),
}

RESTART_KEYS = (
    "restart",
    "restart casename",
    "restart ignore history",
    "interpolate restart",
    "restart meshname",
)

CALLABLE_SEED = "IC_1"


@dataclass
class _BcUser:
    label: str
    type: str
    kind: str


@dataclass
class _Users:
    """Everything in the v1 deck that names one derived IC."""

    bcs: list[_BcUser] = field(default_factory=list)
    holders: list[str] = field(default_factory=list)
    initial_state: bool = False

    @property
    def keep(self) -> bool:
        return bool(self.holders)


def convert(c: Conversion) -> None:
    equations = c.v1.equation_set
    labels = ic_labels(c)
    derived = [lbl for lbl in labels if _is_derived(c, lbl)]
    initial_ic = _initial_state_ic(c)
    for label in labels:
        if label in derived:
            _convert_derived(c, label, _users(c, label, labels, initial_ic))
        else:
            _convert_plain(c, label, equations)
    _convert_initialisation(c, derived)
    _drop_unused_derived(c, derived)
    _convert_restart(c)


def _child(node: Any, key: str) -> Any:
    if not isinstance(node, dict):
        return None
    for k, v in node.items():
        if norm_key(k) == key:
            return v
    return None


def _names(value: Any, label: str) -> bool:
    return isinstance(value, str) and norm_key(value) == norm_key(label)


def _is_derived(c: Conversion, label: str) -> bool:
    return _child(c.v1.value(label), "reference") is not None


def _initial_state_ic(c: Conversion) -> Optional[str]:
    """The IC the domain starts from, when it is named rather than computed."""
    initial = c.v1.value("initial")
    if isinstance(initial, str):
        return initial
    if isinstance(initial, dict):
        if _child(initial, "func") is not None:
            return None
        return _child(initial, "name")
    reference = c.v1.value("reference")
    return reference if isinstance(reference, str) else None


def _users(
    c: Conversion, label: str, labels: list[str], initial_ic: Optional[str]
) -> _Users:
    users = _Users()
    for bc in bc_labels(c):
        block = c.v1.value(bc)
        if not _names(_child(block, "condition"), label):
            continue
        bc_type = str(_child(block, "type") or "").lower()
        if bc_type in ("inflow", "outflow"):
            users.bcs.append(
                _BcUser(bc, bc_type, str(_child(block, "kind") or "").lower())
            )
        else:
            users.holders.append(bc)
    if _names(c.v1.value("sponge layer damping", "condition"), label):
        users.holders.append("sponge layer damping")
    for other in labels:
        if other != label and _names(_child(c.v1.value(other), "reference"), label):
            users.holders.append(other)
    if c.v1.has("initial") and _names(c.v1.value("reference"), label):
        users.holders.append("the initialisation reference")
    users.initial_state = initial_ic is not None and _names(initial_ic, label)
    return users


def _convert_plain(c: Conversion, label: str, equations: str) -> None:
    prefix = REFERENCE_CONDITIONS + (label,)
    for leaf in c.v1.leaves_under(label):
        rel = leaf.path[1:]
        top = rel[0] if rel else ""
        if not rel and callable(leaf.value):
            _convert_callable(c, label, leaf)
        elif (
            len(rel) == 1
            and equations == "euler"
            and top in VISCOSITY_KEYS + TURBULENCE_KEYS
        ):
            c.drop(leaf, INVISCID)
        elif len(rel) == 1 and equations == "viscous" and top in TURBULENCE_KEYS:
            c.drop(leaf, LAMINAR)
        elif rel == ("reynolds no",) and c.v1.has(label, "viscosity"):
            c.drop(leaf, VISCOSITY_WINS)
        elif rel == ("location",):
            continue
        elif rel == ("profile", "abl", "z0"):
            c.move(
                leaf,
                prefix + ("profile", "abl", "ground level"),
                respelled=True,
                reason="v2 names the ABL profile's z0 'ground level'",
            )
        else:
            c.move(leaf, prefix + rel)


def _convert_callable(c: Conversion, label: str, leaf: Leaf) -> None:
    name = _text(leaf)
    farfields = [
        bc
        for bc in bc_labels(c)
        if _names(_child(c.v1.value(bc), "condition"), label)
        and str(_child(c.v1.value(bc), "type") or "").lower() == "farfield"
    ]
    where = listed(farfields) if farfields else "the farfield"
    c.decide(
        [leaf],
        REFERENCE_CONDITIONS + (label,),
        f"{label} is computed by {name}, and a v2 reference condition is a fixed "
        f"state. Which state should {label} hold?",
        details=(
            f"Write {label} as a reference condition with the vector the run "
            f"starts from. To trim alpha to a lift coefficient, give {where} "
            f'a target, such as "target": {{"alpha": {{"cl": ...}}}}; the solver '
            f"turns alpha until cl is met, then holds it. {name} and the code "
            f"it uses no longer run: delete them. A force-report transform that "
            f"rotated by its angle stays at the starting angle; the target "
            f"computes cl itself."
        ),
    )


def _bc_path(c: Conversion, bc: str, key: str) -> V2Path:
    return ("model", c.model, "boundary conditions", bc, key)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _accepts(key: str, value: Any) -> bool:
    if key == "vector":
        return (
            isinstance(value, (list, tuple))
            and len(value) == 3
            and all(_is_number(v) for v in value)
        )
    return _is_number(value)


def _bc_problem(user: _BcUser, key: str, leaf: Leaf) -> Optional[str]:
    """Why ``key`` cannot be written on this BC, or ``None`` when it can."""
    v2_key = BC_KEYS.get((user.type, user.kind), {}).get(key)
    if v2_key is None:
        return f"v2 has no '{key}' on a '{user.kind}' {user.type} boundary condition"
    if not _accepts(key, leaf.value):
        return f"v2 takes a number for '{v2_key}', and the deck gives {_text(leaf)}"
    return None


def _text(leaf: Leaf) -> str:
    return leaf.source if leaf.source is not None else repr(leaf.value)


def _route(c: Conversion, label: str, leaf: Leaf, users: _Users) -> None:
    key = " ".join(leaf.path[1:])
    accepted: list[V2Path] = []
    rejected: list[tuple[V2Path, str]] = []
    unread: list[tuple[str, str]] = []
    for user in users.bcs:
        reason = BC_UNREAD.get((user.type, key))
        if reason is not None:
            unread.append((user.label, reason))
            continue
        problem = _bc_problem(user, key, leaf)
        if problem is None:
            accepted.append(
                _bc_path(c, user.label, BC_KEYS[(user.type, user.kind)][key])
            )
        else:
            rejected.append((_bc_path(c, user.label, key), problem))

    reference_path = REFERENCE_CONDITIONS + (label,) + leaf.path[1:]
    if users.keep and key in REFERENCE_RATIOS:
        c.move(leaf, reference_path)
    for path, problem in rejected:
        c.decide(
            [] if c.claimed(leaf) else [leaf],
            path,
            f"{label} > {key} has no place on {path[3]}: {problem}",
            details=f"{label} = {_derived_text(c, label)}",
        )
    for path in accepted:
        if c.claimed(leaf):
            c.also(leaf, path)
        else:
            c.move(leaf, path)
        if path[-1] != key:
            c.note(
                f"{label} > {key} is written as '{path[-1]}' on {path[3]}", leaf.path
            )
    if users.keep and not c.claimed(leaf):
        c.decide(
            [leaf],
            reference_path,
            f"v2 reference conditions have no '{key}', and {label} stays one "
            f"because {listed(users.holders)} refers to it",
            details=f"{label} = {_derived_text(c, label)}",
        )
    for bc, reason in unread:
        if c.claimed(leaf):
            c.note(f"{label} > {key} is not written to {bc}: {reason}", leaf.path)
        else:
            c.drop(leaf, reason)


def _convert_derived(c: Conversion, label: str, users: _Users) -> None:
    reference = None
    for leaf in c.v1.leaves_under(label):
        if leaf.path[1:] == ("reference",):
            reference = leaf
        elif users.bcs or users.keep:
            _route(c, label, leaf, users)
    if len(users.bcs) > 1:
        c.note(
            f"{label} is inlined into {listed([u.label for u in users.bcs])}",
            (norm_key(label),),
        )
    if reference is None:
        return
    if users.keep:
        move_ic_name(c, reference, REFERENCE_CONDITIONS + (label, "reference"))
    elif users.bcs and not users.initial_state:
        c.drop(
            reference,
            f"inlined into {listed([u.label for u in users.bcs])}, "
            f"whose reference is {reference.value}",
        )


def _drop_unused_derived(c: Conversion, derived: list[str]) -> None:
    for label in derived:
        for leaf in c.v1.leaves_under(label):
            if not c.claimed(leaf):
                c.drop(leaf, f"nothing refers to {label}")


def _derived_text(c: Conversion, label: str) -> str:
    parts = [
        f"{' '.join(leaf.path[1:])} {_text(leaf)}" for leaf in c.v1.leaves_under(label)
    ]
    return ", ".join(parts)


def _derived_label(value: Any, derived: list[str]) -> Optional[str]:
    for label in derived:
        if _names(value, label):
            return label
    return None


def _initial_conditions(c: Conversion, leaf: Leaf, derived: list[str]) -> None:
    label = _derived_label(leaf.value, derived)
    if label is None:
        move_ic_name(c, leaf, INITIAL_CONDITIONS)
        return
    reference = _child(c.v1.value(label), "reference")
    leaves = [leaf] + [lf for lf in c.v1.leaves_under(label) if not c.claimed(lf)]
    c.decide(
        leaves,
        INITIAL_CONDITIONS,
        f"{label} is derived from {reference} and starts the run; "
        "which state should the domain start from?",
        details=(
            f"{label} = {_derived_text(c, label)}. "
            f"Name a reference condition, or give a function that computes the state "
            f"from {reference} and these ratios."
        ),
    )


def _convert_initialisation(c: Conversion, derived: list[str]) -> None:
    reference = c.v1.leaf("reference")
    if not c.v1.has("initial"):
        if reference is not None:
            _initial_conditions(c, reference, derived)
        return
    if reference is not None:
        move_ic_name(c, reference, INITIALISATION + ("reference",))
    initial = c.v1.leaf("initial")
    name = c.v1.leaf("initial", "name")
    func = c.v1.leaf("initial", "func")
    if func is not None:
        _initial_function(c, name, func)
    elif initial is not None:
        _initial_conditions(c, initial, derived)
    elif name is not None:
        _initial_conditions(c, name, derived)


def _initial_function(c: Conversion, name: Optional[Leaf], func: Leaf) -> None:
    if name is None or _names(name.value, CALLABLE_SEED):
        port_function(c, func, INITIAL_CONDITIONS, callables.INITIAL_CONDITIONS)
        if name is not None:
            c.drop(
                name,
                f"v2 seeds an initial function from {CALLABLE_SEED}, as this deck does",
            )
        return
    c.decide(
        [name, func],
        INITIAL_CONDITIONS,
        f"v1 seeded {_text(func)} from {name.value}; v2 seeds an initial "
        f"function from {CALLABLE_SEED}. Which state should it start from?",
        details=(
            f"initial > func = {_text(func)}, initial > name = {_text(name)}. "
            f"Rename {name.value} to {CALLABLE_SEED}, or write a function that "
            f"does not rely on its starting state."
        ),
    )


def _convert_restart(c: Conversion) -> None:
    for key in RESTART_KEYS:
        leaf = c.v1.leaf(key)
        if leaf is not None:
            c.move(leaf, INITIALISATION + (key,))
    smooth = c.v1.leaf("solution smooth cycles")
    if smooth is not None:
        c.move(smooth, SOLVER_SETTINGS + ("solution smooth cycles",))
