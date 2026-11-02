"""The ``time marching`` block: time settings, convergence control and multigrid.

The mode follows the pre-v2 solvers: ``scheme > kind: "global timestepping"`` is
global time stepping; otherwise a ``total time`` equal to ``time step`` is steady,
and anything else is dual time stepping.
"""

from __future__ import annotations

from typing import Optional

from zutil.validate.schemas.callables import CFL_RAMP
from zutil.validate.schemas.convergence_schemas import (
    EXPLICIT_SCHEME_NAMES,
    IMPLICIT_SCHEME_NAMES,
    IncompressibleConvergenceControl,
)

from ..model import Conversion, Expr, Leaf, Source, Value
from ._functions import port_function
from .zones import (
    MOTION_SHAPES,
    ROTATING_MOTION,
    convert_motion,
    fluid_zone_path,
    has_overset_boundary,
    motion_path,
    moving_zones,
    rotating_fluid_zone,
)

OWNS = ["time marching"]
ORDER = 30

TM = "time marching"
TS = ("solver", "time settings")
CC = ("solver", "convergence control")
MG = ("solver", "numerical scheme", "multigrid")

GTS = "global timestepping"
DTS = "dual time stepping"
GTS_STAGES = (3, "rk third order tvd")
CLOSE_TO_ONE = 0.05

STEADY_REASON = "steady: one physical step, no time term"
LOCAL_TIMESTEPPING_REASON = (
    "the pseudo-time loop is implied by steady and dual time stepping"
)
INCOMPRESSIBLE_SCHEME_REASON = (
    "v1 had only the coupled incompressible solver, which read the pseudo-time "
    "scheme into a Runge-Kutta object it never used"
)
INCOMPRESSIBLE_CFL_REASON = (
    "the v1 incompressible solver marched without a pseudo-time CFL or "
    "multigrid, so it never used this key"
)
INCOMPRESSIBLE_MOTION_REASON = "the v1 incompressible solver never moved the mesh"
GTS_TIME_TERM_REASON = (
    "global time stepping has no time-derivative order or start cycles, "
    "and the v1 solver ignored it"
)
COARSE_WITHOUT_MULTIGRID_REASON = (
    "cfl coarse acts only on multigrid coarse levels, and the deck runs no multigrid"
)
MULTIGRID_WITHOUT_LEVELS_REASON = (
    "the deck sets no multigrid levels, so v1 ran one level and never read it"
)

MULTIGRID_KEYS = {
    "multigrid": "levels",
    "multigrid cycles": "cycles",
    "multigrid ramp": "multigrid ramp",
    "prolong factor": "prolong factor",
    "prolong transport factor": "prolong transport factor",
}
CFL_SIBLINGS = {
    "cfl transport": "cfl turbulence",
    "cfl coarse": "cfl coarse",
    "cfl viscous factor": "cfl viscous factor",
}
V1_RAMP_SIGNATURE = "(solve_cycle, real_time_cycle, cfl)"


def convert(c: Conversion) -> None:
    if not c.v1.has(TM):
        return
    incompressible = c.v1.incompressible
    if c.v1.value(TM, "scheme", "kind") == GTS and not incompressible:
        _global_timestepping(c)
        _motion(c, unsteady=True)
        return
    unsteady = _physical_time(c)
    _pseudo_time(c, incompressible)
    _motion(c, unsteady)
    _cycles_per_overset_cycle(c)


def _leaf(c: Conversion, *keys: str) -> Optional[Leaf]:
    return c.v1.leaf(TM, *keys)


def _present(*leaves: Optional[Leaf]) -> list[Leaf]:
    return [lf for lf in leaves if lf is not None]


def _text(leaf: Leaf) -> str:
    return leaf.source if leaf.source is not None else repr(leaf.value)


def _global_timestepping(c: Conversion) -> None:
    c.set(TS + ("type",), Value("unsteady"))
    for key in ("total time", "time step"):
        leaf = _leaf(c, "unsteady", key)
        if leaf is not None:
            c.move(leaf, TS + (key,))
    for leaf in _present(_leaf(c, "unsteady", "order"), _leaf(c, "unsteady", "start")):
        c.drop(leaf, GTS_TIME_TERM_REASON)

    kind = _leaf(c, "scheme", "kind")
    name = _leaf(c, "scheme", "name")
    stage = _leaf(c, "scheme", "stage")
    name_value = name.value if name is not None else "runge kutta"
    stage_value = stage.value if stage is not None else None
    if name_value == "runge kutta" and stage_value in GTS_STAGES:
        c.move(kind, TS + ("kind",))
        for leaf, key in ((name, "name"), (stage, "stage")):
            if leaf is not None:
                c.move(leaf, TS + ("scheme", key))
        return

    marched = f"name {name_value!r}" + (
        f", stage {stage_value!r}" if stage is not None else ", no stage (5-stage)"
    )
    placeholder = c.decide(
        _present(kind, name, stage),
        TS + ("kind",),
        "v2 global time stepping marches only with third-order Runge-Kutta. "
        "Keep global time stepping with that scheme, or use dual time stepping?",
        options=(GTS, DTS),
        details=(
            f"The v1 deck marched physical time with the scheme {marched}. "
            f"For {GTS!r}, set time settings > scheme to "
            '{"name": "runge kutta", "stage": "rk third order tvd"}; this changes '
            f"the time integration. For {DTS!r}, remove time settings > scheme, put "
            "the v1 scheme in convergence control > scheme, and set convergence "
            "control > cycles to the pseudo-time cycles per step."
        ),
    )
    c.set(TS + ("scheme",), placeholder)


def _physical_time(c: Conversion) -> bool:
    """Write ``time settings`` for a steady or dual time stepping deck; true when unsteady."""
    total = _leaf(c, "unsteady", "total time")
    step = _leaf(c, "unsteady", "time step")
    order = _leaf(c, "unsteady", "order")
    start = _leaf(c, "unsteady", "start")
    total_value = c.v1.value(TM, "unsteady", "total time")
    step_value = c.v1.value(TM, "unsteady", "time step")

    if total_value == step_value:
        c.set(TS + ("type",), Value("steady"))
        for leaf in _present(total, step, order, start):
            c.drop(leaf, STEADY_REASON)
        return False

    ratio = _ratio(total_value, step_value)
    if ratio is not None and abs(ratio - 1.0) <= CLOSE_TO_ONE:
        c.decide(
            [total, step],
            TS + ("type",),
            f"total time / time step is {ratio:.6g}, close to 1 but not equal. "
            "Is this a steady run or a short unsteady one?",
            options=("steady", "unsteady"),
            details=(
                "v1 ran steady only when total time exactly equalled time step, so "
                f"this deck ran {int(ratio)} dual time step(s). For 'steady', set "
                'time settings to {"type": "steady"} and delete its other keys. '
                "For 'unsteady', keep the dual time stepping settings as written."
            ),
        )
        c.set(TS + ("total time",), Source(total))
        c.set(TS + ("time step",), Source(step))
    else:
        c.set(TS + ("type",), Value("unsteady"))
        for leaf, key in ((total, "total time"), (step, "time step")):
            if leaf is not None:
                c.move(leaf, TS + (key,))
    c.set(TS + ("kind",), Value(DTS))
    for leaf, key in ((order, "order"), (start, "start")):
        if leaf is not None:
            c.move(leaf, TS + (key,))
    return True


def _ratio(total: object, step: object) -> Optional[float]:
    numbers = (int, float)
    if isinstance(total, numbers) and isinstance(step, numbers) and step != 0:
        return total / step
    return None


def _pseudo_time(c: Conversion, incompressible: bool) -> None:
    cycles = _leaf(c, "cycles")
    if cycles is not None:
        c.move(cycles, CC + ("cycles",))

    if incompressible:
        _incompressible_scheme(c)
        _incompressible_cfl_and_multigrid(c)
        return

    name = _leaf(c, "scheme", "name")
    if name is not None and name.value in EXPLICIT_SCHEME_NAMES:
        if has_overset_boundary(c):
            _explicit_overset_scheme(c, name)
    for key in ("name", "stage"):
        leaf = _leaf(c, "scheme", key)
        if leaf is not None and not c.claimed(leaf):
            c.move(leaf, CC + ("scheme", key))
    kind = _leaf(c, "scheme", "kind")
    if kind is not None and kind.value == "local timestepping":
        c.drop(kind, LOCAL_TIMESTEPPING_REASON)

    multigrid = _multigrid(c)
    _cfl(c, multigrid)


def _explicit_overset_scheme(c: Conversion, name: Leaf) -> None:
    options = tuple(sorted(IMPLICIT_SCHEME_NAMES))
    c.decide(
        [name],
        CC + ("scheme", "name"),
        f"v2 runs overset meshes only with an implicit pseudo-time scheme, and "
        f"this deck marches with {name.value!r}. Which implicit scheme should it use?",
        options=options,
        details=(
            f"The v1 deck set {' > '.join(name.keys)} = {name.value!r} on a mesh "
            "with an overset boundary. The implicit schemes do not take multigrid, "
            "so remove numerical scheme > multigrid with the change."
        ),
    )


def _incompressible_scheme(c: Conversion) -> None:
    leaves = c.v1.leaves_under(TM, "scheme")
    if not leaves:
        return
    for leaf in leaves:
        if leaf.path[-1] in ("name", "stage", "class"):
            c.drop(leaf, INCOMPRESSIBLE_SCHEME_REASON)
        elif leaf.path[-1] == "kind" and leaf.value == "local timestepping":
            c.drop(leaf, LOCAL_TIMESTEPPING_REASON)
    if _v2_incompressible_default_scheme() == "coupled":
        c.note(
            "convergence control > scheme > name is not written: the v2 default "
            "is 'coupled', the solver every v1 incompressible deck ran",
            (TM, "scheme"),
        )
    else:
        c.set(CC + ("scheme", "name"), Value("coupled"))


def _incompressible_cfl_and_multigrid(c: Conversion) -> None:
    keys = ("cfl", *CFL_SIBLINGS, "cfl ramp factor", *MULTIGRID_KEYS)
    for key in keys:
        for leaf in c.v1.leaves_under(TM, key):
            c.drop(leaf, INCOMPRESSIBLE_CFL_REASON)


def _v2_incompressible_default_scheme() -> str:
    field = IncompressibleConvergenceControl.model_fields["scheme"]
    return field.default_factory().name


def _multigrid(c: Conversion) -> bool:
    """Move the multigrid keys; true when ``numerical scheme > multigrid`` is written."""
    leaves = {key: _leaf(c, key) for key in MULTIGRID_KEYS}
    if not any(leaves.values()):
        return False
    if c.v1.value(TM, "scheme", "name") in IMPLICIT_SCHEME_NAMES:
        return False
    if leaves["multigrid"] is None:
        for leaf in _present(*leaves.values()):
            c.drop(leaf, MULTIGRID_WITHOUT_LEVELS_REASON)
        return False
    for key, leaf in leaves.items():
        if leaf is not None:
            c.move(leaf, MG + (MULTIGRID_KEYS[key],))
    return True


def _cfl(c: Conversion, multigrid: bool) -> None:
    cfl = _leaf(c, "cfl")
    siblings = {key: _leaf(c, key) for key in CFL_SIBLINGS}
    if siblings["cfl coarse"] is not None and not multigrid:
        c.drop(siblings.pop("cfl coarse"), COARSE_WITHOUT_MULTIGRID_REASON)
    siblings = {key: leaf for key, leaf in siblings.items() if leaf is not None}
    growth = _leaf(c, "cfl ramp factor", "growth")
    initial = _leaf(c, "cfl ramp factor", "initial")
    ramp_factor = _present(growth, initial)
    ramp_func = _leaf(c, "ramp func")

    if not (siblings or ramp_factor or ramp_func):
        if cfl is not None:
            c.move(cfl, CC + ("cfl",))
        return

    base = CC + ("cfl",)
    if cfl is not None:
        c.move(cfl, base + ("cfl",))
    else:
        c.set(base + ("cfl",), Value(1.0))
        c.note(
            "time marching > cfl is not set, and its siblings need a cfl dict, "
            "so the v1 default cfl: 1.0 is written",
            (TM, "cfl"),
        )
    for key, leaf in siblings.items():
        c.move(leaf, base + (CFL_SIBLINGS[key],))
    _cfl_ramp(c, base + ("cfl ramp",), ramp_func, growth, initial)


def _cfl_ramp(
    c: Conversion,
    path: tuple[str, ...],
    ramp_func: Optional[Leaf],
    growth: Optional[Leaf],
    initial: Optional[Leaf],
) -> None:
    ramp_factor = _present(growth, initial)
    if ramp_func is not None and ramp_factor:
        c.decide(
            [ramp_func, *ramp_factor],
            path,
            "The deck sets both 'ramp func' and 'cfl ramp factor', and v2 takes one "
            "'cfl ramp'. Which should the run use?",
            details=(
                f"ramp func is {_text(ramp_func)}, called in v1 as {V1_RAMP_SIGNATURE} "
                f"and in v2 as {CFL_RAMP.call('')}. cfl ramp factor becomes "
                '[{"type": "exponential", "factor": <growth>, "initial": <initial>}].'
            ),
        )
        return
    if ramp_func is not None:
        port_function(c, ramp_func, path, CFL_RAMP)
        return
    if growth is None or initial is None:
        return
    text = (
        '[{"type": "exponential", '
        f'"factor": {_text(growth)}, "initial": {_text(initial)}}}]'
    )
    value = [{"type": "exponential", "factor": growth.value, "initial": initial.value}]
    c.fold([growth, initial], path, Expr(text, value))


def _motion(c: Conversion, unsteady: bool) -> None:
    """Pattern M1: a v1 deck-wide ``moving mesh`` makes one model's mesh move."""
    leaf = _leaf(c, "moving mesh")
    if leaf is None:
        return
    if not leaf.value:
        c.drop(leaf, "false is the v2 default")
        return
    if c.v1.incompressible:
        c.drop(leaf, INCOMPRESSIBLE_MOTION_REASON)
        return
    path = motion_path(c)
    moving = moving_zones(c)
    if len(moving) > 1:
        zone_leaves = [
            lf for key in moving for lf in c.v1.leaves_under(key) if not c.claimed(lf)
        ]
        c.decide(
            [leaf, *zone_leaves],
            path,
            f"v1 neither moved mesh {c.model} nor applied a frame to it: its solver "
            "refuses 'moving mesh' with more than one moving zone on a mesh "
            f"({', '.join(moving)}), logs an error and runs without them. What "
            "should the mesh do?",
            options=(
                "stay still with no frame, as in v1: delete this key",
                "turn as a whole: give the model 'transforms > motion'",
                "turn in parts on a still mesh: one 'rotating' fluid zone per "
                "region, on its 'cell zones'",
            ),
            details=f"A moving whole-mesh motion is {MOTION_SHAPES}.",
        )
    elif not moving:
        c.decide(
            [leaf],
            path,
            "What is the 'motion' of the mesh?",
            details=(
                "v1 set 'moving mesh' once for the deck, and the deck has no "
                "rotating or translating zone. Give the model's 'transforms' a "
                f"'motion', {MOTION_SHAPES}, or delete this key."
            ),
        )
    elif unsteady and has_overset_boundary(c):
        convert_motion(c, moving[0], leaf)
    else:
        _still_mesh(c, leaf, moving[0], unsteady)


def _still_mesh(c: Conversion, leaf: Leaf, key: str, unsteady: bool) -> None:
    if unsteady:
        why = (
            "v2 moves a mesh only with an overset or sliding interface, and this "
            "deck has none."
        )
        change = "add the interface"
    else:
        why = "v2 moves a mesh only in an unsteady run, and this deck is steady."
        change = "make the run unsteady"
    if not rotating_fluid_zone(c, key):
        c.decide(
            [leaf],
            motion_path(c) + ("moving mesh",),
            f"{why} Set 'moving mesh' False to solve the flow in the moving frame, "
            f"or {change}?",
            details=(
                "v1 set 'moving mesh' once for the deck. The draft writes the rest "
                "of the moving zone's 'motion'."
            ),
        )
        return
    c.decide(
        [leaf],
        fluid_zone_path(c, key) + ("moving mesh",),
        f"{why} Keep fluid zone {key} as a rotating frame on its cell zones, with "
        f"the mesh still, or {change} and move the whole mesh?",
        details=(
            "v1 set 'moving mesh' once for the deck and moved the whole mesh "
            f"whatever 'zone' listed. The draft writes {key} as a rotating fluid "
            "zone: to keep it, delete this key. To move the mesh, replace the "
            f"fluid zone with 'transforms > motion', {ROTATING_MOTION}."
        ),
    )


def _cycles_per_overset_cycle(c: Conversion) -> None:
    leaf = _leaf(c, "cycles per overset cycle")
    if leaf is not None and leaf.value == 1:
        c.drop(leaf, "1 is the v1 default, and v2 has no setting for it")
