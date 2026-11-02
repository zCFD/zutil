"""Tests for the time rule: patterns T1 to T14 and M1 of the v1-to-v2 conversion doc.

Each toy deck is the smallest v1 ``parameters`` dict that shows one pattern.
The v2 fragment a test expects is checked against the real pydantic schema
where the fragment can stand alone in a compressible deck.
"""

from __future__ import annotations

import copy

import pytest

from zutil.update_input.model import (
    Conversion,
    Expr,
    MeshInfo,
    Placeholder,
    Source,
    V1Deck,
    Value,
)
from zutil.update_input.rules import time as time_rule
from zutil.update_input.tests.engine.fakes import flatten, shocktube_v2
from zutil.validate.schemas.callables import CFL_RAMP

LABEL = "case"
TS = ("solver", "time settings")
CC = ("solver", "convergence control")
MG = ("solver", "numerical scheme", "multigrid")
STEADY_REASON = "steady: one physical step, no time term"


def run(
    tree: dict, sources: dict | None = None, cell_zones: list[int] | None = None
) -> Conversion:
    evaluated = copy.deepcopy(tree)
    mesh = None if cell_zones is None else MeshInfo("m.h5", LABEL, {}, cell_zones)
    c = Conversion(V1Deck(flatten(evaluated, sources), evaluated), LABEL, mesh)
    time_rule.convert(c)
    return c


def tm(**blocks) -> dict:
    return {"time marching": blocks}


def steady_tm(**extra) -> dict:
    return {"unsteady": {"total time": 1.0, "time step": 1.0}, **extra}


def evaluate(value):
    if isinstance(value, Source):
        return value.leaf.value
    if isinstance(value, (Value, Expr)):
        return value.value
    if isinstance(value, Placeholder):
        return value.text
    raise TypeError(value)


def nested(c: Conversion) -> dict:
    root: dict = {}
    for path, value in c.v2.items():
        node = root
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = evaluate(value)
    return root


def under(c: Conversion, *prefix: str) -> dict:
    node = nested(c)
    for key in prefix:
        node = node.get(key, {})
    return node


def disposition(c: Conversion, *v1_path: str):
    matches = [d for d in c.dispositions if d.v1_path == v1_path]
    assert len(matches) == 1, f"{v1_path} has {len(matches)} dispositions"
    return matches[0]


def unclaimed_paths(c: Conversion) -> set:
    return {lf.path for lf in c.unclaimed()}


def validate_compressible(c: Conversion) -> None:
    """Put the rule's solver blocks into a deck that validates and validate it."""
    from zutil.validate import validate_zcfd_input

    deck = shocktube_v2(LABEL)
    deck["config version"] = 2
    deck["model"][LABEL]["mesh"] = "case.h5"
    solver = nested(c)["solver"]
    deck["solver"]["time settings"] = solver["time settings"]
    if "convergence control" in solver:
        deck["solver"]["convergence control"] = solver["convergence control"]
    if "numerical scheme" in solver:
        deck["solver"]["numerical scheme"].update(solver["numerical scheme"])
    validate_zcfd_input(deck)


def test_owns_time_marching_and_runs_third():
    assert time_rule.OWNS == ["time marching"]
    assert time_rule.ORDER == 30


def test_t1_steady_by_the_sentinel():
    c = run(
        tm(
            unsteady={"total time": 1.0, "time step": 1.0},
            scheme={"name": "implicit euler", "stage": 1},
            cfl=50.0,
            cycles=2000,
        )
    )
    assert under(c, *TS) == {"type": "steady"}
    assert under(c, *CC) == {
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 50.0,
        "cycles": 2000,
    }
    for key in ("total time", "time step"):
        d = disposition(c, "time marching", "unsteady", key)
        assert d.kind == "dropped"
        assert d.reason == STEADY_REASON
    assert c.unclaimed() == []
    validate_compressible(c)


def test_t1_steady_drops_order_and_start_with_the_steady_reason():
    c = run(
        tm(
            unsteady={
                "total time": 1.0,
                "time step": 1.0,
                "order": "second",
                "start": 5,
            },
            cycles=100,
        )
    )
    assert under(c, *TS) == {"type": "steady"}
    for key in ("order", "start"):
        d = disposition(c, "time marching", "unsteady", key)
        assert d.kind == "dropped"
        assert d.reason == STEADY_REASON
    validate_compressible(c)


def test_t1_steady_is_decided_on_the_evaluated_values():
    c = run(
        tm(unsteady={"total time": 0.5, "time step": 0.5}, cycles=10),
        {
            ("time marching", "unsteady", "total time"): "dt",
            ("time marching", "unsteady", "time step"): "dt",
        },
    )
    assert under(c, *TS) == {"type": "steady"}


def test_t2_dual_time_stepping_with_order_and_start():
    c = run(
        tm(
            unsteady={
                "total time": 0.2,
                "time step": 1.0e-4,
                "order": "second",
                "start": 100,
            },
            scheme={"name": "implicit euler", "stage": 1, "kind": "local timestepping"},
            cfl=20.0,
            cycles=50,
        )
    )
    assert under(c, *TS) == {
        "type": "unsteady",
        "kind": "dual time stepping",
        "total time": 0.2,
        "time step": 1.0e-4,
        "order": "second",
        "start": 100,
    }
    assert under(c, *CC) == {
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 20.0,
        "cycles": 50,
    }
    assert disposition(c, "time marching", "unsteady", "order").v2_path == TS + (
        "order",
    )
    assert c.unclaimed() == []
    validate_compressible(c)


def test_t2_dual_time_stepping_with_order_and_no_start():
    c = run(tm(unsteady={"total time": 2.0, "time step": 0.01, "order": 2}, cycles=20))
    ts = under(c, *TS)
    assert ts["kind"] == "dual time stepping"
    assert ts["order"] == 2
    assert "start" not in ts
    validate_compressible(c)


def test_t2_moves_the_source_text_of_the_time_keys():
    c = run(
        tm(unsteady={"total time": 0.2, "time step": 1.0e-4}, cycles=20),
        {
            ("time marching", "unsteady", "total time"): "2000 * dt",
            ("time marching", "unsteady", "time step"): "dt",
        },
    )
    value = c.v2[TS + ("time step",)]
    assert isinstance(value, Source)
    assert value.leaf.source == "dt"


GTS_RK3 = {
    "unsteady": {"total time": 0.2, "time step": 1.0e-5},
    "scheme": {
        "name": "runge kutta",
        "stage": "rk third order tvd",
        "kind": "global timestepping",
    },
}


def test_t3_global_timestepping_with_third_order_runge_kutta():
    c = run(tm(**copy.deepcopy(GTS_RK3)))
    assert under(c, *TS) == {
        "type": "unsteady",
        "kind": "global timestepping",
        "total time": 0.2,
        "time step": 1.0e-5,
        "scheme": {"name": "runge kutta", "stage": "rk third order tvd"},
    }
    assert under(c, *CC) == {}
    assert disposition(c, "time marching", "scheme", "kind").v2_path == TS + ("kind",)
    assert c.unclaimed() == []
    assert c.decisions == []
    validate_compressible(c)


def test_t3_global_timestepping_with_total_time_equal_to_time_step_stays_gts():
    tree = copy.deepcopy(GTS_RK3)
    tree["unsteady"] = {"total time": 1.0e-5, "time step": 1.0e-5}
    c = run(tm(**tree))
    ts = under(c, *TS)
    assert ts["type"] == "unsteady"
    assert ts["kind"] == "global timestepping"
    assert ts["total time"] == ts["time step"] == 1.0e-5
    validate_compressible(c)


def test_t4_open_question_3_gts_with_euler_is_a_decision_between_rk3_and_dts():
    c = run(
        tm(
            unsteady={"total time": 0.2, "time step": 1.0e-5},
            scheme={"name": "euler", "kind": "global timestepping"},
        )
    )
    assert len(c.decisions) == 1
    decision = c.decisions[0]
    assert decision.v2_path == TS + ("kind",)
    assert decision.options == ("global timestepping", "dual time stepping")
    assert "runge kutta" in decision.details
    placeholder = c.v2[TS + ("kind",)]
    assert isinstance(placeholder, Placeholder)
    assert c.v2[TS + ("scheme",)] == placeholder
    for key in ("name", "kind"):
        d = disposition(c, "time marching", "scheme", key)
        assert d.kind == "decide"
        assert d.decision_id == decision.id
    assert under(c, *TS)["total time"] == 0.2
    assert c.unclaimed() == []


def test_t4_open_question_3_gts_runge_kutta_without_a_stage_is_a_decision():
    c = run(
        tm(
            unsteady={"total time": 0.2, "time step": 1.0e-5},
            scheme={"name": "runge kutta", "kind": "global timestepping"},
        )
    )
    assert [d.v2_path for d in c.decisions] == [TS + ("kind",)]


def test_t5_ratio_close_to_one_is_a_decision():
    c = run(tm(unsteady={"total time": 1.02, "time step": 1.0}, cycles=100))
    assert len(c.decisions) == 1
    decision = c.decisions[0]
    assert decision.v2_path == TS + ("type",)
    assert decision.options == ("steady", "unsteady")
    for key in ("total time", "time step"):
        assert disposition(c, "time marching", "unsteady", key).kind == "decide"
    ts = under(c, *TS)
    assert ts["kind"] == "dual time stepping"
    assert ts["total time"] == 1.02
    assert c.unclaimed() == []


def test_t5_a_genuine_unsteady_ratio_is_not_a_decision():
    c = run(tm(unsteady={"total time": 1.5, "time step": 1.0}, cycles=100))
    assert c.decisions == []
    assert under(c, *TS)["kind"] == "dual time stepping"


INCOMP = {"equations": "incomp viscous", "incomp viscous": {"order": "second"}}


def test_t6_incompressible_steady_writes_cycles_and_no_scheme():
    c = run(
        {**INCOMP, **tm(unsteady={"total time": 1.0, "time step": 1.0}, cycles=2000)}
    )
    assert under(c, *TS) == {"type": "steady"}
    assert under(c, *CC) == {"cycles": 2000}
    assert not any(p[:3] == CC + ("scheme",) for p in c.v2)
    assert unclaimed_paths(c) == {("equations",), ("incomp viscous", "order")}


def test_t6_incompressible_dual_time_stepping():
    c = run(
        {
            **INCOMP,
            **tm(
                unsteady={
                    "total time": 2.0,
                    "time step": 0.002,
                    "order": "second",
                    "start": 3000,
                },
                cycles=5,
            ),
        }
    )
    assert under(c, *TS) == {
        "type": "unsteady",
        "kind": "dual time stepping",
        "total time": 2.0,
        "time step": 0.002,
        "order": "second",
        "start": 3000,
    }
    assert under(c, *CC) == {"cycles": 5}


def test_t6_incompressible_scheme_is_dropped_because_v1_ran_only_coupled():
    c = run(
        {
            **INCOMP,
            **tm(
                unsteady={"total time": 1.0, "time step": 1.0},
                scheme={
                    "name": "implicit euler",
                    "stage": 1,
                    "kind": "local timestepping",
                },
                cycles=100,
            ),
        }
    )
    assert c.decisions == []
    for key in ("name", "stage"):
        d = disposition(c, "time marching", "scheme", key)
        assert d.kind == "dropped"
        assert "only the coupled incompressible solver" in d.reason
    assert disposition(c, "time marching", "scheme", "kind").kind == "dropped"
    assert not any(p[:3] == CC + ("scheme",) for p in c.v2)
    assert any("'coupled'" in n.text for n in c.notes)
    assert under(c, *CC) == {"cycles": 100}


def test_t6_incompressible_cfl_and_multigrid_are_dropped_as_v1_never_used_them():
    c = run(
        {
            **INCOMP,
            **tm(
                unsteady={"total time": 1.0, "time step": 1.0},
                multigrid=1,
                cfl=10.0,
                **{
                    "cfl transport": 5.0,
                    "cfl ramp factor": {"growth": 1.05, "initial": 0.1},
                },
                cycles=600,
            ),
        }
    )
    assert c.decisions == []
    for keys in (
        ("multigrid",),
        ("cfl",),
        ("cfl transport",),
        ("cfl ramp factor", "growth"),
        ("cfl ramp factor", "initial"),
    ):
        d = disposition(c, "time marching", *keys)
        assert d.kind == "dropped"
        assert d.reason == time_rule.INCOMPRESSIBLE_CFL_REASON
    assert under(c, *CC) == {"cycles": 600}
    assert not any(p[:3] == ("solver", "numerical scheme", "multigrid") for p in c.v2)
    assert unclaimed_paths(c) == {("equations",), ("incomp viscous", "order")}


def test_t6_incompressible_global_timestepping_kind_is_unclaimed():
    c = run(
        {
            **INCOMP,
            **tm(
                unsteady={"total time": 0.2, "time step": 0.01},
                scheme={"kind": "global timestepping"},
                cycles=10,
            ),
        }
    )
    assert ("time marching", "scheme", "kind") in unclaimed_paths(c)
    assert c.decisions == []


def test_t15_a_block_with_only_unsteady_converts_its_own_keys():
    c = run(tm(unsteady={"total time": 0.2, "time step": 1.0e-4, "order": 2}))
    assert under(c, *TS) == {
        "type": "unsteady",
        "kind": "dual time stepping",
        "total time": 0.2,
        "time step": 1.0e-4,
        "order": 2,
    }
    assert under(c, *CC) == {}
    assert c.unclaimed() == []
    assert c.decisions == []


def test_t15_a_block_with_no_unsteady_is_steady():
    c = run(tm(cfl=5.0, cycles=200))
    assert under(c, *TS) == {"type": "steady"}
    assert under(c, *CC) == {"cfl": 5.0, "cycles": 200}
    assert c.unclaimed() == []
    validate_compressible(c)


def test_t15_a_block_with_only_the_gts_scheme_has_no_time_keys():
    c = run(tm(scheme=dict(GTS_RK3["scheme"])))
    assert under(c, *TS) == {
        "type": "unsteady",
        "kind": "global timestepping",
        "scheme": {"name": "runge kutta", "stage": "rk third order tvd"},
    }
    assert c.unclaimed() == []


def test_t7_a_bare_cfl_stays_a_number():
    c = run(tm(**steady_tm(cfl=2.5, cycles=10)))
    assert c.v2[CC + ("cfl",)] == Source(c.v1.leaf("time marching", "cfl"))
    validate_compressible(c)


def test_t7_cfl_with_its_siblings_becomes_a_dict():
    c = run(
        tm(
            **steady_tm(
                scheme={"name": "runge kutta", "stage": 5},
                cfl=2.0,
                **{
                    "cfl transport": 1.0,
                    "cfl coarse": 1.5,
                    "cfl viscous factor": 0.5,
                    "multigrid": 3,
                },
                cycles=10,
            )
        )
    )
    assert under(c, *CC, "cfl") == {
        "cfl": 2.0,
        "cfl turbulence": 1.0,
        "cfl coarse": 1.5,
        "cfl viscous factor": 0.5,
    }
    assert disposition(c, "time marching", "cfl transport").v2_path == CC + (
        "cfl",
        "cfl turbulence",
    )
    assert c.unclaimed() == []
    validate_compressible(c)


def test_t7_cfl_coarse_without_multigrid_is_dropped():
    c = run(tm(**steady_tm(cfl=2.0, **{"cfl coarse": 1.5}, cycles=10)))
    d = disposition(c, "time marching", "cfl coarse")
    assert d.kind == "dropped"
    assert "multigrid" in d.reason
    assert c.v2[CC + ("cfl",)] == Source(c.v1.leaf("time marching", "cfl"))
    validate_compressible(c)


def test_t7_proposed_siblings_without_cfl_get_the_v1_default():
    c = run(tm(**steady_tm(**{"cfl transport": 0.5}, cycles=10)))
    assert under(c, *CC, "cfl") == {"cfl": 1.0, "cfl turbulence": 0.5}
    assert c.v2[CC + ("cfl", "cfl")] == Value(1.0)
    assert any("1.0" in n.text and "cfl" in n.text for n in c.notes)
    validate_compressible(c)


def test_t8_cfl_ramp_factor_folds_into_one_exponential_ramp():
    c = run(
        tm(
            **steady_tm(
                cfl=10.0,
                **{"cfl ramp factor": {"growth": 1.1, "initial": 0.1}},
                cycles=100,
            )
        ),
        {("time marching", "cfl ramp factor", "growth"): "growth"},
    )
    path = CC + ("cfl", "cfl ramp")
    value = c.v2[path]
    assert isinstance(value, Expr)
    assert value.value == [{"type": "exponential", "factor": 1.1, "initial": 0.1}]
    assert value.text == '[{"type": "exponential", "factor": growth, "initial": 0.1}]'
    for key in ("growth", "initial"):
        d = disposition(c, "time marching", "cfl ramp factor", key)
        assert d.kind == "moved"
        assert d.v2_path == path
    assert under(c, *CC, "cfl")["cfl"] == 10.0
    validate_compressible(c)


def ramp_two(cycle, current_cfl):
    return min(current_cfl * 1.05, 10.0)


def ramp_three(solve_cycle, real_time_cycle, cfl):
    return 1.0


def test_t9_open_question_2_three_argument_ramp_func_is_a_decision():
    c = run(
        tm(**steady_tm(cfl=10.0, **{"ramp func": ramp_three}, cycles=100)),
        {("time marching", "ramp func"): "ramp_three"},
    )
    assert len(c.decisions) == 1
    decision = c.decisions[0]
    assert decision.v2_path == CC + ("cfl", "cfl ramp")
    question = f"rewrite ramp_three as {CFL_RAMP.definition('ramp_three')}"
    assert decision.question == question
    assert "(solve_cycle, real_time_cycle, cfl)" in decision.details
    assert "def ramp_three(*, cycle, current_cfl)" in decision.details
    assert "A ramp list can replace the function" in decision.details
    assert disposition(c, "time marching", "ramp func").kind == "decide"
    assert c.v2[CC + ("cfl", "cfl")] == Source(c.v1.leaf("time marching", "cfl"))


def test_t9_two_argument_ramp_func_moves():
    c = run(
        tm(**steady_tm(cfl=10.0, **{"ramp func": ramp_two}, cycles=100)),
        {("time marching", "ramp func"): "ramp_two"},
    )
    assert c.decisions == []
    value = c.v2[CC + ("cfl", "cfl ramp")]
    assert isinstance(value, Source)
    assert value.leaf.source == "ramp_two"
    [note] = [n for n in c.notes if n.text.startswith("ramp_two:")]
    assert note.text.startswith(
        "ramp_two: `def ramp_two(cycle, current_cfl):` already takes the keywords"
    )
    validate_compressible(c)


def test_t9_ramp_func_and_cfl_ramp_factor_together_is_a_decision():
    c = run(
        tm(
            **steady_tm(
                cfl=10.0,
                **{
                    "ramp func": ramp_two,
                    "cfl ramp factor": {"growth": 1.1, "initial": 0.1},
                },
                cycles=100,
            )
        ),
        {("time marching", "ramp func"): "ramp_two"},
    )
    assert [d.v2_path for d in c.decisions] == [CC + ("cfl", "cfl ramp")]
    assert c.unclaimed() == []


def test_t10_multigrid_moves_under_the_numerical_scheme():
    c = run(
        tm(
            **steady_tm(
                scheme={"name": "runge kutta", "stage": 5},
                cfl=2.0,
                multigrid=3,
                **{
                    "multigrid cycles": 500,
                    "multigrid ramp": 2.0,
                    "prolong factor": 0.75,
                    "prolong transport factor": 0.3,
                },
                cycles=1000,
            )
        )
    )
    assert under(c, *MG) == {
        "levels": 3,
        "cycles": 500,
        "multigrid ramp": 2.0,
        "prolong factor": 0.75,
        "prolong transport factor": 0.3,
    }
    assert c.unclaimed() == []
    validate_compressible(c)


def test_t10_multigrid_with_an_implicit_scheme_is_left_unclaimed():
    c = run(
        tm(
            **steady_tm(
                scheme={"name": "implicit euler", "stage": 1},
                cfl=2.0,
                multigrid=3,
                **{"prolong factor": 0.75},
                cycles=1000,
            )
        )
    )
    assert unclaimed_paths(c) == {
        ("time marching", "multigrid"),
        ("time marching", "prolong factor"),
    }
    assert under(c, *MG) == {}


def test_t11_explicit_scheme_on_an_overset_mesh_is_a_decision():
    c = run(
        {
            **tm(**steady_tm(scheme={"name": "runge kutta", "stage": 5}, cycles=10)),
            "BC_1": OVERSET,
        }
    )
    assert len(c.decisions) == 1
    decision = c.decisions[0]
    assert decision.v2_path == CC + ("scheme", "name")
    assert decision.options == ("implicit euler", "lu-sgs", "mf-gmres")
    assert disposition(c, "time marching", "scheme", "name").kind == "decide"
    assert c.v2[CC + ("scheme", "stage")] == Source(
        c.v1.leaf("time marching", "scheme", "stage")
    )


def test_t11_explicit_scheme_without_an_overset_boundary_moves():
    c = run(tm(**steady_tm(scheme={"name": "runge kutta", "stage": 5}, cycles=10)))
    assert c.decisions == []
    assert disposition(c, "time marching", "scheme", "name").kind == "moved"


def test_t11_local_timestepping_is_dropped_and_class_is_unclaimed():
    class Smoother:
        pass

    c = run(
        tm(
            **steady_tm(
                scheme={
                    "name": "runge kutta",
                    "kind": "local timestepping",
                    "class": Smoother,
                },
                cycles=10,
            )
        ),
        {("time marching", "scheme", "class"): "Smoother"},
    )
    d = disposition(c, "time marching", "scheme", "kind")
    assert d.kind == "dropped"
    assert (
        d.reason == "the pseudo-time loop is implied by steady and dual time stepping"
    )
    assert unclaimed_paths(c) == {("time marching", "scheme", "class")}
    assert under(c, *CC, "scheme") == {"name": "runge kutta"}


DTS_TM = {"unsteady": {"total time": 1.0, "time step": 0.01}, "cycles": 20}
ROTOR = {
    "type": "rotating",
    "zone": [3],
    "axis": [0, 0, 1],
    "origin": [0, 0, 0],
    "omega": 10.0,
}


OVERSET = {"type": "overset", "zone": [7]}


MOTION = ("model", LABEL, "transforms", "motion")
MOVING = MOTION + ("moving mesh",)


def linear_accel(**kwargs):
    return {"velocity": (-69.4 * min(kwargs["time"] / 1.5, 1.0), 0.0, 0.0)}


TRAIN = {
    "type": "translating",
    "zone": [6],
    "vector": [-69.4, 0.0, 0.0],
    "translation function": linear_accel,
}


def test_m1_moving_mesh_true_makes_the_one_rotating_zone_the_moving_mesh_motion():
    c = run({**tm(**DTS_TM, **{"moving mesh": True}), "FZ_1": ROTOR, "BC_1": OVERSET})
    assert under(c, *MOTION) == {
        "type": "rotating",
        "moving mesh": True,
        "axis": [0, 0, 1],
        "origin": [0, 0, 0],
        "omega": 10.0,
    }
    d = disposition(c, "time marching", "moving mesh")
    assert d.kind == "respelled" and d.v2_path == MOVING
    assert disposition(c, "fz 1", "zone").kind == "dropped"
    assert c.decisions == []
    [note] = [n for n in c.notes if "zone" in n.text]
    assert note.text == (
        "v1 moved the whole mesh whatever 'zone' listed, so a moving mesh's "
        "'motion' takes no 'zones'."
    )
    assert ("model", LABEL, "fluid zones", "FZ_1", "type") not in c.v2


def test_m1_a_translating_zone_becomes_motion_with_its_velocity():
    zone = {"type": "translating", "zone": [6], "vector": [-69.4, 0.0, 0.0]}
    c = run({**tm(**DTS_TM, **{"moving mesh": True}), "FZ_1": zone, "BC_1": OVERSET})
    assert under(c, *MOTION) == {
        "type": "translating",
        "moving mesh": True,
        "velocity": [-69.4, 0.0, 0.0],
    }
    assert c.decisions == []


def test_m1_a_translation_function_asks_for_a_ramp_time():
    c = run({**tm(**DTS_TM, **{"moving mesh": True}), "FZ_1": TRAIN, "BC_1": OVERSET})
    [decision] = c.decisions
    assert decision.v2_path == MOTION + ("ramp time",)
    assert decision.question == (
        "A v2 control deck carries no functions. 'linear_accel' was the "
        "translation function of fluid zone 'FZ_1'. Write the motion in "
        "'transforms > motion' as 'velocity', its full value in m/s, and, if "
        "the body starts from rest with a linear rise, 'ramp time', the seconds "
        "it takes to reach that velocity. Any other profile cannot be expressed."
    )
    assert decision.v1_paths == (("fz 1", "translation function"),)
    assert under(c, *MOTION)["velocity"] == [-69.4, 0.0, 0.0]
    assert isinstance(c.v2[MOTION + ("ramp time",)], Placeholder)


TWO_ZONES = (
    "v1 neither moved mesh {model} nor applied a frame to it: its solver refuses "
    "'moving mesh' with more than one moving zone on a mesh ({zones}), logs "
    "an error and runs without them. What should the mesh do?"
)


def test_m1_two_moving_zones_on_one_model_moved_no_mesh_in_v1():
    from zutil.update_input.rules import zones

    c = run(
        {
            **tm(**DTS_TM, **{"moving mesh": True}),
            "FZ_1": ROTOR,
            "fz_2": {"type": "translating", "zone": [4], "vector": [1, 0, 0]},
            "FZ_3": {"type": "porous", "zone": [5]},
            "BC_1": OVERSET,
        }
    )
    zones.convert(c)
    [decision] = c.decisions
    assert decision.v2_path == MOTION
    assert decision.question == TWO_ZONES.format(model=LABEL, zones="FZ_1, fz_2")
    assert disposition(c, "time marching", "moving mesh").kind == "decide"
    assert all(c.claimed(lf) for lf in c.v1.leaves_under("FZ_1"))
    assert all(c.claimed(lf) for lf in c.v1.leaves_under("fz_2"))


def test_m1_two_rotating_zones_under_moving_mesh_are_a_decision():
    from zutil.update_input.rules import zones

    c = run(
        {
            **tm(**steady_tm(cycles=10), **{"moving mesh": True}),
            "FZ_1": ROTOR,
            "FZ_2": dict(ROTOR, zone=[4]),
        }
    )
    zones.convert(c)
    [decision] = c.decisions
    assert decision.v2_path == MOTION
    assert decision.question == TWO_ZONES.format(model=LABEL, zones="FZ_1, FZ_2")
    assert len(decision.options) == 3
    assert ("model", LABEL, "fluid zones", "FZ_2", "type") not in c.v2
    assert not c.unclaimed()


def test_m1_moving_mesh_true_with_no_moving_zone_is_a_decision():
    c = run(tm(**DTS_TM, **{"moving mesh": True}))
    [decision] = c.decisions
    assert decision.v2_path == MOTION
    assert "'transforms'" in decision.details
    assert disposition(c, "time marching", "moving mesh").kind == "decide"


STILL_MESH = {
    "steady": (
        steady_tm(cycles=10),
        "v2 moves a mesh only in an unsteady run, and this deck is steady.",
        "make the run unsteady",
    ),
    "no interface": (
        DTS_TM,
        "v2 moves a mesh only with an overset or sliding interface, and this "
        "deck has none.",
        "add the interface",
    ),
}


@pytest.mark.parametrize("case", STILL_MESH)
def test_m1_a_mesh_v2_cannot_move_under_a_whole_mesh_rotating_zone_is_a_decision(
    case,
):
    from zutil.update_input.rules import zones

    marching, why, change = STILL_MESH[case]
    c = run({**tm(**marching, **{"moving mesh": True}), "FZ_1": ROTOR}, cell_zones=[3])
    [decision] = c.decisions
    assert decision.v2_path == MOVING
    assert decision.question == (
        f"{why} Set 'moving mesh' False to solve the flow in the moving frame, "
        f"or {change}?"
    )
    assert disposition(c, "time marching", "moving mesh").kind == "decide"
    assert not any(c.claimed(lf) for lf in c.v1.leaves_under("FZ_1"))
    zones.convert(c)
    assert under(c, *MOTION)["type"] == "rotating"
    assert under(c, *MOTION)["origin"] == [0, 0, 0]
    assert isinstance(c.v2[MOVING], Placeholder)
    assert not under(c, "model", LABEL, "fluid zones")
    assert len(c.decisions) == 1
    assert not c.unclaimed()


@pytest.mark.parametrize("cell_zones", [None, [0, 3]], ids=["no mesh", "part"])
@pytest.mark.parametrize("case", STILL_MESH)
def test_m1_a_mesh_v2_cannot_move_under_a_part_mesh_rotating_zone_is_a_decision(
    case, cell_zones
):
    from zutil.update_input.rules import zones

    marching, why, change = STILL_MESH[case]
    c = run(
        {**tm(**marching, **{"moving mesh": True}), "FZ_1": ROTOR},
        cell_zones=cell_zones,
    )
    fz = ("model", LABEL, "fluid zones", "FZ_1")
    [decision] = c.decisions
    assert decision.v2_path == fz + ("moving mesh",)
    assert decision.question == (
        f"{why} Keep fluid zone FZ_1 as a rotating frame on its cell zones, with "
        f"the mesh still, or {change} and move the whole mesh?"
    )
    assert decision.details == (
        "v1 set 'moving mesh' once for the deck and moved the whole mesh "
        "whatever 'zone' listed. The draft writes FZ_1 as a rotating fluid "
        "zone: to keep it, delete this key. To move the mesh, replace the "
        'fluid zone with \'transforms > motion\', {"type": "rotating", '
        '"axis": [...], "origin": [...], "omega": ..., "moving mesh": True}.'
    )
    zones.convert(c)
    assert under(c, *fz)["cell zones"] == [3]
    assert under(c, *fz)["type"] == "rotating"
    assert not under(c, *MOTION)
    assert len(c.decisions) == 1
    assert not c.unclaimed()


def test_m1_moving_mesh_false_is_dropped_and_leaves_the_zone_to_be_a_frame():
    c = run({**tm(**DTS_TM, **{"moving mesh": False}), "FZ_1": ROTOR})
    assert disposition(c, "time marching", "moving mesh").kind == "dropped"
    assert not under(c, *MOTION)
    assert not any(c.claimed(lf) for lf in c.v1.leaves_under("FZ_1"))


def test_m1_moving_mesh_on_an_incompressible_deck_is_dropped_and_the_zone_stays_a_frame():
    c = run(
        {
            **INCOMP,
            **tm(**DTS_TM, **{"moving mesh": True}),
            "FZ_1": ROTOR,
            "BC_1": OVERSET,
        }
    )
    d = disposition(c, "time marching", "moving mesh")
    assert d.kind == "dropped"
    assert d.reason == "the v1 incompressible solver never moved the mesh"
    assert not under(c, *MOTION)
    assert c.decisions == []
    assert not any(c.claimed(lf) for lf in c.v1.leaves_under("FZ_1"))


def test_m1_global_time_stepping_with_moving_mesh_makes_motion():
    gts = {
        "unsteady": {"total time": 1.0, "time step": 0.01},
        "scheme": {"kind": "global timestepping", "name": "runge kutta", "stage": 3},
        "moving mesh": True,
    }
    c = run({**tm(**gts), "FZ_1": ROTOR, "BC_1": OVERSET})
    assert under(c, *MOTION)["type"] == "rotating"
    assert under(c, *MOTION)["moving mesh"] is True


def test_t13_proposed_cycles_per_overset_cycle_of_one_is_dropped():
    c = run(tm(**DTS_TM, **{"cycles per overset cycle": 1}))
    assert disposition(c, "time marching", "cycles per overset cycle").kind == "dropped"


def test_t13_proposed_cycles_per_overset_cycle_above_one_is_unclaimed():
    c = run(tm(**DTS_TM, **{"cycles per overset cycle": 4}))
    assert unclaimed_paths(c) == {("time marching", "cycles per overset cycle")}


def test_t14_cycles_with_dual_time_stepping_go_to_convergence_control():
    c = run(tm(unsteady={"total time": 0.2, "time step": 1.0e-4}, cycles=40))
    assert disposition(c, "time marching", "cycles").v2_path == CC + ("cycles",)
    assert under(c, *CC) == {"cycles": 40}


@pytest.mark.parametrize("key", ["cycles", "cfl"])
def test_gts_leaves_keys_v1_rejects_unclaimed(key):
    tree = copy.deepcopy(GTS_RK3)
    tree[key] = 10
    c = run(tm(**tree))
    assert unclaimed_paths(c) == {("time marching", key)}


def test_gts_drops_order_and_start_which_the_v1_solver_ignored():
    tree = copy.deepcopy(GTS_RK3)
    tree["unsteady"].update({"order": "second", "start": 10})
    c = run(tm(**tree))
    for key in ("order", "start"):
        assert disposition(c, "time marching", "unsteady", key).kind == "dropped"
    validate_compressible(c)


def test_t10_multigrid_keys_without_levels_are_dropped():
    c = run(tm(**steady_tm(cfl=2.0, **{"prolong factor": 0.75}, cycles=10)))
    d = disposition(c, "time marching", "prolong factor")
    assert d.kind == "dropped"
    assert "one level" in d.reason
    assert under(c, *MG) == {}
