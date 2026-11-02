"""Rule ``conditions``: reference conditions, initialisation and restart (C1 to C14).

Each toy deck in ``decks/conditions`` is run with ``runpy`` and flattened to
leaves here, so these tests need no source layer. Where a v2 fragment is
checked against the real schema, the blocks other rules own are supplied by
a fixed scaffold.
"""

from __future__ import annotations

import contextlib
import copy
import os
import runpy
from pathlib import Path

import pytest

from zutil.update_input.model import (
    Conversion,
    Expr,
    Leaf,
    Placeholder,
    Source,
    V1Deck,
    Value,
    norm_key,
)
from zutil.update_input.rules import conditions

DECKS = Path(__file__).parent / "decks" / "conditions"
GOLDEN = Path(__file__).parent / "decks" / "golden"

REF = ("solver", "reference conditions")
INIT = ("solver", "initialisation")
MODEL = "toy"


def bc_path(label: str, key: str) -> tuple:
    return ("model", MODEL, "boundary conditions", label, key)


def _source_text(value) -> str:
    if callable(value):
        return value.__name__
    return repr(value)


def flatten(tree: dict, prefix: tuple = ()) -> list[Leaf]:
    leaves = []
    for key, value in tree.items():
        keys = prefix + (key,)
        if isinstance(value, dict):
            leaves.extend(flatten(value, keys))
        else:
            leaves.append(Leaf(keys, value, _source_text(value)))
    return leaves


@contextlib.contextmanager
def _in_dir(path: Path):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def load(path: Path) -> V1Deck:
    with _in_dir(path.parent):
        parameters = runpy.run_path(str(path))["parameters"]
    return V1Deck(flatten(parameters), parameters)


def run(deck: V1Deck, path: Path | None = None) -> Conversion:
    c = Conversion(deck, MODEL)
    if path is not None:
        c.deck_files = (path.resolve(),)
    conditions.convert(c)
    return c


def toy(name: str) -> Conversion:
    path = DECKS / f"{name}.py"
    return run(load(path), path)


def disposition(c: Conversion, *keys):
    path = tuple(norm_key(k) for k in keys)
    matches = [d for d in c.dispositions if d.v1_path == path]
    assert len(matches) == 1, f"{path} has {len(matches)} dispositions"
    return matches[0]


def leaf(c: Conversion, *keys) -> Leaf:
    found = c.v1.leaf(*keys)
    assert found is not None
    return found


def owned_leaves(c: Conversion) -> list[Leaf]:
    import fnmatch

    return [
        lf
        for lf in c.v1
        if any(fnmatch.fnmatchcase(lf.path[0], pat) for pat in conditions.OWNS)
    ]


def _python_value(value):
    if isinstance(value, Source):
        return value.leaf.value
    if isinstance(value, (Value, Expr)):
        return value.value
    if isinstance(value, Placeholder):
        return value.text
    raise TypeError(value)


def v2_tree(c: Conversion) -> dict:
    root: dict = {}
    for path, value in c.v2.items():
        node = root
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = _python_value(value)
    return root


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


EQUATIONS = {
    "euler": {"type": "euler"},
    "viscous": {"type": "viscous"},
    "rans": {"type": "rans", "turbulence": {"model": "sst"}},
}


def scaffold(equations: str, bcs: dict | None = None) -> dict:
    """The blocks other rules own, for a compressible GTS deck."""
    return {
        "config version": 2,
        "solver": {
            "equations": EQUATIONS[equations],
            "numerical scheme": {"order": "first"},
            "fluid properties": {
                "material": "air",
                "gamma": 1.4,
                "gas constant": 287.0,
                "sutherlands const": 110.4,
                "prandtl no": 0.72,
            },
            "solver settings": {"type": "compressible"},
            "time settings": {
                "type": "unsteady",
                "kind": "global timestepping",
                "total time": 1.0,
                "time step": 0.1,
                "scheme": {"name": "runge kutta", "stage": 3},
            },
            "output settings": {
                "solution": {"format": "vtk"},
                "report": {"frequency": 1},
            },
        },
        "model": {
            MODEL: {
                "mesh": "toy.h5",
                "boundary conditions": bcs
                or {"BC_0": {"zones": [0], "type": "symmetry"}},
            }
        },
    }


def validate(c: Conversion, equations: str, bcs: dict | None = None):
    from zutil.validate import validate_zcfd_input

    deck = scaffold(equations, bcs)
    if equations != "euler":
        deck["solver"]["fluid properties"]["turbulent prandtl no"] = 0.9
    return validate_zcfd_input(_merge(deck, v2_tree(c)))


def test_owns_and_order():
    assert conditions.ORDER == 10
    assert set(conditions.OWNS) == {
        "ic *",
        "reference",
        "initial",
        "restart",
        "restart casename",
        "restart ignore history",
        "interpolate restart",
        "restart meshname",
        "solution smooth cycles",
    }


def test_c1_plain_rans_ic_moves_every_key_under_its_label():
    c = toy("rans_plain")
    ic = REF + ("IC_1",)
    assert c.v2[ic + ("temperature",)] == Source(leaf(c, "IC_1", "temperature"))
    assert c.v2[ic + ("reynolds no",)] == Source(leaf(c, "IC_1", "Reynolds No"))
    assert c.v2[ic + ("reference length",)] == Source(
        leaf(c, "IC_1", "Reference Length")
    )
    assert c.v2[ic + ("turbulence intensity",)] == Source(
        leaf(c, "IC_1", "turbulence intensity")
    )
    assert disposition(c, "IC_1", "pressure").kind == "moved"
    validate(c, "rans")


def test_c2_velocity_moves_with_only_the_key_case_changed():
    c = toy("rans_plain")
    ic = REF + ("IC_1",)
    assert c.v2[ic + ("v", "vector")] == Source(leaf(c, "IC_1", "V", "vector"))
    assert c.v2[ic + ("v", "mach")] == Source(leaf(c, "IC_1", "V", "Mach"))
    assert disposition(c, "IC_1", "V", "Mach").kind == "moved"


def test_c4_viscosity_stays_viscosity():
    c = toy("viscosity_only")
    assert c.v2[REF + ("IC_1", "viscosity")] == Source(leaf(c, "IC_1", "viscosity"))
    assert not any(p[-1] == "reynolds no" for p in c.v2)
    validate(c, "rans")


def test_c4_viscosity_beside_reynolds_no_drops_reynolds_no_as_v1_ignored_it():
    c = toy("viscosity_and_reynolds")
    assert c.v2[REF + ("IC_1", "viscosity")] == Source(leaf(c, "IC_1", "viscosity"))
    d = disposition(c, "IC_1", "Reynolds No")
    assert d.kind == "dropped"
    assert d.reason == conditions.VISCOSITY_WINS
    assert REF + ("IC_1", "reynolds no") not in c.v2
    validate(c, "rans")


def test_c3_euler_drops_viscosity_and_turbulence_with_the_inviscid_reason():
    c = toy("euler_drops")
    for key in ("Reynolds No", "turbulence intensity", "eddy viscosity ratio"):
        d = disposition(c, "IC_1", key)
        assert d.kind == "dropped"
        assert d.reason == "inviscid: no viscosity in the reference condition"
    assert REF + ("IC_1", "reynolds no") not in c.v2
    validate(c, "euler")


def test_c3_laminar_drops_turbulence_with_the_laminar_reason():
    c = toy("laminar_drops")
    for key in ("turbulence intensity", "ambient turbulence intensity"):
        d = disposition(c, "IC_1", key)
        assert d.kind == "dropped"
        assert d.reason == "laminar: no turbulence state"
    assert disposition(c, "IC_1", "Reynolds No").kind == "moved"
    validate(c, "viscous")


def test_c5_abl_profile_moves_unchanged():
    c = toy("abl_profile")
    abl = REF + ("IC_1", "profile", "abl")
    for key in (
        "roughness length",
        "friction velocity",
        "surface layer height",
        "Monin-Obukhov length",
        "tke",
        "up",
    ):
        assert c.v2[abl + (norm_key(key),)] == Source(
            leaf(c, "IC_1", "profile", "ABL", key)
        )
    validate(c, "rans")


def test_c5_abl_z0_is_respelled_ground_level():
    c = toy("abl_profile")
    abl = REF + ("IC_1", "profile", "abl")
    assert c.v2[abl + ("ground level",)] == Source(
        leaf(c, "IC_1", "profile", "ABL", "z0")
    )
    assert abl + ("z0",) not in c.v2
    assert disposition(c, "IC_1", "profile", "ABL", "z0").kind == "respelled"
    validate(c, "rans")


def test_c5_field_profile_moves_unchanged():
    c = toy("field_profile")
    profile = REF + ("IC_1", "profile")
    assert c.v2[profile + ("field",)] == Source(leaf(c, "IC_1", "profile", "field"))
    assert c.v2[profile + ("use wall distance",)] == Source(
        leaf(c, "IC_1", "profile", "use wall distance")
    )
    validate(c, "rans")


def test_c1_ambient_turbulence_keys_move():
    c = toy("ambient_turbulence")
    ic = REF + ("IC_1",)
    assert c.v2[ic + ("ambient turbulence intensity",)] == Source(
        leaf(c, "IC_1", "ambient turbulence intensity")
    )
    assert c.v2[ic + ("ambient eddy viscosity ratio",)] == Source(
        leaf(c, "IC_1", "ambient eddy viscosity ratio")
    )
    validate(c, "rans")


def test_c6_ic_location_is_left_unclaimed_for_the_engine():
    c = toy("ic_location")
    assert not c.claimed(leaf(c, "IC_1", "location"))
    assert [lf.keys for lf in owned_leaves(c) if not c.claimed(lf)] == [
        ("IC_1", "location")
    ]
    assert not any(p[-1] == "location" for p in c.v2)


def test_c7_callable_ic_is_decide_naming_the_farfield_target():
    c = toy("callable_ic")
    assert isinstance(c.v2[REF + ("IC_2",)], Placeholder)
    assert disposition(c, "IC_2").kind == "decide"
    [decision] = c.decisions
    assert decision.v2_path == REF + ("IC_2",)
    assert "lift_target_ic" in decision.question
    assert "give BC_1 a target" in decision.details
    assert '"target": {"alpha": {"cl": ...}}' in decision.details
    assert not c.claimed(leaf(c, "BC_1", "condition"))


def test_c7_callable_ic_is_never_called():
    def lift_ic(RealTimeStep, Cycle):
        raise AssertionError("update_input called the deck's controller")

    evaluated = {"equations": "RANS", "reference": "IC_1", "IC_1": lift_ic}
    c = run(V1Deck(flatten(evaluated), evaluated))
    assert disposition(c, "IC_1").kind == "decide"
    [decision] = c.decisions
    assert "give the farfield a target" in decision.details


def test_c8_derived_ic_ratios_inline_into_its_pressure_inflow():
    c = toy("derived_pressure_inflow")
    assert c.v2[bc_path("BC_4", "total pressure ratio")] == Source(
        leaf(c, "IC_2", "total pressure ratio")
    )
    assert c.v2[bc_path("BC_4", "total temperature ratio")] == Source(
        leaf(c, "IC_2", "total temperature ratio")
    )
    assert not any(p[:3] == REF + ("IC_2",) for p in c.v2)
    ref = disposition(c, "IC_2", "reference")
    assert ref.kind == "dropped"
    assert "BC_4" in ref.reason and "IC_1" in ref.reason
    assert not any(c.claimed(lf) for lf in c.v1.leaves_under("BC_4"))
    validate(
        c,
        "rans",
        {
            "BC_4": {
                "zones": [2],
                "type": "inflow",
                "kind": "pressure",
                "reference": "IC_1",
            }
        },
    )


def test_c8_one_derived_ic_used_by_two_outflows_inlines_into_both():
    c = toy("derived_two_outflows")
    ratio = leaf(c, "IC_2", "static pressure ratio")
    assert c.v2[bc_path("BC_5", "static pressure ratio")] == Source(ratio)
    assert c.v2[bc_path("BC_6", "static pressure ratio")] == Source(ratio)
    assert disposition(c, "IC_2", "static pressure ratio").kind == "moved"
    assert any("BC_5" in n.text and "BC_6" in n.text for n in c.notes)
    outflow = {"type": "outflow", "kind": "pressure", "reference": "IC_1"}
    validate(
        c,
        "rans",
        {"BC_5": dict(outflow, zones=[3]), "BC_6": dict(outflow, zones=[4])},
    )


def test_c8_mass_flow_rate_inlines_into_a_massflow_inflow():
    c = toy("derived_massflow_inflow")
    assert c.v2[bc_path("bc_2", "mass flow rate")] == Source(
        leaf(c, "IC_3", "mass flow rate")
    )
    assert c.v2[bc_path("bc_2", "total temperature ratio")] == Source(
        leaf(c, "IC_3", "total temperature ratio")
    )
    validate(
        c,
        "rans",
        {
            "bc_2": {
                "zones": [2],
                "type": "inflow",
                "kind": "massflow",
                "reference": "IC_1",
            }
        },
    )


def test_c8_derived_vector_becomes_direction_on_a_pressure_inflow():
    c = toy("derived_vector_pressure_inflow")
    assert c.v2[bc_path("BC_4", "direction")] == Source(leaf(c, "IC_2", "vector"))
    assert disposition(c, "IC_2", "vector").v2_path == bc_path("BC_4", "direction")
    validate(
        c,
        "rans",
        {
            "BC_4": {
                "zones": [2],
                "type": "inflow",
                "kind": "pressure",
                "reference": "IC_1",
            }
        },
    )


def test_c8_derived_vector_becomes_direction_on_a_massflow_inflow():
    c = toy("derived_vector_massflow_inflow")
    assert c.v2[bc_path("BC_4", "direction")] == Source(leaf(c, "IC_2", "vector"))
    assert disposition(c, "IC_2", "vector").v2_path == bc_path("BC_4", "direction")
    assert c.v2[bc_path("BC_4", "mass flow ratio")] == Source(
        leaf(c, "IC_2", "mass flow ratio")
    )
    validate(
        c,
        "rans",
        {
            "BC_4": {
                "zones": [2],
                "type": "inflow",
                "kind": "massflow",
                "reference": "IC_1",
            }
        },
    )


def test_c8_derived_mach_inlines_into_a_pressure_inflow():
    evaluated = {
        "equations": "RANS",
        "IC_2": {
            "reference": "IC_1",
            "total pressure ratio": 1.1,
            "total temperature ratio": 1.0,
            "mach": 0.3,
        },
        "BC_4": {"zone": [2], "type": "inflow", "kind": "default", "condition": "IC_2"},
    }
    c = run(V1Deck(flatten(evaluated), evaluated))
    assert c.v2[bc_path("BC_4", "mach")] == Source(leaf(c, "IC_2", "mach"))
    assert disposition(c, "IC_2", "mach").kind == "moved"


def test_c8_derived_mach_on_a_massflow_inflow_is_decide():
    evaluated = {
        "equations": "RANS",
        "IC_2": {"reference": "IC_1", "mass flow ratio": 0.8, "mach": 0.3},
        "BC_4": {
            "zone": [2],
            "type": "inflow",
            "kind": "massflow",
            "condition": "IC_2",
        },
    }
    c = run(V1Deck(flatten(evaluated), evaluated))
    assert disposition(c, "IC_2", "mach").kind == "decide"


def test_c8_static_pressure_ratio_inlines_into_a_radial_outflow():
    evaluated = {
        "equations": "RANS",
        "IC_2": {"reference": "IC_1", "static pressure ratio": 0.95},
        "BC_7": {
            "zone": [7],
            "type": "outflow",
            "kind": "radial pressure gradient",
            "condition": "IC_2",
        },
    }
    c = run(V1Deck(flatten(evaluated), evaluated))
    d = disposition(c, "IC_2", "static pressure ratio")
    assert d.kind == "moved"
    assert d.v2_path == bc_path("BC_7", "static pressure ratio")


OUTFLOW_TEMPERATURE = "the solver never used a total temperature ratio on an outflow"


@pytest.mark.parametrize(
    "kind, ratio",
    [
        ("pressure", {"static pressure ratio": 0.95}),
        ("massflow", {"mass flow rate": 2.0}),
        ("radial pressure gradient", {"static pressure ratio": 0.95}),
    ],
)
def test_c8_total_temperature_ratio_on_an_outflow_is_dropped(kind, ratio):
    evaluated = {
        "equations": "RANS",
        "IC_2": {"reference": "IC_1", "total temperature ratio": 1.0, **ratio},
        "BC_5": {"zone": [5], "type": "outflow", "kind": kind, "condition": "IC_2"},
    }
    c = run(V1Deck(flatten(evaluated), evaluated))
    d = disposition(c, "IC_2", "total temperature ratio")
    assert d.kind == "dropped"
    assert d.reason == OUTFLOW_TEMPERATURE
    assert bc_path("BC_5", "total temperature ratio") not in c.v2
    assert not c.decisions


def test_c8_total_temperature_ratio_shared_with_an_outflow_stays_on_the_inflow():
    evaluated = {
        "equations": "RANS",
        "IC_2": {"reference": "IC_1", "total temperature ratio": 1.0},
        "BC_4": {
            "zone": [4],
            "type": "inflow",
            "kind": "massflow",
            "condition": "IC_2",
        },
        "BC_5": {
            "zone": [5],
            "type": "outflow",
            "kind": "massflow",
            "condition": "IC_2",
        },
    }
    c = run(V1Deck(flatten(evaluated), evaluated))
    ratio = leaf(c, "IC_2", "total temperature ratio")
    assert c.v2[bc_path("BC_4", "total temperature ratio")] == Source(ratio)
    assert bc_path("BC_5", "total temperature ratio") not in c.v2
    assert disposition(c, "IC_2", "total temperature ratio").kind == "moved"
    assert any(
        n.text
        == f"IC_2 > total temperature ratio is not written to BC_5: {OUTFLOW_TEMPERATURE}"
        for n in c.notes
    )


def test_c8_a_ratio_given_as_a_function_is_decide():
    def ratio(**kwargs):
        return 1.1

    evaluated = {
        "equations": "RANS",
        "IC_2": {"reference": "IC_1", "total pressure ratio": ratio},
        "BC_4": {"zone": [2], "type": "inflow", "kind": "default", "condition": "IC_2"},
    }
    c = run(V1Deck(flatten(evaluated), evaluated))
    d = disposition(c, "IC_2", "total pressure ratio")
    assert d.kind == "decide"
    assert d.v2_path == bc_path("BC_4", "total pressure ratio")
    assert "v2 takes a number" in c.decisions[0].question


def test_c10_derived_ic_as_the_initial_state_is_decide_with_its_reference_and_ratios():
    c = toy("derived_initial")
    path = INIT + ("initial conditions",)
    assert isinstance(c.v2[path], Placeholder)
    assert c.v2[INIT + ("reference",)] == Source(leaf(c, "reference"))
    decision = c.decisions[0]
    assert decision.v2_path == path
    assert "IC_1" in decision.details
    assert "total pressure ratio" in decision.details and "1.2" in decision.details
    for key in ("reference", "total pressure ratio", "total temperature ratio"):
        assert disposition(c, "IC_2", key).kind == "decide"
    assert disposition(c, "initial").kind == "decide"


def test_c9_derived_ic_nothing_refers_to_is_dropped_naming_it():
    c = toy("derived_unused")
    for key in ("reference", "static pressure ratio"):
        d = disposition(c, "IC_3", key)
        assert d.kind == "dropped"
        assert d.reason == "nothing refers to IC_3"
    assert not any(p[:3] == REF + ("IC_3",) for p in c.v2)
    validate(c, "rans")


def test_c9_derived_ic_used_by_a_farfield_stays_in_reference_conditions():
    c = toy("derived_farfield")
    ic = REF + ("IC_2",)
    assert c.v2[ic + ("reference",)] == Source(leaf(c, "IC_2", "reference"))
    assert c.v2[ic + ("total pressure ratio",)] == Source(
        leaf(c, "IC_2", "total pressure ratio")
    )
    validate(
        c,
        "rans",
        {
            "BC_1": {
                "zones": [1],
                "type": "farfield",
                "kind": "riemann",
                "condition": "IC_2",
            }
        },
    )


def test_c11_reference_only_becomes_initial_conditions():
    c = toy("reference_only")
    assert c.v2[INIT + ("initial conditions",)] == Source(leaf(c, "reference"))
    assert INIT + ("reference",) not in c.v2
    validate(c, "rans")


def test_c11_reference_and_initial_string():
    c = toy("reference_and_initial")
    assert c.v2[INIT + ("reference",)] == Source(leaf(c, "reference"))
    assert c.v2[INIT + ("initial conditions",)] == Source(leaf(c, "initial"))
    validate(c, "rans")


def test_c11_initial_name_dict():
    c = toy("initial_name")
    assert c.v2[INIT + ("reference",)] == Source(leaf(c, "reference"))
    assert c.v2[INIT + ("initial conditions",)] == Source(leaf(c, "initial", "name"))
    validate(c, "rans")


def test_c12_initial_function_seeded_from_ic_1():
    c = toy("initial_function")
    assert c.v2[INIT + ("reference",)] == Source(leaf(c, "reference"))
    assert c.v2[INIT + ("initial conditions",)] == Source(leaf(c, "initial", "func"))
    assert leaf(c, "initial", "func").source == "shock"
    name = disposition(c, "initial", "name")
    assert name.kind == "dropped" and "IC_1" in name.reason
    validate(c, "euler")


def test_c12_initial_function_seeded_from_another_ic_is_decide():
    c = toy("initial_function_other_ic")
    path = INIT + ("initial conditions",)
    assert isinstance(c.v2[path], Placeholder)
    assert disposition(c, "initial", "name").kind == "decide"
    assert disposition(c, "initial", "func").kind == "decide"
    assert "IC_2" in c.decisions[0].question
    assert "shock" in c.decisions[0].details


def test_c13_restart_keys_move_to_initialisation():
    c = toy("restart")
    for key in ("restart", "restart casename", "restart ignore history"):
        assert c.v2[INIT + (key,)] == Source(leaf(c, key))
    validate(c, "rans")


def test_c14_interpolated_restart_and_smooth_cycles():
    c = toy("interpolate_restart")
    for key in ("interpolate restart", "restart meshname", "restart casename"):
        assert c.v2[INIT + (key,)] == Source(leaf(c, key))
    assert c.v2[("solver", "solver settings", "solution smooth cycles")] == Source(
        leaf(c, "solution smooth cycles")
    )
    validate(c, "rans")


SWEEP = {
    "equations": "RANS",
    "reference": "IC_1",
    "restart": True,
    "restart casename": "old",
    "restart ignore history": True,
    "interpolate restart": True,
    "restart meshname": "coarse",
    "solution smooth cycles": 2,
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [10.0, 0.0, 0.0]},
        "viscosity": 1.8e-5,
        "location": [0.0, 0.0, 0.0],
        "profile": {
            "ABL": {
                "roughness length": 0.1,
                "friction velocity": 0.4,
                "surface layer height": 100.0,
                "Monin-Obukhov length": -50.0,
                "tke": 3.3,
            }
        },
    },
}

SWEEP_NAMES = [
    ("location",),
    ("surface layer height",),
    ("monin-obukhov length",),
    ("tke",),
    ("restart ignore history",),
    ("interpolate restart",),
    ("restart meshname",),
    ("solution smooth cycles",),
]


@pytest.mark.parametrize("name", SWEEP_NAMES, ids=lambda n: n[0])
def test_owned_names_the_corpus_never_uses_have_a_disposition(name):
    c = run(V1Deck(flatten(SWEEP), SWEEP))
    leaves = [lf for lf in c.v1 if lf.path[-1] == name[0]]
    assert len(leaves) == 1
    if name == ("location",):
        assert not c.claimed(leaves[0])
    else:
        assert disposition(c, *leaves[0].keys).kind == "moved"


def test_every_owned_leaf_but_location_is_claimed_and_nothing_else_is():
    c = run(V1Deck(flatten(SWEEP), SWEEP))
    owned = owned_leaves(c)
    assert {lf.keys for lf in owned if not c.claimed(lf)} == {("IC_1", "location")}
    assert not c.claimed(leaf(c, "equations"))
    validate_tree = v2_tree(c)
    assert "unconverted v1 keys" not in validate_tree


@pytest.mark.parametrize(
    "deck",
    [
        "plate_derived_ic/plate_coarse.py",
        "pipe_massflow/pipe_massflow.py",
        "shocktube_dts/shocktube_fv_second_imp.py",
        "shocktube_gts/shocktube_fv_first.py",
        "bolund_abl/bolund.py",
        "naca_ic_callable/n0012_257_cL_target_1_imp.py",
        "overset_chain_a1/cylinderA1.py",
        "bump_namespace/3DBump_5.py",
        "plate_wallfunction/plate_coarse_visc_wf.py",
        "powered_restart/wingbody_powered.py",
    ],
)
def test_golden_decks_claim_every_owned_leaf(deck):
    c = run(load(GOLDEN / deck))
    assert not [
        lf.keys
        for lf in owned_leaves(c)
        if not c.claimed(lf) and lf.path[-1] != "location"
    ]
    assert all(p[0] in ("solver", "model") for p in c.v2)


def test_golden_plate_derived_ic_inlines_into_bc_4():
    c = run(load(GOLDEN / "plate_derived_ic/plate_coarse.py"))
    assert c.v2[bc_path("BC_4", "total pressure ratio")] == Source(
        leaf(c, "IC_2", "total pressure ratio")
    )
    assert c.v2[INIT + ("initial conditions",)] == Source(leaf(c, "reference"))


def test_ic_names_are_respelt_to_the_label_when_only_case_differs():
    evaluated = {
        "equations": "RANS",
        "reference": "IC_1",
        "ic_1": {
            "temperature": 288.15,
            "pressure": 101325.0,
            "v": {"vector": [1.0, 0.0, 0.0], "mach": 0.2},
            "reynolds no": 6.0e6,
        },
    }
    c = run(V1Deck(flatten(evaluated), evaluated))
    assert c.v2[INIT + ("initial conditions",)] == Value("ic_1")
    assert disposition(c, "reference").kind == "respelled"
    validate(c, "rans")


def test_golden_pipe_massflow_inlines_mass_flow_into_its_inflow():
    c = run(load(GOLDEN / "pipe_massflow/pipe_massflow.py"))
    assert c.v2[bc_path("bc_2", "mass flow rate")] == Source(
        leaf(c, "IC_3", "mass flow rate")
    )
    assert c.v2[INIT + ("reference",)] == Value("ic_1")
    assert c.v2[INIT + ("initial conditions",)] == Value("ic_1")
    assert disposition(c, "IC_4", "reference").reason == "nothing refers to ic_4"


def test_golden_overset_chain_initial_differs_from_reference():
    c = run(load(GOLDEN / "overset_chain_a1/cylinderA1.py"))
    assert c.v2[INIT + ("reference",)] == Source(leaf(c, "reference"))
    assert c.v2[INIT + ("initial conditions",)] == Source(leaf(c, "initial"))
    assert leaf(c, "initial").value != leaf(c, "reference").value


def test_golden_shocktube_dts_initial_function():
    path = GOLDEN / "shocktube_dts/shocktube_fv_second_imp.py"
    c = run(load(path), path)
    assert c.v2[INIT + ("reference",)] == Source(leaf(c, "reference"))
    assert c.v2[INIT + ("initial conditions",)] == Source(leaf(c, "initial", "func"))
