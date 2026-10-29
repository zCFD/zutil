import copy
import runpy
from pathlib import Path

import pytest

from zutil.update_input.model import (
    Conversion,
    Expr,
    Placeholder,
    Source,
    V1Deck,
    Value,
    norm_key,
)
from zutil.update_input.rules import equations
from zutil.update_input.tests.engine.fakes import flatten
from zutil.validate.schemas import block_versions as versions
from zutil.validate.schemas.convergence_schemas import CoupledScheme
from zutil.validate.schemas.variants import Variants
from zutil.validate.validation_utils import replace_spaces_in_keys

DECKS = Path(__file__).parent / "decks" / "equations"

SOLVER = ("solver",)
EQ = ("solver", "equations")
NS = ("solver", "numerical scheme")
FP = ("solver", "fluid properties")
CS = ("solver", "convergence control", "scheme")


def toy(name):
    return runpy.run_path(str(DECKS / f"{name}.py"))["parameters"]


def convert(params):
    params = copy.deepcopy(params)
    deck = V1Deck(flatten(params), params)
    c = Conversion(deck, "model")
    equations.convert(c)
    return c


def shown(c, path):
    node = c.v2[path]
    if isinstance(node, Source):
        return node.leaf.value
    if isinstance(node, (Value, Expr)):
        return node.value
    return node.text


def tree(c, prefix):
    out = {}
    for path, node in c.v2.items():
        if path[: len(prefix)] == prefix and not isinstance(node, Placeholder):
            cursor = out
            for key in path[len(prefix) : -1]:
                cursor = cursor.setdefault(key, {})
            cursor[path[-1]] = shown(c, path)
    return out


def kinds(c):
    return {d.v1_path: d for d in c.dispositions}


def unclaimed(c):
    return {lf.path for lf in c.unclaimed()}


def validate(c):
    """Check the equations, scheme and fluid fragments against the versions the
    solver's own tables pick for this solver and equation type."""
    kind = shown(c, SOLVER + ("solver settings", "type"))
    equation = replace_spaces_in_keys(tree(c, EQ))
    deck = {
        "solver": {
            "solver_settings": {"type": kind},
            "equations": {"type": equation["type"]},
        }
    }
    for table, fragment in (
        (versions.EQUATIONS_BLOCK, equation),
        (versions.NUMERICAL_SCHEME, tree(c, NS)),
        (versions.FLUID_PROPERTIES, tree(c, FP)),
    ):
        version = Variants(table).version_for(deck)
        assert version is not None, (kind, equation["type"])
        version.model_validate(replace_spaces_in_keys(fragment))
    if kind == "incompressible":
        CoupledScheme.model_validate(replace_spaces_in_keys(tree(c, CS)))


def with_equation(name, block=None, **top):
    params = {"equations": name, "material": "air", "air": {}}
    if block is not None:
        params[name.lower()] = block
    params.update(top)
    return params


@pytest.mark.parametrize("name", ["euler", "Euler", "EULER", "viscous", "Viscous"])
def test_E1_compressible_name_moves_unchanged(name):
    c = convert(with_equation(name, {"order": "second"}))
    assert shown(c, EQ + ("type",)) == name
    assert shown(c, SOLVER + ("solver settings", "type")) == "compressible"
    assert not c.unclaimed()
    validate(c)


@pytest.mark.parametrize("name", ["rans", "RANS"])
def test_E1_rans_name_moves_unchanged(name):
    c = convert(
        with_equation(
            name, {"order": "second", "turbulence": {"model": "sst", "les": "none"}}
        )
    )
    assert shown(c, EQ + ("type",)) == name
    assert shown(c, SOLVER + ("solver settings", "type")) == "compressible"
    validate(c)


@pytest.mark.parametrize("name", ["les", "LES"])
def test_E1_les_name_moves_unchanged(name):
    c = convert(with_equation(name, {"order": "second"}))
    assert shown(c, EQ + ("type",)) == name
    validate(c)


@pytest.mark.parametrize(
    "name, v2_type",
    [
        ("incomp viscous", "viscous"),
        ("Incomp Viscous", "viscous"),
        ("incomp rans", "rans"),
        ("incomp les", "les"),
    ],
)
def test_E1_incompressible_name_sets_type_and_solver_settings(name, v2_type):
    block = {"order": "second", "relax": 0.5}
    if v2_type == "rans":
        block["turbulence"] = {"model": "sst", "les": "none", "relax": 0.5}
    if v2_type == "les":
        block["turbulence"] = {"les": "none", "relax": 0.5}
    c = convert(with_equation(name, block))
    assert shown(c, EQ + ("type",)) == v2_type
    assert kinds(c)[("equations",)].kind == "respelled"
    assert shown(c, SOLVER + ("solver settings", "type")) == "incompressible"
    assert not c.unclaimed()
    validate(c)


def test_E1_equations_leaf_is_claimed_once_and_solver_type_is_not_a_leaf_move():
    c = convert(with_equation("rans", {"turbulence": {"model": "sst"}}))
    claims = [d for d in c.dispositions if d.v1_path == ("equations",)]
    assert len(claims) == 1


def test_E2_unselected_equation_block_is_dropped_with_reason():
    params = with_equation("euler", {"order": "second"})
    params["rans"] = {"order": "first", "turbulence": {"model": "sst"}}
    c = convert(params)
    assert not c.unclaimed()
    dropped = kinds(c)[("rans", "order")]
    assert dropped.kind == "dropped"
    assert "rans" in dropped.reason
    assert ("solver", "numerical scheme", "order") in c.v2
    assert shown(c, NS + ("order",)) == "second"


def test_E3_compressible_scheme_block_moves_every_key():
    c = convert(toy("euler_scheme_keys"))
    assert tree(c, NS) == {
        "order": "second",
        "limiter": "van albada",
        "reconstruction": "umuscl",
        "inviscid flux scheme": "Roe",
        "entropy fix": 0.05,
        "linear gradients": True,
        "first order cycles": 20,
        "freeze limiter cycle": 500,
        "roe low dissipation sensor": "NTS",
        "roe dissipation sensor minimum": 0.1,
        "roe entropy fix coefficient": 0.002,
        "permit backward difference during reconstruction": True,
        "gradient": "least-squares",
    }
    assert not c.unclaimed()
    validate(c)


def test_E3_roe_is_moved_without_respelling():
    c = convert(toy("euler_scheme_keys"))
    assert kinds(c)[("euler", "inviscid flux scheme")].kind == "moved"
    assert isinstance(c.v2[NS + ("inviscid flux scheme",)], Source)


@pytest.mark.parametrize(
    "v1, v2",
    [("vanalbada", "van albada"), ("barth jespersen", "symmetric ratio")],
)
def test_E3_limiter_is_respelled_to_the_v2_name_of_the_same_limiter(v1, v2):
    c = convert(with_equation("euler", {"order": "second", "limiter": v1}))
    assert shown(c, NS + ("limiter",)) == v2
    assert kinds(c)[("euler", "limiter")].kind == "respelled"
    validate(c)


@pytest.mark.parametrize("name", ["van albada", "van leer", "none"])
def test_E3_limiter_v2_also_names_moves_unchanged(name):
    c = convert(with_equation("euler", {"order": "second", "limiter": name}))
    assert isinstance(c.v2[NS + ("limiter",)], Source)
    assert kinds(c)[("euler", "limiter")].kind == "moved"
    validate(c)


def test_E3_roe_low_diss_is_respelled():
    c = convert(
        with_equation(
            "euler", {"order": "first", "inviscid flux scheme": "Roe low diss"}
        )
    )
    assert shown(c, NS + ("inviscid flux scheme",)) == "roe low-diss"
    assert kinds(c)[("euler", "inviscid flux scheme")].kind == "respelled"
    validate(c)


def test_E3_hllc_moves_unchanged():
    c = convert(
        with_equation("euler", {"order": "first", "inviscid flux scheme": "HLLC"})
    )
    assert isinstance(c.v2[NS + ("inviscid flux scheme",)], Source)
    assert kinds(c)[("euler", "inviscid flux scheme")].kind == "moved"
    validate(c)


def test_E4_leastsq_true_becomes_gradient():
    c = convert(with_equation("euler", {"order": "first", "leastsq gradients": True}))
    assert shown(c, NS + ("gradient",)) == "least-squares"
    assert kinds(c)[("euler", "leastsq gradients")].kind == "respelled"
    validate(c)


def test_E4_leastsq_false_is_dropped():
    c = convert(with_equation("euler", {"order": "first", "leastsq gradients": False}))
    assert NS + ("gradient",) not in c.v2
    gone = kinds(c)[("euler", "leastsq gradients")]
    assert gone.kind == "dropped"
    assert "green-gauss" in gone.reason


def test_E5_precondition_on_moves_with_minimum_mach():
    params = with_equation(
        "euler",
        {
            "order": "first",
            "precondition": True,
            "preconditioner": {"minimum mach number": 0.02},
        },
    )
    c = convert(params)
    assert shown(c, EQ + ("precondition",)) is True
    assert shown(c, NS + ("preconditioner", "minimum mach number")) == 0.02
    assert not c.unclaimed()


def test_E5_precondition_off_drops_preconditioner_with_reason():
    params = with_equation(
        "euler",
        {
            "order": "first",
            "precondition": False,
            "preconditioner": {"minimum mach number": 0.02},
        },
    )
    c = convert(params)
    gone = kinds(c)[("euler", "preconditioner", "minimum mach number")]
    assert gone.kind == "dropped"
    assert "precondition" in gone.reason
    assert NS + ("preconditioner", "minimum mach number") not in c.v2
    assert shown(c, EQ + ("precondition",)) is False


def test_E5_preconditioner_without_precondition_key_is_dropped():
    params = with_equation(
        "euler", {"order": "first", "preconditioner": {"minimum mach number": 0.02}}
    )
    c = convert(params)
    assert (
        kinds(c)[("euler", "preconditioner", "minimum mach number")].kind == "dropped"
    )


def test_E5_incompressible_preconditioner_is_left_unclaimed():
    params = with_equation(
        "incomp viscous",
        {
            "order": "second",
            "precondition": True,
            "preconditioner": {"minimum mach number": 0.02},
        },
    )
    c = convert(params)
    assert unclaimed(c) == {("incomp viscous", "preconditioner", "minimum mach number")}


def test_E6_rans_turbulence_block_moves():
    c = convert(toy("rans_sst"))
    assert tree(c, EQ + ("turbulence",)) == {
        "model": "sst",
        "betastar": 0.09,
        "limit mut": True,
        "limit gradient k": 0.5,
        "qcr": 2000,
        "rotation correction": True,
        "production": 1,
        "separation correction": False,
        "cdes kw": 0.78,
        "a1": 0.31,
    }
    assert not c.unclaimed()
    validate(c)


def test_E6_les_none_is_dropped_as_the_default():
    c = convert(toy("rans_sst"))
    gone = kinds(c)[("rans", "turbulence", "les")]
    assert gone.kind == "dropped"
    assert EQ + ("turbulence", "hybrid") not in c.v2
    assert EQ + ("turbulence", "les") not in c.v2


def test_E6_ddes_becomes_hybrid():
    c = convert(toy("rans_ddes"))
    assert shown(c, EQ + ("turbulence", "hybrid")) == "DDES"
    assert kinds(c)[("rans", "turbulence", "les")].v2_path == EQ + (
        "turbulence",
        "hybrid",
    )
    assert shown(c, EQ + ("turbulence", "cd2")) == 3
    validate(c)


@pytest.mark.parametrize("hybrid", ["DES", "DDES", "IDDES", "SAS"])
def test_E6_every_hybrid_name_is_accepted(hybrid):
    c = convert(
        with_equation(
            "rans", {"order": "second", "turbulence": {"model": "sst", "les": hybrid}}
        )
    )
    assert shown(c, EQ + ("turbulence", "hybrid")) == hybrid
    validate(c)


def test_E6_approximate_sa_from_sst_results_is_left_unclaimed():
    params = with_equation(
        "rans",
        {
            "order": "second",
            "turbulence": {
                "model": "sa-neg",
                "approximate sa from sst results": True,
            },
        },
    )
    c = convert(params)
    assert unclaimed(c) == {("rans", "turbulence", "approximate sa from sst results")}


def test_E7_les_wale_is_respelled_lower_case():
    c = convert(toy("les_wale"))
    assert shown(c, EQ + ("turbulence", "les")) == "wale"
    assert kinds(c)[("les", "turbulence", "les")].kind == "respelled"
    assert not c.unclaimed()
    validate(c)


def test_E7_les_none_moves_unchanged():
    c = convert(
        with_equation("les", {"order": "second", "turbulence": {"les": "none"}})
    )
    assert kinds(c)[("les", "turbulence", "les")].kind == "moved"
    validate(c)


def test_E8_incomp_rans_relax_goes_to_the_convergence_scheme():
    c = convert(toy("incomp_rans"))
    assert shown(c, CS + ("implicit relax",)) == 0.7
    assert shown(c, CS + ("turbulence relax",)) == 0.3
    assert EQ + ("turbulence", "relax") not in c.v2
    assert EQ + ("relax",) not in c.v2
    assert not c.unclaimed()
    validate(c)


def test_E8_incomp_viscous_relax_cylinder_incomp_refonly():
    c = convert(toy("incomp_viscous"))
    assert shown(c, CS + ("implicit relax",)) == 0.5
    assert CS + ("turbulence relax",) not in c.v2
    assert not c.unclaimed()
    validate(c)


def test_E8_incomp_les_turbulence_relax_stays_in_equations():
    c = convert(toy("incomp_les"))
    assert shown(c, EQ + ("turbulence", "relax")) == 0.4
    assert shown(c, CS + ("implicit relax",)) == 0.6
    assert CS + ("turbulence relax",) not in c.v2
    assert shown(c, EQ + ("turbulence", "les")) == "wale"
    assert not c.unclaimed()
    validate(c)


def test_E8_compressible_deck_writes_nothing_under_convergence_control():
    c = convert(toy("rans_sst"))
    assert not [p for p in c.v2 if p[:3] == ("solver", "convergence control", "scheme")]


def test_E9_incompressible_only_keys_move_to_equations():
    c = convert(toy("incomp_rans"))
    assert tree(c, EQ) == {
        "type": "rans",
        "correction": "over_relaxed",
        "correction limiter": 0.4,
        "gradient limiter type": "cell",
        "gradient limiter": 0.6,
        "max second order non orthogonal angle": 80.0,
        "central blend": 0.9,
        "turbulence": {"model": "sst"},
    }
    assert shown(c, NS + ("inviscid flux scheme",)) == "HLLC"


def test_E9_over_relaxed_is_respelled_and_others_move_unchanged():
    c = convert(toy("incomp_rans"))
    assert kinds(c)[("incomp rans", "correction")].kind == "respelled"
    assert kinds(c)[("incomp rans", "correction limiter")].kind == "moved"
    for other in ["none", "orthogonal", "minimum"]:
        c = convert(
            with_equation(
                "incomp viscous",
                {"order": "second", "correction": other},
            )
        )
        assert kinds(c)[("incomp viscous", "correction")].kind == "moved"
        validate(c)


def test_E9_roe_entropy_fix_coefficient_at_default_is_dropped():
    params = with_equation(
        "incomp rans",
        {
            "order": "second",
            "roe entropy fix coefficient": 0.001,
            "turbulence": {"model": "sst"},
        },
    )
    c = convert(params)
    gone = kinds(c)[("incomp rans", "roe entropy fix coefficient")]
    assert gone.kind == "dropped"
    assert "default" in gone.reason
    assert not c.unclaimed()


def test_E9_roe_entropy_fix_coefficient_off_default_is_unclaimed():
    params = with_equation(
        "incomp les",
        {
            "order": "second",
            "roe entropy fix coefficient": 0.01,
            "turbulence": {"les": "none"},
        },
    )
    c = convert(params)
    assert unclaimed(c) == {("incomp les", "roe entropy fix coefficient")}


def test_E9_compressible_roe_entropy_fix_coefficient_moves():
    c = convert(
        with_equation("euler", {"order": "first", "roe entropy fix coefficient": 0.01})
    )
    assert shown(c, NS + ("roe entropy fix coefficient",)) == 0.01


def test_E10_material_air_and_viscous_block():
    c = convert(toy("incomp_viscous"))
    assert tree(c, FP) == {
        "material": "ideal gas",
        "gamma": 1.4,
        "gas constant": 287.0,
        "sutherlands const": 110.4,
        "prandtl no": 0.72,
        "turbulent prandtl no": 0.9,
    }
    assert kinds(c)[("air", "sutherlands const")].v2_path == FP + ("sutherlands const",)
    assert not c.unclaimed()
    validate(c)


def test_E10_material_air_is_respelled_ideal_gas():
    c = convert({"equations": "viscous", "material": "air", "air": {}})
    assert kinds(c)[("material",)].kind == "respelled"
    assert c.v2[FP + ("material",)] == Value("ideal gas")


def test_E10_a_material_other_than_air_moves_unchanged():
    c = convert(toy("material_water"))
    assert kinds(c)[("material",)].kind == "moved"
    assert isinstance(c.v2[FP + ("material",)], Source)


def test_E10_dynamic_claim_follows_the_material_name():
    c = convert(toy("material_water"))
    assert tree(c, FP) == {
        "material": "water",
        "gamma": 1.01,
        "gas constant": 461.5,
        "prandtl no": 7.0,
    }
    assert not c.unclaimed()
    validate(c)


def test_E10_dynamic_claim_leaves_an_unknown_material_key_unclaimed():
    params = toy("material_water")
    params["water"]["viscosity law"] = "power"
    c = convert(params)
    assert unclaimed(c) == {("water", "viscosity law")}


def test_E10_a_block_that_is_not_the_material_is_not_claimed():
    params = toy("material_water")
    params["argon"] = {"gamma": 1.67}
    c = convert(params)
    assert unclaimed(c) == {("argon", "gamma")}


def test_E11_euler_drops_turbulent_prandtl():
    c = convert(toy("euler_scheme_keys"))
    gone = kinds(c)[("air", "turbulent prandtl no")]
    assert gone.kind == "dropped"
    assert "inviscid" in gone.reason
    assert FP + ("turbulent prandtl no",) not in c.v2
    validate(c)


def test_E11_viscous_moves_turbulent_prandtl():
    c = convert(toy("les_wale"))
    assert shown(c, FP + ("turbulent prandtl no",)) == 0.9


def test_E12_gravity_on_incompressible_moves():
    c = convert(toy("incomp_rans"))
    assert shown(c, FP + ("gravity",)) == [0.0, 0.0, -9.81]
    assert kinds(c)[("air", "gravity")].kind == "moved"
    assert not c.decisions
    validate(c)


def test_E12_gravity_on_compressible_is_a_decision():
    params = with_equation("viscous", {"order": "second"})
    params["air"] = {"gravity": [0.0, 0.0, -9.81]}
    c = convert(params)
    assert len(c.decisions) == 1
    decision = c.decisions[0]
    assert decision.v2_path == FP + ("gravity",)
    assert "incompressible" in decision.question
    assert isinstance(c.v2[FP + ("gravity",)], Placeholder)
    assert kinds(c)[("air", "gravity")].kind == "decide"


@pytest.mark.parametrize(
    "key, value",
    [
        ("a1", 0.31),
        ("ca1", 2.0),
        ("ca2", 0.06),
        ("cd1", 20.0),
        ("cd2", 3),
        ("cdes", 0.65),
        ("cdes_keps", 0.61),
        ("cdes_kw", 0.78),
        ("ce1", 1.0),
        ("ce2", 50.0),
        ("cthetat", 0.03),
        ("cw", 0.15),
        ("qcr", 2021),
        ("separation correction", True),
        ("sigmagamma", 1.0),
        ("sigmathetat", 2.0),
    ],
)
def test_sweep_turbulence_constants_move_to_equations_turbulence(key, value):
    for name in ("rans", "incomp rans"):
        block = {"order": "second", "turbulence": {"model": "sst", key: value}}
        c = convert(with_equation(name, block))
        assert shown(c, EQ + ("turbulence", norm_key(key))) == value
        assert not c.unclaimed()


@pytest.mark.parametrize(
    "key, value",
    [
        ("correction", "orthogonal"),
        ("correction limiter", 0.3),
        ("gradient limiter", 0.3),
        ("gradient limiter type", "cell"),
        ("max second order non orthogonal angle", 45.0),
    ],
)
def test_sweep_incompressible_keys_on_every_incompressible_set(key, value):
    for name in ("incomp viscous", "incomp rans", "incomp les"):
        block = {"order": "second", key: value}
        if name == "incomp rans":
            block["turbulence"] = {"model": "sst"}
        if name == "incomp les":
            block["turbulence"] = {"les": "none"}
        c = convert(with_equation(name, block))
        assert shown(c, EQ + (key,)) == value
        assert not c.unclaimed()


@pytest.mark.parametrize("name", ["euler", "viscous", "rans"])
def test_sweep_freeze_limiter_cycle_and_backward_difference(name):
    block = {
        "order": "second",
        "freeze limiter cycle": 300,
        "permit backward difference during reconstruction": True,
    }
    if name == "rans":
        block["turbulence"] = {"model": "sst"}
    c = convert(with_equation(name, block))
    assert shown(c, NS + ("freeze limiter cycle",)) == 300
    assert shown(c, NS + ("permit backward difference during reconstruction",)) is True
    assert not c.unclaimed()
    validate(c)


def test_sweep_gradient_limiter_keys_are_not_numerical_scheme_keys():
    c = convert(toy("incomp_rans"))
    assert not [
        p for p in c.v2 if p[:2] == NS and "limiter" in p[-1] and p[-1] != "limiter"
    ]


def test_a_deck_with_every_owned_key_leaves_nothing_unclaimed_but_the_gaps():
    params = toy("rans_sst")
    params["RANS"]["turbulence"]["approximate sa from sst results"] = False
    c = convert(params)
    assert unclaimed(c) == {("rans", "turbulence", "approximate sa from sst results")}


def test_module_contract():
    assert equations.ORDER == 20
    assert set(equations.OWNS) >= {
        "equations",
        "euler",
        "viscous",
        "rans",
        "les",
        "incomp viscous",
        "incomp rans",
        "incomp les",
        "material",
        "air",
    }


def test_foreign_keys_are_never_read_or_claimed():
    params = toy("incomp_viscous")
    params["time marching"] = {"cycles": 10}
    params["IC_1"] = {"temperature": 300.0}
    c = convert(params)
    assert unclaimed(c) == {("time marching", "cycles"), ("ic 1", "temperature")}
