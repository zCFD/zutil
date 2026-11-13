"""Deck functions move to v2 with a def line v2 can call them with.

Each toy deck in decks/functions holds one v1 function form. A form that
already takes the v2 keywords moves unchanged; a form whose v1 arguments map
onto the v2 keywords mechanically is rewritten in place; any other form is
kept as written beside a decision that names the def line, and the key
carries the DECIDE placeholder until the engineer restores the name.
"""

from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path

import pytest

from zutil.update_input.convert import convert_deck, in_deck_dir
from zutil.update_input.model import (
    Conversion,
    Leaf,
    MeshInfo,
    Placeholder,
    Source,
    V1Deck,
)
from zutil.update_input.rules import conditions, zones
from zutil.validate.schemas.callables import (
    INITIAL_CONDITIONS,
    LEAF_AREA_DENSITY,
    MESH_DEFORMATION,
    MOMENTUM_SOURCE,
)

DECKS = Path(__file__).parent / "decks" / "functions"
FZ = ("model", "m", "fluid zones")


def flatten(node: dict, keys: tuple, out: list[Leaf]) -> list[Leaf]:
    for key, value in node.items():
        here = keys + (key,)
        if isinstance(value, dict):
            flatten(value, here, out)
        else:
            source = getattr(value, "__name__", None) if callable(value) else None
            out.append(Leaf(here, value, source))
    return out


def load(name: str) -> Conversion:
    path = (DECKS / f"{name}.py").resolve()
    with in_deck_dir(path):
        parameters = runpy.run_path(str(path))["parameters"]
    sys.modules.pop("motions", None)
    c = Conversion(V1Deck(flatten(parameters, (), []), parameters), "m")
    c.deck_files = (path,)
    return c


def zone(name: str) -> Conversion:
    c = load(name)
    c.mesh = MeshInfo("m.h5", "m", {}, [0])
    zones.convert(c)
    return c


def placeholder(c: Conversion, decision) -> str:
    value = c.v2[decision.v2_path]
    assert isinstance(value, Placeholder)
    assert value.decision_id == decision.id
    return value.text


def rewritten(c: Conversion) -> str:
    [rewrite] = c.rewrites
    return rewrite.text


IC_PATH = ("solver", "initialisation", "initial conditions")


def initial(name: str) -> Conversion:
    c = load(name)
    conditions.convert(c)
    return c


@pytest.mark.parametrize(
    "deck, name, def_line",
    [
        (
            "initial_explicit",
            "steady",
            "def steady(*, location, wall_distance, pressure, temperature, velocity):",
        ),
        ("initial_get", "shock", "def shock(**kwargs):"),
        ("initial_kwargs", "shock", "def shock(**kwargs):"),
        ("initial_named", "surge", "def surge(location, **kwargs):"),
        ("initial_passes_kwargs", "shock", "def shock(**kwargs):"),
        ("initial_reads_a_global", "ramp", "def ramp(**kwargs):"),
        ("initial_recursive", "settle", "def settle(**kwargs):"),
        ("initial_unknown_key", "gust", "def gust(**kwargs):"),
    ],
)
def test_a_function_that_takes_the_v2_keywords_moves_unchanged(deck, name, def_line):
    c = initial(deck)
    assert c.decisions == []
    assert c.rewrites == []
    assert c.v2[IC_PATH] == Source(c.v1.leaf("initial", "func"))
    [note] = [n for n in c.notes if n.text.startswith(f"{name}:")]
    returns = "" if name == "gust" else f", and it returns {INITIAL_CONDITIONS.reads}"
    assert note.text == (
        f"{name}: `{def_line}` already takes the keywords v2 calls it with{returns}"
    )
    assert note.v1_paths == (("initial", "func"),)


def test_an_initial_function_v1_could_not_call_is_a_decision():
    c = initial("initial_positional")
    [decision] = c.decisions
    assert decision.v2_path == IC_PATH
    question = f"rewrite lurch as {INITIAL_CONDITIONS.definition('lurch')}"
    assert decision.question == question
    assert "cannot take the keywords v1 passed, which v2 passes too" in decision.details
    assert "v1 called it as func(**kwargs) with pressure" in decision.details
    assert c.rewrites == []
    assert placeholder(c, decision) == (
        f"DECIDE: {question} (see conversion report, item 1)"
    )


@pytest.mark.parametrize(
    "deck, name",
    [
        ("translation_kwargs", "linear_accel"),
        ("translation_named", "surge"),
        ("translation_explicit", "steady"),
        ("translation_passes_kwargs", "rotor_motion"),
        ("translation_unknown_key", "gust"),
        ("translation_reads_a_global", "ramp"),
        ("translation_positional", "lurch"),
        ("translation_recursive", "settle"),
    ],
)
def test_a_translation_function_is_never_ported(deck, name):
    """v2 has no translation function: whatever its form, it becomes a velocity decision."""
    c = zone(deck)
    [decision] = c.decisions
    assert decision.v2_path == ("model", "m", "transforms", "motion", "velocity")
    assert decision.question.startswith(
        f"A v2 control deck carries no functions. '{name}' was the translation "
        "function of fluid zone 'FZ_1'."
    )
    assert c.rewrites == []
    assert not [n for n in c.notes if name in n.text]
    assert not [p for p in c.v2 if "translation function" in p]


def test_three_positional_coordinates_unpack_the_v2_location():
    c = zone("rbf_positional")
    assert [d.v2_path for d in c.decisions] == []
    assert rewritten(c) == (
        "def bend(*, location):\n"
        '    """Rotate the surface about the z axis."""\n'
        "    x, y, z = location\n"
        "    c, s = math.cos(0.1), math.sin(0.1)\n"
        '    return {"v1": c * x - s * y, "v2": s * x + c * y, "v3": z}'
    )
    [note] = [n for n in c.notes if "bend" in n.text]
    assert note.text == (
        "bend: `def bend(x, y, z):` is now `def bend(*, location):`; its body "
        "still returns the v1/v2/v3 dict, which v2 accepts"
    )


def test_an_idw_function_keeps_its_own_coordinate_names():
    c = zone("idw_positional")
    assert c.decisions == []
    assert rewritten(c) == (
        "def lift(*, location):\n"
        "    px, py, pz = location\n"
        "    return (px, py, pz + 0.01)"
    )
    [note] = [n for n in c.notes if "lift" in n.text]
    assert note.text == (
        "lift: `def lift(px, py, pz):` is now `def lift(*, location):`, and it "
        f"returns {MESH_DEFORMATION.reads}"
    )


@pytest.mark.parametrize(
    "deck, name, reason",
    [
        ("idw_reads_location", "shift", "reads location from its deck"),
        ("idw_one_argument", "sway", "takes 1 argument"),
        ("idw_from_module", "heave", "is defined in motions.py"),
        ("idw_varargs", "nudge", "takes *args, **kwargs or keywords"),
    ],
)
def test_a_mesh_deformation_function_with_no_mechanical_rewrite_is_a_decision(
    deck, name, reason
):
    c = zone(deck)
    [decision] = c.decisions
    assert decision.v2_path == FZ + ("TR_1", "func")
    assert decision.question == f"rewrite {name} as {MESH_DEFORMATION.definition(name)}"
    assert reason in decision.details
    assert "v1 called it as func(x, y, z)" in decision.details
    assert c.rewrites == []
    assert placeholder(c, decision) == (
        f"DECIDE: rewrite {name} as {MESH_DEFORMATION.definition(name)} "
        "(see conversion report, item 1)"
    )


def test_a_function_another_deck_function_calls_is_a_decision():
    c = zone("idw_called_by_helper")
    [decision] = c.decisions
    assert decision.v2_path == FZ + ("TR_1", "func")
    assert (
        decision.question
        == f"rewrite rotate as {MESH_DEFORMATION.definition('rotate')}"
    )
    assert "rotate is called by other code in its deck" in decision.details
    assert rewritten(c) == (
        "def rotate_twice(*, location):\n"
        "    x, y, z = location\n"
        "    return rotate(*rotate(x, y, z))"
    )


def test_one_function_on_two_zones_is_rewritten_once():
    c = zone("shared")
    assert c.decisions == []
    assert len(c.rewrites) == 1
    assert FZ + ("TR_1", "func") in c.v2 and FZ + ("TR_2", "func") in c.v2
    [note] = [n for n in c.notes if "bend" in n.text]
    assert note.v1_paths == (("tr 1", "func"), ("tr 2", "func"))


def test_a_momentum_function_renames_its_argument_to_locations():
    c = zone("momentum_positional")
    assert c.decisions == []
    assert rewritten(c) == (
        "def drive(*, locations):\n    return [(1.0, 0.0, 0.0) for _ in locations]"
    )


def test_a_momentum_argument_a_lambda_rebinds_is_a_decision():
    c = zone("momentum_shadowed")
    [decision] = c.decisions
    assert (
        decision.question == f"rewrite drive as {MOMENTUM_SOURCE.definition('drive')}"
    )
    assert "binds cells again inside the function" in decision.details


def test_a_v1_canopy_function_is_a_decision_naming_both_lists():
    c = zone("canopy_v1")
    [decision] = c.decisions
    assert decision.v2_path == FZ + ("FZ_1", "function")
    assert decision.question == (
        f"rewrite leaf_area as {LEAF_AREA_DENSITY.definition('leaf_area')}"
    )
    assert "v1 called it as func(cells)" in decision.details


@pytest.fixture
def work(tmp_path) -> Path:
    shutil.copytree(DECKS, tmp_path / "functions")
    return tmp_path / "functions"


def test_a_deck_with_every_zone_function_moved_validates(work, stand_in_mesh):
    result = convert_deck(work / "every_function.py", stand_in_mesh(work / "m.h5"))
    assert result.conversion.decisions == []
    assert result.validated is True
    for line in (
        "def bend(*, location):",
        "    x, y, z = location",
        "def drive(*, locations):",
        "def shock(**kwargs):",
        "def wind_axes(*, vector):",
    ):
        assert line in result.text.splitlines()


def test_a_force_transform_a_deformation_calls_stays_as_written(work, stand_in_mesh):
    deck = work / "every_function.py"
    deck.write_text(
        deck.read_text().replace(
            '    return {"v1": x, "v2": y, "v3": z}', "    return wind_axes(x, y, z)"
        )
    )
    result = convert_deck(deck, stand_in_mesh(work / "m.h5"))
    [decision] = result.conversion.decisions
    assert decision.question == (
        "rewrite wind_axes as def wind_axes(*, vector), returning the "
        "transformed vector as three numbers"
    )
    lines = result.text.splitlines()
    assert "def wind_axes(x, y, z):" in lines
    assert "def bend(*, location):" in lines
    assert "    return wind_axes(x, y, z)" in lines


def test_a_function_a_base_defines_is_rewritten_in_the_base(work, stand_in_mesh):
    result = convert_deck(work / "chain" / "variant.py", stand_in_mesh(work / "m.h5"))
    assert result.conversion.decisions == []
    assert result.validated is True
    [base] = result.bases
    assert "def bend(*, location):" in base.text.splitlines()
    assert "def bend(*, location):" not in result.text
