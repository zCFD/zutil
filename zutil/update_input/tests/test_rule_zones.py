"""Tests for the zones rule: fluid zones, mesh deformation, mesh placement."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from pydantic import TypeAdapter

from zutil.update_input.model import (
    Conversion,
    Expr,
    Leaf,
    MeshInfo,
    Placeholder,
    Source,
    V1Deck,
    Value,
)
from zutil.update_input.rules import zones
from zutil.validate.schemas.fluid_zones_schemas import Fluid_Zone_Base_Model
from zutil.validate.schemas.transform_schemas import TransformsModel
from zutil.validate.validation_utils import replace_spaces_in_keys

DECKS = Path(__file__).parent / "decks" / "zones"
MODEL = "m"
FZ = ("model", MODEL, "fluid zones")
TF = ("model", MODEL, "transforms")
MOTION = TF + ("motion",)
ROTOR = {
    "type": "rotating",
    "zone": [3],
    "axis": [0, 0, 1],
    "origin": [0, 0, 0],
    "omega": 2.0,
}


def flatten(
    node: dict, keys: tuple, sources: dict[str, str], out: list[Leaf]
) -> list[Leaf]:
    for key, value in node.items():
        here = keys + (key,)
        if isinstance(value, dict):
            flatten(value, here, sources, out)
        else:
            out.append(Leaf(here, value, sources.get(" > ".join(here))))
    return out


def load(name: str, sources: Optional[dict[str, str]] = None) -> Conversion:
    path = (DECKS / f"{name}.py").resolve()
    parameters = runpy.run_path(str(path))["parameters"]
    leaves = flatten(parameters, (), sources or {}, [])
    c = Conversion(V1Deck(leaves, parameters), MODEL)
    c.deck_files = (path,)
    return c


def build(d: dict) -> Conversion:
    leaves = flatten(d, (), {}, [])
    return Conversion(V1Deck(leaves, d), MODEL)


def run(c: Conversion) -> Conversion:
    zones.convert(c)
    return c


def plain(value: Any) -> Any:
    if isinstance(value, Source):
        return value.leaf.value
    if isinstance(value, Placeholder):
        return value.text
    if isinstance(value, (Value, Expr)):
        return value.value
    return value


def fragment(c: Conversion, *prefix: str) -> dict:
    out: dict = {}
    for path, value in c.v2.items():
        if path[: len(prefix)] != prefix:
            continue
        node = out
        for key in path[len(prefix) : -1]:
            node = node.setdefault(key, {})
        node[path[-1]] = plain(value)
    return out


def validate_zone(c: Conversion, label: str) -> Any:
    zone = fragment(c, *FZ, label)
    return TypeAdapter(Fluid_Zone_Base_Model).validate_python(
        replace_spaces_in_keys(zone)
    )


def unclaimed_paths(c: Conversion) -> set[tuple[str, ...]]:
    return {lf.path for lf in c.unclaimed()}


def test_owns_and_order():
    assert zones.ORDER == 50
    assert set(zones.OWNS) == {
        "fz *",
        "tr *",
        "scale",
        "mesh transform matrix",
        "mesh transform function",
    }


def with_mesh(c: Conversion, cell_zones: list[int]) -> Conversion:
    c.mesh = MeshInfo("m.h5", MODEL, {}, cell_zones)
    return c


OVERSET = {"BC_1": {"type": "overset", "zone": [7]}}
MOVING_MESH = {"time marching": {"moving mesh": True}}


def validate_transforms(c: Conversion) -> TransformsModel:
    return TransformsModel.model_validate(replace_spaces_in_keys(fragment(c, *TF)))


def test_z1_rotating_zone_on_a_still_mesh_is_a_rotating_fluid_zone():
    c = run(load("rotating", {"FZ_1 > omega": "omega"}))
    assert fragment(c, *FZ, "FZ_1") == {
        "type": "rotating",
        "cell zones": [17],
        "axis": [0, 0, 1],
        "origin": [0, 0, 0],
        "omega": 2.0,
    }
    assert c.v2[FZ + ("FZ_1", "omega")] == Source(c.v1.leaf("FZ_1", "omega"))
    assert not [p for p in c.v2 if p[:4] == MOTION]
    assert not c.unclaimed()
    assert not c.decisions
    zone = validate_zone(c, "FZ_1")
    assert zone.type == "rotating" and zone.cell_zones == [17]


@pytest.mark.parametrize(
    "cell_zones", [None, [], [0, 3]], ids=["no mesh", "no cell zones", "part"]
)
def test_z1_a_rotating_zone_on_part_of_a_mesh_keeps_its_cell_zones(cell_zones):
    c = build({"FZ_1": ROTOR})
    if cell_zones is not None:
        with_mesh(c, cell_zones)
    run(c)
    assert fragment(c, *FZ, "FZ_1")["cell zones"] == [3]
    assert not [p for p in c.v2 if p[:4] == MOTION]
    assert not c.decisions and not c.notes
    assert not c.unclaimed()


@pytest.mark.parametrize("spelling", ["origin", "centre", "center"])
def test_z1_a_rotating_zone_on_the_whole_mesh_is_the_mesh_frame(spelling):
    rotor = {k: v for k, v in ROTOR.items() if k != "origin"}
    c = run(with_mesh(build({"FZ_1": {**rotor, spelling: [1, 2, 3]}}), [3]))
    assert fragment(c, *MOTION) == {
        "type": "rotating",
        "moving mesh": False,
        "axis": [0, 0, 1],
        "origin": [1, 2, 3],
        "omega": 2.0,
    }
    assert not fragment(c, *FZ)
    assert [n.text for n in c.notes] == [
        "Fluid zone FZ_1 covers the whole mesh, and so does a frame in "
        "'transforms > motion'."
    ]
    assert not c.decisions
    assert not c.unclaimed()
    motion = validate_transforms(c).motion
    assert motion.type == "rotating" and motion.moving_mesh is False


def test_z1_a_rotating_zone_on_the_whole_mesh_beside_another_is_one_decision():
    c = run(
        with_mesh(
            build({"FZ_1": dict(ROTOR, zone=[3, 4]), "FZ_2": dict(ROTOR, zone=[4])}),
            [3, 4],
        )
    )
    [decision] = c.decisions
    assert decision.v2_path == MOTION
    assert not fragment(c, *FZ)
    assert not c.unclaimed()


@pytest.mark.parametrize("spelling", ["centre", "center"])
def test_z1_the_centre_of_a_rotating_zone_is_its_origin(spelling):
    rotor = {k: v for k, v in ROTOR.items() if k != "origin"}
    c = run(build({"FZ_1": {**rotor, spelling: [1, 2, 3]}}))
    assert fragment(c, *FZ, "FZ_1")["origin"] == [1, 2, 3]
    assert not c.unclaimed()


def test_z1_two_rotating_zones_on_one_mesh_are_two_fluid_zones():
    c = run(build({"FZ_1": ROTOR, "FZ_2": dict(ROTOR, zone=[4], omega=-2.0)}))
    assert fragment(c, *FZ, "FZ_1")["cell zones"] == [3]
    assert fragment(c, *FZ, "FZ_2")["cell zones"] == [4]
    assert not c.decisions
    assert not c.unclaimed()


def test_z1_a_rotating_zone_selected_by_a_definition_is_a_decision():
    def inside(x, y, z):
        return x < 1.0

    zone = {k: v for k, v in ROTOR.items() if k != "zone"}
    c = run(build({"FZ_1": {**zone, "def": inside}}))
    [decision] = c.decisions
    assert decision.v2_path == FZ + ("FZ_1", "cell zones")
    assert decision.v1_paths == (("fz 1", "def"),)
    assert decision.question == (
        "Fluid zone FZ_1 rotates the cells its definition selects, and a v2 "
        "rotating fluid zone selects its cells by 'cell zones' only. Which cell "
        "zones does it rotate?"
    )
    assert fragment(c, *FZ, "FZ_1")["omega"] == 2.0
    assert not c.unclaimed()


def test_z1_a_rotating_zone_that_selects_no_cells_is_a_decision():
    zone = {k: v for k, v in ROTOR.items() if k != "zone"}
    c = run(build({"FZ_1": zone}))
    [decision] = c.decisions
    assert decision.v2_path == FZ + ("FZ_1", "cell zones")
    assert decision.v1_paths == ()
    assert decision.question == (
        "Fluid zone FZ_1 gives neither 'zone' nor 'def', and a v2 rotating fluid "
        "zone selects its cells by 'cell zones' only. Which cell zones does it "
        "rotate?"
    )
    assert not c.unclaimed()


def test_z1_a_ramp_on_a_rotating_zone_is_dropped_because_v1_never_read_it():
    c = run(build({"FZ_1": {**ROTOR, "ramp": {"initial": 0.0, "time": 1.0}}}))
    assert c.decisions == []
    ramp = [d for d in c.dispositions if d.v1_path[:2] == ("fz 1", "ramp")]
    assert {d.v1_path[2] for d in ramp} == {"initial", "time"}
    assert {(d.kind, d.reason) for d in ramp} == {("dropped", zones.V1_ZONE_RAMP)}
    assert fragment(c, *FZ, "FZ_1")["cell zones"] == [3]
    assert not c.unclaimed()


WHOLE_MESH_NOTE = (
    "Fluid zone FZ_1 covers the whole mesh, and so does a frame in "
    "'transforms > motion'."
)
SLIDE = {"type": "translating", "zone": [3], "vector": [1, 0, 0]}


def test_z1_a_translating_frame_on_zone_0_of_an_unread_mesh_is_a_decision():
    c = run(build({"FZ_1": dict(SLIDE, zone=[0])}))
    [decision] = c.decisions
    assert decision.v2_path == MOTION + ("zones",)
    assert decision.v1_paths == (("fz 1", "zone"),)
    assert not c.notes


@pytest.mark.parametrize("bcs", [{}, OVERSET], ids=["no interface", "overset"])
def test_z1_a_translating_frame_on_the_whole_mesh_takes_no_zones(bcs):
    c = run(with_mesh(build({**bcs, "FZ_1": dict(SLIDE, zone=[3, 4])}), [3, 4]))
    assert MOTION + ("zones",) not in c.v2
    assert c.v2[MOTION + ("moving mesh",)] == Value(False)
    assert [n.text for n in c.notes] == [WHOLE_MESH_NOTE]
    assert not c.decisions


PART_MESH_QUESTION = (
    "Fluid zone FZ_1 translates {selected} only, and 'transforms > motion' "
    "covers the whole mesh. Should the frame cover the whole mesh, or should "
    "the translating region be a mesh of its own?"
)
PART_MESH_OPTIONS = (
    "cover the whole mesh",
    "give the region its own mesh, joined by a sliding or overset interface",
)
PART_MESH_FIX = (
    "To cover the whole mesh, remove 'zones'. Otherwise split the region into "
    "a mesh of its own, with its own model, joined to the rest by a sliding or "
    "overset interface."
)


@pytest.mark.parametrize("bcs", [{}, OVERSET], ids=["no interface", "overset"])
def test_z1_a_translating_frame_on_part_of_a_mesh_is_a_decision(bcs):
    c = run(with_mesh(build({**bcs, "FZ_1": SLIDE}), [0, 3]))
    [decision] = c.decisions
    assert decision.v2_path == MOTION + ("zones",)
    assert decision.v1_paths == (("fz 1", "zone"),)
    assert decision.question == PART_MESH_QUESTION.format(selected="cell zones [3]")
    assert decision.options == PART_MESH_OPTIONS
    assert decision.details == f"The mesh has cell zones [0, 3]. {PART_MESH_FIX}"
    assert c.v2[MOTION + ("velocity",)] == Source(c.v1.leaf("FZ_1", "vector"))


ROTATING_PART_MESH_QUESTION = (
    "Fluid zone FZ_1 rotates {selected} only, and 'transforms > motion' covers "
    "the whole mesh. Should the frame cover the whole mesh, or should the "
    "rotating region be a mesh of its own?"
)


def test_z1_a_rotating_zone_on_an_overset_mesh_covering_it_is_the_mesh_frame():
    c = run(with_mesh(build({**OVERSET, "FZ_1": ROTOR}), [3]))
    assert fragment(c, *MOTION)["type"] == "rotating"
    assert c.v2[MOTION + ("moving mesh",)] == Value(False)
    assert FZ + ("FZ_1", "type") not in c.v2
    assert not c.decisions


@pytest.mark.parametrize("cell_zones", [None, [0, 3]], ids=["unread", "part"])
def test_z1_a_rotating_zone_on_part_of_an_overset_mesh_is_a_decision(cell_zones):
    c = build({**OVERSET, "FZ_1": ROTOR})
    if cell_zones is not None:
        c = with_mesh(c, cell_zones)
    run(c)
    [decision] = c.decisions
    assert decision.v2_path == MOTION + ("zones",)
    assert decision.question == ROTATING_PART_MESH_QUESTION.format(
        selected="cell zones [3]"
    )
    assert FZ + ("FZ_1", "type") not in c.v2


def test_z1_a_translating_frame_on_a_mesh_update_input_cannot_read_is_a_decision():
    c = run(build({"FZ_1": SLIDE}))
    [decision] = c.decisions
    assert decision.question == PART_MESH_QUESTION.format(selected="cell zones [3]")
    assert decision.details == (
        "update_input reads the cell zones from the mesh given with --mesh, and "
        f"none was read. {PART_MESH_FIX}"
    )


def test_z1_a_translating_frame_selected_by_a_definition_is_a_decision():
    def inside(x, y, z):
        return x < 1.0

    zone = {k: v for k, v in SLIDE.items() if k != "zone"}
    c = run(build({"FZ_1": {**zone, "def": inside}}))
    [decision] = c.decisions
    assert decision.v2_path == MOTION + ("zones",)
    assert decision.v1_paths == (("fz 1", "def"),)
    assert decision.question == PART_MESH_QUESTION.format(
        selected="the cells its definition selects"
    )
    assert not c.unclaimed()


def test_z1_a_rotating_and_a_translating_frame_on_one_mesh_are_one_decision():
    c = run(
        build(
            {
                "FZ_1": ROTOR,
                "FZ_2": {"type": "translating", "zone": [4], "vector": [1, 0, 0]},
            }
        )
    )
    [decision] = c.decisions
    assert decision.v2_path == MOTION
    assert decision.question == (
        "v2 gives a mesh one 'motion', and fluid zones FZ_1, FZ_2 each move a "
        "frame on mesh m. Which is the motion of the mesh? Split the mesh into "
        "one model per frame to keep them all."
    )
    assert "FZ_1: " in decision.details and "FZ_2: " in decision.details
    assert not c.unclaimed()


def test_z2_translating_vector_only_becomes_a_translating_frame_velocity():
    c = run(with_mesh(load("translating_vector"), [0]))
    assert fragment(c, *MOTION) == {
        "type": "translating",
        "moving mesh": False,
        "velocity": [-1.0, 0.0, 0.0],
    }
    assert not c.unclaimed()
    motion = validate_transforms(c).motion
    assert motion.type == "translating"
    assert motion.velocity == [-1.0, 0.0, 0.0]


def test_z2_open_question_5_mach_computes_velocity_expression():
    sources = {
        "FZ_1 > mach": "mach",
        "IC_1 > temperature": "T_inf",
        "air > gamma": "gamma",
        "air > gas constant": "R_gas",
    }
    c = run(load("translating_mach", sources))
    velocity = c.v2[MOTION + ("velocity",)]
    assert isinstance(velocity, Expr)
    names = {"mach": 0.2, "T_inf": 300.0, "gamma": 1.4, "R_gas": 287.0}
    assert eval(velocity.text, dict(names)) == pytest.approx(velocity.value)
    speed = 0.2 * (1.4 * 287.0 * 300.0) ** 0.5
    assert velocity.value == pytest.approx([-speed, 0.0, 0.0])
    for name in names:
        assert name in velocity.text
    dispositions = {d.v1_path: d for d in c.dispositions}
    assert dispositions[("fz 1", "vector")].v2_path == MOTION + ("velocity",)
    assert dispositions[("fz 1", "mach")].v2_path == MOTION + ("velocity",)
    assert any("speed of sound" in n.text for n in c.notes)


@pytest.mark.parametrize("vector", [[2.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
def test_z2_mach_scales_the_unit_vector_whatever_its_length(vector):
    deck = runpy.run_path(str(DECKS / "translating_mach.py"))["parameters"]
    deck["FZ_1"]["vector"] = vector
    c = run(build(deck))
    velocity = c.v2[MOTION + ("velocity",)]
    speed = 0.2 * (1.4 * 287.0 * 300.0) ** 0.5
    length = float(np.linalg.norm(vector))
    assert velocity.value == pytest.approx([speed * x / length for x in vector])
    assert eval(velocity.text, {"mach": 0.2}) == pytest.approx(velocity.value)


def test_z2_open_question_5_mach_without_material_is_decide():
    c = run(with_mesh(load("translating_mach_no_material"), [0]))
    assert len(c.decisions) == 1
    decision = c.decisions[0]
    assert decision.v2_path == MOTION + ("velocity",)
    assert isinstance(c.v2[decision.v2_path], Placeholder)
    assert ("fz 1", "mach") in decision.v1_paths


def test_z2_a_translation_function_on_a_frame_asks_for_its_velocity():
    c = run(with_mesh(load("translating_function"), [0]))
    [decision] = c.decisions
    assert decision.v2_path == MOTION + ("velocity",)
    assert decision.question == (
        "A v2 control deck carries no functions. 'linear_accel' was the "
        "translation function of fluid zone 'FZ_1'. A frame has a fixed "
        "velocity: give 'velocity' in 'transforms > motion' in m/s."
    )
    assert 'func(**{"time": t, "time step": dt})' in decision.details
    assert "[-69.4, 0.0, 0.0]" in decision.details
    assert set(decision.v1_paths) == {
        ("fz 1", "translation function"),
        ("fz 1", "vector"),
    }
    assert c.rewrites == []
    assert MOTION + ("translation function",) not in c.v2
    assert not c.unclaimed()


def test_z2_a_zone_the_motion_took_is_left_alone():
    c = build({**MOVING_MESH, "FZ_1": load("translating_function").v1.value("FZ_1")})
    zones.convert_motion(c, "FZ_1", c.v1.leaf("time marching", "moving mesh"))
    claimed = set(c.v2)
    run(c)
    assert set(c.v2) == claimed
    assert not [p for p in c.v2 if p[:3] == FZ]
    assert zones.moving_zones(c) == ["FZ_1"]


def test_z2_a_moving_mesh_decision_already_written_is_kept():
    c = load("rotating")
    c.set(MOTION + ("moving mesh",), Placeholder(1, "DECIDE: item 1"))
    run(c)
    assert fragment(c, *MOTION)["moving mesh"] == "DECIDE: item 1"
    assert fragment(c, *MOTION)["omega"] == 2.0


def test_z3_canopy_def_func_constants_function_moves():
    c = run(load("canopy", {"FZ_1 > func": "leaf_area"}))
    base = FZ + ("FZ_1",)
    assert base + ("definition",) in c.v2
    assert base + ("cell zones",) not in c.v2
    for key in ("cd", "beta p", "beta d", "ceps 4", "ceps 5"):
        assert base + (key,) in c.v2
    assert not c.decisions
    assert not c.unclaimed()
    function = c.v2[base + ("function",)]
    assert isinstance(function, Source) and function.leaf.source == "leaf_area"
    del c.v2[base + ("function",)]
    zone = validate_zone(c, "FZ_1")
    assert zone.cd == 0.2 and zone.beta_p == 1.0 and zone.ceps_5 == 0.8


def test_z4_momentum_function_moves_as_a_callable_with_its_source():
    c = run(with_mesh(load("momentum", {"FZ_1 > func": "drive"}), [0]))
    base = FZ + ("FZ_1",)
    assert base + ("cell zones",) not in c.v2
    assert not c.decisions
    func = c.v2[base + ("func",)]
    assert isinstance(func, Source) and func.leaf.source == "drive"
    assert not c.unclaimed()
    assert validate_zone(c, "FZ_1").func is func.leaf.value


@pytest.mark.parametrize("cell_zones", [None, [0, 4]], ids=["unread mesh", "part"])
def test_z5_a_source_on_cell_zones_keeps_them(cell_zones):
    c = load("porous_zone")
    if cell_zones is not None:
        with_mesh(c, cell_zones)
    run(c)
    base = FZ + ("FZ_1",)
    assert fragment(c, *base) == {
        "type": "porous",
        "cell zones": [4],
        "alpha": 1.0e6,
        "c2": 10.0,
    }
    assert not c.decisions and not c.notes
    assert not c.unclaimed()
    assert validate_zone(c, "FZ_1").cell_zones == [4]


@pytest.mark.parametrize("kind", ["momentum", "porous", "canopy"])
def test_z5_a_source_on_the_whole_mesh_drops_its_zones(kind):
    c = run(with_mesh(build({"FZ_1": {"type": kind, "zone": [3, 4]}}), [3, 4]))
    base = FZ + ("FZ_1",)
    assert base + ("cell zones",) not in c.v2
    assert base + ("definition",) not in c.v2
    assert not c.decisions
    [d] = [d for d in c.dispositions if d.v1_path == ("fz 1", "zone")]
    assert d.kind == "dropped"
    assert any(
        n.text == "Fluid zone FZ_1 covers the whole mesh, and so does a source "
        "without 'cell zones'."
        for n in c.notes
    )
    assert not c.unclaimed()


def test_z5_a_v1_definition_key_beside_a_zone_is_dropped():
    c = run(
        with_mesh(
            build({"FZ_1": {"type": "porous", "zone": [3], "definition": "p.vtp"}}),
            [0, 3],
        )
    )
    assert not c.decisions
    [d] = [d for d in c.dispositions if d.v1_path == ("fz 1", "definition")]
    assert d.kind == "dropped"
    assert d.reason == "v1 read no 'definition', so it selected no cells"
    assert validate_zone(c, "FZ_1").cell_zones == [3]
    assert not c.unclaimed()


def test_z5_zone_beside_a_definition_is_dropped():
    c = run(build({"FZ_1": {"type": "porous", "zone": [4], "def": "p.vtp"}}))
    assert not c.decisions
    [d] = [d for d in c.dispositions if d.v1_path == ("fz 1", "zone")]
    assert d.kind == "dropped"
    assert validate_zone(c, "FZ_1").definition == "p.vtp"


def test_z5_a_canopy_with_both_def_and_field_is_a_decision():
    c = run(
        build(
            {
                "FZ_1": {
                    "type": "canopy",
                    "zone": [4],
                    "def": "forest.vtp",
                    "field": "lad",
                    "cd": 0.2,
                }
            }
        )
    )
    [decision] = c.decisions
    assert decision.v2_path == FZ + ("FZ_1", "definition")
    assert decision.v1_paths == (("fz 1", "def"), ("fz 1", "field"))
    assert decision.question == (
        "Canopy FZ_1 gives both 'def' and 'field', and a v2 canopy selects its "
        "cells with one of 'cell zones', 'definition' and 'field'. Which selects "
        "the cells?"
    )
    assert decision.options == ("definition", "field")
    assert decision.details == (
        "'def' is 'forest.vtp' and 'field' is 'lad'. Write the chosen one, as "
        "'definition' or 'field', in place of this key."
    )
    assert FZ + ("FZ_1", "field") not in c.v2
    assert FZ + ("FZ_1", "cell zones") not in c.v2
    assert c.v2[FZ + ("FZ_1", "cd")] == Source(c.v1.leaf("FZ_1", "cd"))
    [d] = [d for d in c.dispositions if d.v1_path == ("fz 1", "zone")]
    assert d.kind == "dropped"
    assert not c.unclaimed()


def test_z5_canopy_zone_beside_a_field_is_dropped():
    c = run(build({"FZ_1": {"type": "canopy", "zone": [4], "field": "lad"}}))
    assert not c.decisions
    assert FZ + ("FZ_1", "cell zones") not in c.v2
    assert not c.unclaimed()


def test_z8_a_disc_on_cell_zones_is_a_decision_for_a_definition():
    c = run(build({"FZ_1": {"type": "disc", "zone": [0]}}))
    [decision] = c.decisions
    assert decision.v2_path == FZ + ("FZ_1", "definition")
    assert decision.question == (
        "Disc FZ_1 acts on cell zones [0], and a v2 disc selects its cells with "
        "its geometry and 'definition'. Give a 'definition' VTP that selects "
        "the disc region."
    )


def test_z5_porous_with_def():
    c = run(load("porous_def"))
    base = FZ + ("FZ_1",)
    assert base + ("definition",) in c.v2
    assert base + ("zones",) not in c.v2
    assert not c.unclaimed()
    assert validate_zone(c, "FZ_1").definition == "screen.vtp"


def test_z5_porous_with_def_and_an_empty_zone_list_drops_the_list():
    c = run(
        build(
            {
                "FZ_1": {
                    "zone": [],
                    "def": "porous.vtp",
                    "type": "porous",
                    "C2": 1.0e8,
                    "alpha": 1.0e-8,
                }
            }
        )
    )
    base = FZ + ("FZ_1",)
    assert base + ("zones",) not in c.v2
    assert [d.kind for d in c.dispositions if d.v1_path == ("fz 1", "zone")] == [
        "dropped"
    ]
    assert not c.unclaimed()
    assert validate_zone(c, "FZ_1").definition == "porous.vtp"


def test_z6_abl_zone_is_decide():
    c = run(load("abl", {"FZ_1 > func": "drive"}))
    base = FZ + ("FZ_1",)
    assert len(c.decisions) == 1
    decision = c.decisions[0]
    assert decision.v2_path == base + ("type",)
    assert isinstance(c.v2[base + ("type",)], Placeholder)
    assert ("fz 1", "func") in decision.v1_paths
    assert ("fz 1", "kind") in decision.v1_paths
    assert ("fz 1", "zone") in decision.v1_paths
    assert base + ("zones",) not in c.v2
    assert not c.unclaimed()


def test_z7_an_affine_transform_zone_is_one_decision():
    c = run(load("affine", {"FZ_1 > transform function": "move"}))
    base = FZ + ("FZ_1",)
    [decision] = c.decisions
    assert decision.v2_path == base + ("type",)
    assert decision.question == "v2 has no affine transform zone. What replaces FZ_1?"
    assert set(decision.v1_paths) == {
        ("fz 1", "type"),
        ("fz 1", "zone"),
        ("fz 1", "transform function"),
    }
    assert decision.details == (
        'v1 called move as func(**{"dt": dt, "real time": t}) every step and read '
        "back a 4 x 4 numpy array, which moved the cells of the zone. Prescribed "
        "motion is declarative in v2. A rigid motion of the whole mesh is the "
        "'motion' in the model's 'transforms', "
        """{"type": "rotating", "axis": [...], "origin": [...], "omega": ..., """
        '"moving mesh": True} or {"type": "translating", "velocity": [...], '
        '"moving mesh": True, "ramp time": ...}. A deformation is a mesh '
        "deformation transform whose func is def move(*, location), returning "
        "the deformed position as three numbers. A fixed placement is the "
        "model's 'transforms > transform', a 4 x 4 matrix or a function "
        "def place(*, location), returning the placed position as three numbers."
    )
    assert not c.unclaimed()


def test_z8_disc_simple_model_tsr_curve_controller():
    c = run(load("disc_simple"))
    base = FZ + ("FZ_1",)
    expected = {
        ("type",),
        ("name",),
        ("definition",),
        ("rotation direction",),
        ("reference plane",),
        ("reference point",),
        ("verbose",),
        ("update frequency",),
        ("discretisation", "type"),
        ("discretisation", "number of elements"),
        ("model", "type"),
        ("model", "kind"),
        ("model", "power model"),
        ("model", "power coefficient"),
        ("model", "thrust coefficient"),
        ("controller", "type"),
        ("controller", "pitch"),
        ("controller", "tip speed ratio"),
        ("controller", "tip speed ratio curve"),
        ("geometry", "centre"),
        ("geometry", "normal"),
        ("geometry", "up"),
        ("geometry", "inner radius"),
        ("geometry", "outer radius"),
    }
    assert {p[len(base) :] for p in c.v2} == expected
    assert not c.unclaimed() and not c.decisions
    zone = validate_zone(c, "FZ_1")
    assert zone.controller.type == "tsr curve"
    assert zone.geometry.outer_radius == 60.0


def test_z8_disc_bet_model_with_aerofoils_keeps_user_names():
    c = run(load("disc_bet"))
    base = FZ + ("FZ_1",)
    assert base + ("model", "aerofoils", "Foil_A", "cd") in c.v2
    assert base + ("model", "aerofoils", "Foil_A", "cl") in c.v2
    assert base + ("definition",) in c.v2
    for key in (
        "blade chord",
        "blade twist",
        "number of sections",
        "number of blades",
        "aerofoil positions",
        "tip loss correction",
        "tip loss correction radius",
    ):
        assert base + ("model", key) in c.v2
    assert not c.unclaimed() and not c.decisions
    zone = validate_zone(c, "FZ_1")
    assert zone.model.type == "bet"
    assert "foil_a" in zone.model.aerofoils


def test_z8_disc_controller_type_none_is_decide():
    c = run(load("disc_controller_none"))
    base = FZ + ("FZ_1",)
    assert len(c.decisions) == 1
    decision = c.decisions[0]
    assert decision.v2_path == base + ("controller", "type")
    assert set(decision.options) == {"tsr curve", "fixed", "schedule"}
    assert base + ("controller", "pitch") in c.v2
    assert not c.unclaimed()


def test_z9_disc_status_on_is_dropped():
    c = run(load("disc_status_on"))
    dropped = [d for d in c.dispositions if d.kind == "dropped"]
    assert [d.v1_path for d in dropped] == [("fz 1", "status")]
    assert dropped[0].reason
    assert not c.decisions
    assert not any(p[-1] == "status" for p in c.v2)


def test_z9_disc_status_off_is_decide():
    c = run(load("disc_status_off"))
    assert len(c.decisions) == 1
    assert c.decisions[0].v2_path == FZ + ("FZ_1", "status")
    assert ("fz 1", "status") in c.decisions[0].v1_paths


def test_z9_disc_output_and_turbine_extras_are_unclaimed():
    c = run(load("disc_extras"))
    assert unclaimed_paths(c) == {
        ("fz 1", "output", "frequency"),
        ("fz 1", "auto yaw"),
        ("fz 1", "rated power"),
        ("fz 1", "cut in speed"),
        ("fz 1", "cut out speed"),
        ("fz 1", "inertia"),
    }
    assert not c.decisions


def test_z10_rbf_multiscale():
    c = run(load("rbf_multiscale", {"TR_1 > func": "deform"}))
    base = FZ + ("TR_1",)
    assert c.v2[base + ("type",)] == Value("rbf transform")
    assert base + ("zones",) in c.v2
    assert base + ("support radius",) in c.v2
    assert base + ("alpha",) not in c.v2
    assert base + ("basis",) in c.v2
    assert base + ("base point fraction",) in c.v2
    assert c.decisions == []
    [rewrite] = c.rewrites
    assert rewrite.new_def == "def deform(*, location):"
    assert not c.unclaimed()


def test_z10_idw():
    c = run(load("idw", {"TR_1 > func": "deform"}))
    base = FZ + ("TR_1",)
    assert c.v2[base + ("type",)] == Value("idw transform")
    for key in (
        "support radius",
        "power",
        "fixed zones",
        "n nearest",
        "deformation distance",
        "blending stiffness",
    ):
        assert base + (key,) in c.v2
    assert not c.unclaimed()
    stripped = {k: v for k, v in c.v2.items() if k != base + ("func",)}
    c.v2.clear()
    c.v2.update(stripped)
    c.v2[base + ("func",)] = Value(lambda *, location: location)
    zone = validate_zone(c, "TR_1")
    assert zone.n_nearest == 50 and zone.support_radius == 2.0


def test_z10_open_question_4_greedy_rbf_offers_full_rbf():
    c = run(load("rbf_greedy", {"TR_1 > func": "deform"}))
    base = FZ + ("TR_1",)
    type_decisions = [d for d in c.decisions if d.v2_path == base + ("type",)]
    assert len(type_decisions) == 1
    decision = type_decisions[0]
    assert decision.options == ("rbf transform",)
    assert "base point fraction" in decision.details
    assert ("tr 1", "tol") in decision.v1_paths
    assert isinstance(c.v2[base + ("type",)], Placeholder)
    assert c.v2[base + ("base point fraction",)] == Value(1.0)
    assert base + ("support radius",) in c.v2
    assert not c.unclaimed()


def test_z10_tr_with_no_type_is_the_v1_default_greedy_rbf():
    c = run(
        build(
            {
                "TR_1": {
                    "zone": [4],
                    "alpha": 200.0,
                    "basis": "c2",
                    "tol": 0.01,
                }
            }
        )
    )
    base = FZ + ("TR_1",)
    assert [d.v2_path for d in c.decisions] == [base + ("type",)]
    assert c.decisions[0].options == ("rbf transform",)
    assert ("tr 1", "tol") in c.decisions[0].v1_paths
    assert base + ("support radius",) in c.v2
    assert base + ("zones",) in c.v2
    assert not c.unclaimed()


def test_z11_scale_moves():
    c = run(load("placement"))
    assert c.v2[TF + ("scale",)] == Source(c.v1.leaf("scale"))


def test_z12_mesh_transform_matrix_moves_numpy_array():
    c = run(load("placement", {"mesh transform matrix": "matrix()"}))
    value = c.v2[TF + ("transform",)]
    assert value == Source(c.v1.leaf("mesh transform matrix"))
    assert isinstance(value.leaf.value, np.ndarray)
    assert not c.unclaimed()
    model = validate_transforms(c)
    assert model.scale == [0.001, 0.001, 0.001]
    assert np.array(model.transform).shape == (4, 4)


def v1_placed(scale, matrix, x):
    return (np.diag([*scale, 1.0]) @ np.asarray(matrix, dtype=float) @ [*x, 1.0])[:3]


def v2_placed(transforms, x):
    scaled = np.asarray(transforms.scale) * x
    return (np.asarray(transforms.transform, dtype=float) @ [*scaled, 1.0])[:3]


SCALE_ORDER_NOTE = (
    "v2 applies 'transforms > scale' before 'transforms > transform', and v1 "
    "applied the matrix first, so the matrix is rewritten as "
    "scale * matrix * scale^-1 to keep the mesh where v1 placed it."
)


@pytest.mark.parametrize(
    "scale, matrix",
    [
        (
            [0.0254, 0.0254, 0.0254],
            [[1, 0, 0, 2.0], [0, 0, -1, 0.5], [0, 1, 0, -3.0], [0, 0, 0, 1]],
        ),
        (
            [1.0, 2.0, 3.0],
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        ),
    ],
)
def test_z11_z12_a_scaled_matrix_is_rewritten_to_keep_the_v1_placement(scale, matrix):
    c = run(build({"scale": scale, "mesh transform matrix": np.array(matrix)}))
    assert isinstance(c.v2[TF + ("transform",)], Value)
    assert SCALE_ORDER_NOTE in [n.text for n in c.notes]
    assert not c.unclaimed()
    model = validate_transforms(c)
    for x in ([1.0, 2.0, 3.0], [-4.0, 0.5, 7.0]):
        np.testing.assert_allclose(
            v2_placed(model, x), v1_placed(scale, matrix, x), atol=1e-12
        )


def test_z11_z12_a_matrix_that_commutes_with_the_scale_keeps_its_source():
    matrix = np.diag([2.0, 2.0, 2.0, 1.0])
    c = run(build({"scale": [0.5, 0.5, 0.5], "mesh transform matrix": matrix}))
    assert c.v2[TF + ("transform",)] == Source(c.v1.leaf("mesh transform matrix"))
    assert SCALE_ORDER_NOTE not in [n.text for n in c.notes]


def test_z13_a_mesh_transform_function_v2_cannot_call_is_a_decision():
    c = run(load("mesh_transform_function", {"mesh transform function": "place"}))
    [decision] = c.decisions
    assert decision.v2_path == TF + ("transform",)
    assert decision.question == (
        "rewrite place as def place(*, location), returning the placed position "
        "as three numbers"
    )
    assert decision.details.startswith(
        'v1 called it as func(**{"coord": (x, y, z)}) and read back '
        '{"coord": (x, y, z)}. place '
    )
    assert [n.text for n in c.notes] == [V1_PLACEMENT_NOTE]
    assert not c.unclaimed()


V1_PLACEMENT_NOTE = (
    "v1 never placed the mesh with 'mesh transform function': it looked the "
    "function up under an empty key, only to pick partition cells, and before "
    "'scale'. v2 applies 'transforms > transform' to every node, after 'scale'."
)


def test_z13_a_mesh_transform_function_v2_can_call_moves_to_transform():
    def place(*, location):
        x, y, z = location
        return (x + 2.0, y, z)

    c = run(build({"scale": [2.0, 2.0, 2.0], "mesh transform function": place}))
    assert c.v2[TF + ("transform",)] == Source(c.v1.leaf("mesh transform function"))
    assert not c.decisions
    assert V1_PLACEMENT_NOTE in [n.text for n in c.notes]
    assert validate_transforms(c).transform is place


def test_z13_a_mesh_transform_function_beside_a_matrix_is_a_decision():
    def place(*, location):
        return location

    c = run(
        build({"mesh transform matrix": np.eye(4), "mesh transform function": place})
    )
    [decision] = c.decisions
    assert decision.v2_path == TF + ("transform",)
    assert set(decision.v1_paths) == {
        ("mesh transform matrix",),
        ("mesh transform function",),
    }
    assert decision.question == (
        "v1 sets both 'mesh transform matrix' and 'mesh transform function'. "
        "v2 places mesh m with one 'transform'. Which places it?"
    )
    assert decision.options == ("the matrix", "place")
    assert not c.unclaimed()


def test_z3_z4_a_v1_function_is_a_decision_to_port_it():
    def leaf_area(cells):
        return [(0.1,) for _ in cells]

    def drive(cell_centres):
        return [(1.0, 0.0, 0.0) for _ in cell_centres]

    c = run(
        build(
            {
                "FZ_1": {"type": "canopy", "def": "c.vtp", "func": leaf_area},
                "FZ_2": {"type": "momentum", "def": "m.vtp", "func": drive},
            }
        )
    )
    canopy, momentum = c.decisions
    assert canopy.v2_path == FZ + ("FZ_1", "function")
    assert canopy.question == (
        "rewrite leaf_area as def leaf_area(*, locations, wall_distances), "
        "returning a list of one leaf area density per location"
    )
    assert "one (x, y, z, wall distance) tuple per cell" in canopy.details
    assert momentum.v2_path == FZ + ("FZ_2", "func")
    assert "def drive(*, locations)" in momentum.question
    assert plain(c.v2[canopy.v2_path]).startswith("DECIDE: rewrite leaf_area as ")
    assert not c.unclaimed()


def test_z14_update_frequency_on_rotating_zone_is_unclaimed():
    c = run(load("update_frequency"))
    assert unclaimed_paths(c) == {("fz 1", "update frequency")}


def test_unknown_zone_type_leaves_everything_unclaimed():
    c = run(build({"FZ_1": {"type": "mystery", "zone": [1]}}))
    assert not c.v2 and not c.decisions
    assert len(c.unclaimed()) == 2


def test_deck_without_zones_writes_nothing():
    c = run(build({"equations": "euler"}))
    assert not c.v2 and not c.decisions


def test_user_labels_are_kept_exactly():
    c = run(build({"FZ_Hub": {"type": "porous", "def": "hub.vtp"}}))
    assert FZ + ("FZ_Hub", "definition") in c.v2


SWEEP_TYPES = [
    "momentum",
    "porous",
    "affine transform",
    "disc",
    "canopy",
]
SWEEP_EXTRAS = [
    "inertia",
    "rotor moment of inertia",
    "mean blade material density",
    "auto yaw",
    "rated power",
    "tip speed limit",
    "old yaw",
    "old omega",
    "dt",
    "damage ti",
    "damage speed",
    "friction loss",
    "cut in speed",
    "cut out speed",
    "rpm ramp",
    "thrust factor",
    "power coefficient",
    "tilt",
    "yaw",
    "blade pitch tol",
    "blade pitch range",
    "blade pitch step",
]


@pytest.mark.parametrize("kind", SWEEP_TYPES)
def test_sweep_zone_types_are_all_handled(kind):
    c = run(build({"FZ_1": {"type": kind, "zone": [1]}}))
    assert FZ + ("FZ_1", "type") in c.v2


@pytest.mark.parametrize("extra", SWEEP_EXTRAS)
def test_sweep_turbine_extras_are_never_claimed(extra):
    d = {
        "FZ_1": {
            "type": "disc",
            "zone": [1],
            extra: 1.0,
            "controller": {"type": "fixed", extra: 1.0},
        }
    }
    c = run(build(d))
    assert unclaimed_paths(c) == {("fz 1", extra), ("fz 1", "controller", extra)}


def test_every_leaf_is_claimed_at_most_once_across_toy_decks():
    for path in sorted(DECKS.glob("*.py")):
        c = run(load(path.stem))
        claimed = [d.v1_path for d in c.dispositions]
        assert len(claimed) == len(set(claimed)), path.name
