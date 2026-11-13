"""The boundaries rule: v1 ``BC_N``, ``modal model``, ``fsi``, ``sponge layer damping``
and ``mapping`` blocks to v2 boundary conditions.

Each test builds a toy v1 deck as a dict, flattens it to leaves and runs the
rule alone. Where a test ends with no DECIDE on a boundary condition, the v2
fragment is validated against the real pydantic boundary-condition schema.
Pattern ids (B1 to B20) are those of the "Deck patterns" section of
``developer_docs/v1-to-v2-conversion.md``.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import TypeAdapter

from zutil.update_input.model import (
    Conversion,
    ConversionError,
    Expr,
    MeshInfo,
    Placeholder,
    Source,
    V1Deck,
    Value,
    norm_path,
)
from zutil.update_input.rules import boundaries
from zutil.update_input.tests.engine.fakes import flatten
from zutil.validate.schemas.boundary_conditions_schemas import BoundaryCondition
from zutil.validate.validation_utils import replace_spaces_in_keys

LABEL = "toy"
BCS = ("model", LABEL, "boundary conditions")
UNCONVERTED = "unconverted v1 keys"

IC_1 = {
    "temperature": 288.15,
    "pressure": 101325.0,
    "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
    "Reynolds No": 6.0e6,
}


def run(tree: dict, mesh: MeshInfo | None = None, sources: dict | None = None):
    evaluated = copy.deepcopy(tree)
    c = Conversion(V1Deck(flatten(evaluated, sources), evaluated), LABEL, mesh)
    boundaries.convert(c)
    return c


def deck(**bcs) -> dict:
    tree = {"equations": "RANS", "reference": "IC_1", "IC_1": copy.deepcopy(IC_1)}
    tree.update(bcs)
    return tree


def resolve(value):
    if isinstance(value, Source):
        return value.leaf.value
    if isinstance(value, (Value, Expr)):
        return value.value
    if isinstance(value, Placeholder):
        return value.text
    raise TypeError(value)


def subtree(c: Conversion, prefix: tuple) -> dict:
    root: dict = {}
    for path, value in c.v2.items():
        if path[: len(prefix)] != prefix:
            continue
        node = root
        for key in path[len(prefix) : -1]:
            node = node.setdefault(key, {})
        node[path[-1]] = resolve(value)
    return root


def bc(c: Conversion, label: str) -> dict:
    return subtree(c, BCS + (label,))


def validate(fragment: dict):
    return TypeAdapter(BoundaryCondition).validate_python(
        replace_spaces_in_keys(copy.deepcopy(fragment))
    )


def disposition(c: Conversion, *keys):
    path = norm_path(*keys)
    matches = [d for d in c.dispositions if d.v1_path == path]
    assert len(matches) == 1, f"{path}: {matches}"
    return matches[0]


def decision_at(c: Conversion, path: tuple):
    matches = [d for d in c.decisions if d.v2_path == path]
    assert len(matches) == 1, f"{path}: {c.decisions}"
    return matches[0]


def owned_unclaimed(c: Conversion) -> list[tuple]:
    import fnmatch

    return [
        lf.path
        for lf in c.unclaimed()
        if any(fnmatch.fnmatchcase(lf.path[0], p) for p in boundaries.OWNS)
    ]


def mesh(**zones_by_type) -> MeshInfo:
    return MeshInfo("toy.h5", "toy", {int(k[1:]): v for k, v in zones_by_type.items()})


def test_module_contract():
    assert boundaries.OWNS == [
        "bc *",
        "modal model",
        "fsi",
        "sponge layer damping",
        "mapping",
        "create overset halos",
    ]
    assert boundaries.ORDER == 40


@pytest.mark.parametrize(
    "v1, v2",
    [
        ({"zone": [1], "type": "symmetry"}, {"zones": [1], "type": "symmetry"}),
        (
            {"zone": [1], "type": "wall", "kind": "slip"},
            {"zones": [1], "type": "wall", "kind": "slip"},
        ),
        (
            {"zone": [1], "type": "farfield", "kind": "riemann", "condition": "IC_1"},
            {"zones": [1], "type": "farfield", "kind": "riemann", "condition": "IC_1"},
        ),
        (
            {"zone": [1], "type": "inflow", "kind": "default", "condition": "IC_1"},
            {"zones": [1], "type": "inflow", "kind": "pressure", "reference": "IC_1"},
        ),
        (
            {"zone": [1], "type": "outflow", "kind": "pressure", "condition": "IC_1"},
            {"zones": [1], "type": "outflow", "kind": "pressure", "reference": "IC_1"},
        ),
        (
            {"zone": [1], "type": "overset"},
            {"zones": [1], "type": "interface", "kind": "overset"},
        ),
    ],
)
def test_b1_every_type_with_zone(v1, v2):
    c = run(deck(BC_1=v1))
    assert bc(c, "BC_1") == v2
    assert c.decisions == []
    assert owned_unclaimed(c) == []
    validate(v2)
    assert disposition(c, "BC_1", "zone").v2_path == BCS + ("BC_1", "zones")


def test_b1_zone_moves_its_source_text():
    c = run(
        deck(BC_1={"zone": [14, 15], "type": "symmetry"}),
        sources={("BC_1", "zone"): "sym"},
    )
    value = c.v2[BCS + ("BC_1", "zones")]
    assert isinstance(value, Source) and value.leaf.source == "sym"


def test_b2_ref_only_with_mesh_reads_zones_for_each_code():
    c = run(
        deck(
            BC_1={"ref": 7, "type": "symmetry"},
            BC_2={"ref": 3, "type": "wall", "kind": "slip"},
            BC_3={"ref": 9, "type": "farfield", "condition": "IC_1", "kind": "riemann"},
            BC_4={"ref": 4, "type": "inflow", "kind": "default", "condition": "IC_1"},
            BC_5={"ref": 5, "type": "outflow", "kind": "pressure", "condition": "IC_1"},
        ),
        mesh=mesh(c3=[4, 5], c4=[1], c5=[2], c7=[14], c9=[12, 13]),
    )
    expected = {
        "BC_1": [14],
        "BC_2": [4, 5],
        "BC_3": [12, 13],
        "BC_4": [1],
        "BC_5": [2],
    }
    for label, zones in expected.items():
        assert bc(c, label)["zones"] == zones
        d = disposition(c, label, "ref")
        assert d.kind == "respelled"
        assert d.v2_path == BCS + (label, "zones")
        validate(bc(c, label))
    assert c.decisions == []


def test_b2_ref_only_with_mesh_read_through_mesh_reader(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    from zutil.update_input.mesh import read_mesh

    path = tmp_path / "toy.h5"
    face_bc = [0, 2, 3, 3, 7, 9]
    zones = [0, 0, 5, 6, 14, 12]
    info = np.stack([np.asarray(zones), np.zeros(len(zones), dtype=int)], axis=1)
    with h5py.File(path, "w") as f:
        g = f.create_group("mesh")
        g.create_dataset("faceBC", data=np.asarray(face_bc, dtype="i4"))
        g.create_dataset("faceInfo", data=info)
    c = run(deck(BC_2={"ref": 3, "type": "wall", "kind": "slip"}), mesh=read_mesh(path))
    assert bc(c, "BC_2") == {"zones": [5, 6], "type": "wall", "kind": "slip"}


def test_b2_ref_only_without_mesh_is_decide():
    c = run(deck(BC_2={"ref": 3, "type": "wall", "kind": "slip"}))
    d = decision_at(c, BCS + ("BC_2", "zones"))
    assert "--mesh" in d.details
    assert disposition(c, "BC_2", "ref").kind == "decide"
    assert isinstance(c.v2[BCS + ("BC_2", "zones")], Placeholder)


def test_b2_ref_only_overset_is_decide_even_with_mesh():
    c = run(deck(BC_1={"ref": 10, "type": "overset"}), mesh=mesh(c10=[3], c20=[3]))
    d = decision_at(c, BCS + ("BC_1", "zones"))
    assert "0x14" in d.details
    assert bc(c, "BC_1")["type"] == "interface"


def test_b2_ref_only_code_absent_from_mesh_is_decide():
    c = run(deck(BC_2={"ref": 3, "type": "wall"}), mesh=mesh(c9=[12]))
    d = decision_at(c, BCS + ("BC_2", "zones"))
    assert "3" in d.question or "3" in d.details


def test_b2_open_question_8_ref_and_zone_with_mesh_covered_drops_ref():
    c = run(
        deck(BC_2={"ref": 3, "zone": [4, 5], "type": "wall"}),
        mesh=mesh(c3=[4, 5]),
    )
    assert bc(c, "BC_2")["zones"] == [4, 5]
    assert disposition(c, "BC_2", "ref").kind == "dropped"
    assert c.decisions == []


def test_b2_open_question_8_ref_and_zone_with_mesh_not_covered_is_decide():
    c = run(
        deck(BC_2={"ref": 3, "zone": [4], "type": "wall"}),
        mesh=mesh(c3=[4, 5]),
    )
    d = decision_at(c, BCS + ("BC_2", "zones"))
    assert d.options == ("[4]", "[4, 5]")
    assert disposition(c, "BC_2", "ref").kind == "decide"
    assert disposition(c, "BC_2", "zone").kind == "decide"


def test_b2_open_question_8_ref_and_zone_without_mesh_drops_ref_with_note():
    c = run(deck(BC_2={"ref": 3, "zone": [4], "type": "wall"}))
    assert bc(c, "BC_2")["zones"] == [4]
    assert disposition(c, "BC_2", "ref").kind == "dropped"
    assert any("BC_2" in n.text and "3" in n.text for n in c.notes)
    assert c.decisions == []


@pytest.mark.parametrize(
    "v1, v2",
    [("noslip", "no slip"), ("wallfunction", "wall function"), ("slip", "slip")],
)
def test_b3_wall_kinds(v1, v2):
    c = run(deck(BC_3={"zone": [5], "type": "wall", "kind": v1}))
    assert bc(c, "BC_3")["kind"] == v2
    assert disposition(c, "BC_3", "kind").kind == ("moved" if v1 == v2 else "respelled")
    validate(bc(c, "BC_3"))


def test_b3_compressible_default_inflow_becomes_pressure():
    c = run(
        deck(
            BC_4={"zone": [2], "type": "inflow", "kind": "default", "condition": "IC_1"}
        )
    )
    assert bc(c, "BC_4")["kind"] == "pressure"
    assert disposition(c, "BC_4", "kind").kind == "respelled"


def test_b3_incompressible_default_inflow_stays_default():
    tree = deck(
        BC_3={"zone": [1], "type": "inflow", "kind": "default", "condition": "IC_1"}
    )
    tree["equations"] = "incomp viscous"
    c = run(tree)
    assert bc(c, "BC_3")["kind"] == "default"
    assert disposition(c, "BC_3", "kind").kind == "moved"
    validate(bc(c, "BC_3"))


def test_b4_farfield_condition_kept():
    c = run(
        deck(
            BC_5={
                "zone": [3],
                "type": "farfield",
                "condition": "IC_1",
                "kind": "riemann",
            }
        )
    )
    assert bc(c, "BC_5")["condition"] == "IC_1"


def test_b4_plain_ic_condition_becomes_reference():
    c = run(
        deck(
            BC_3={
                "zone": [1],
                "type": "inflow",
                "kind": "massflow",
                "condition": "IC_1",
            },
            BC_4={
                "zone": [4],
                "type": "outflow",
                "kind": "pressure",
                "condition": "IC_1",
            },
        )
    )
    for label in ("BC_3", "BC_4"):
        assert bc(c, label)["reference"] == "IC_1"
        assert "condition" not in bc(c, label)
        assert disposition(c, label, "condition").kind == "moved"


def test_b4_derived_ic_condition_becomes_its_reference():
    tree = deck(
        BC_4={"zone": [2], "type": "inflow", "condition": "IC_2", "kind": "default"}
    )
    tree["IC_2"] = {
        "reference": "IC_1",
        "total pressure ratio": 1.02828,
        "total temperature ratio": 1.008,
    }
    c = run(tree)
    assert bc(c, "BC_4") == {
        "zones": [2],
        "type": "inflow",
        "kind": "pressure",
        "reference": "IC_1",
    }
    assert disposition(c, "BC_4", "condition").kind == "respelled"
    for ratio in ("total pressure ratio", "total temperature ratio"):
        assert BCS + ("BC_4", ratio) not in c.v2


def test_b4_lower_case_ic_label_is_found():
    tree = deck(
        bc_3={"zone": [3], "type": "outflow", "kind": "pressure", "condition": "IC_2"}
    )
    tree["ic_2"] = {"reference": "IC_1", "static pressure ratio": 1.0}
    c = run(tree)
    assert bc(c, "bc_3")["reference"] == "IC_1"


def test_b4_ratio_paths_are_left_for_conditions_rule():
    tree = deck(
        BC_4={"zone": [2], "type": "inflow", "condition": "IC_2", "kind": "default"}
    )
    tree["IC_2"] = {"reference": "IC_1", "total pressure ratio": 1.2}
    c = run(tree)
    c.set(BCS + ("BC_4", "total pressure ratio"), Value(1.2))
    validate(bc(c, "BC_4"))
    with pytest.raises(ConversionError):
        c.set(BCS + ("BC_4", "reference"), Value("IC_1"))


def test_b5_wall_linear_velocity():
    c = run(
        deck(
            BC_1={
                "zone": [1],
                "type": "wall",
                "v": {"linear": {"vector": [1, 0, 0], "mach": 0.1}},
            }
        )
    )
    assert bc(c, "BC_1")["velocity"] == {
        "type": "linear",
        "vector": [1, 0, 0],
        "mach": 0.1,
    }
    validate(bc(c, "BC_1"))


def test_b5_wall_rotating_velocity_keeps_origin():
    c = run(
        deck(
            BC_1={
                "zone": [1],
                "type": "wall",
                "v": {
                    "rotating": {"axis": [0, 0, 1], "origin": [1, 2, 3], "omega": 5.0}
                },
            }
        )
    )
    assert bc(c, "BC_1")["velocity"] == {
        "type": "rotating",
        "axis": [0, 0, 1],
        "origin": [1, 2, 3],
        "omega": 5.0,
    }
    assert disposition(c, "BC_1", "v", "rotating", "origin").v2_path == BCS + (
        "BC_1",
        "velocity",
        "origin",
    )
    validate(bc(c, "BC_1"))


def test_b6_roughness_moves_unchanged():
    c = run(
        deck(
            BC_1={
                "zone": [1],
                "type": "wall",
                "kind": "wallfunction",
                "roughness": {"type": "height", "scalar": 1e-4},
            }
        )
    )
    assert bc(c, "BC_1")["roughness"] == {"type": "height", "scalar": 1e-4}
    validate(bc(c, "BC_1"))


def test_b7_wall_temperature_scalar_respelled():
    c = run(deck(BC_1={"zone": [1], "type": "wall", "temperature": {"scalar": 300.0}}))
    assert bc(c, "BC_1")["temperature"] == 300.0
    validate(bc(c, "BC_1"))


def test_b7_wall_temperature_field_moved():
    c = run(deck(BC_1={"zone": [1], "type": "wall", "temperature": {"field": "t.vtp"}}))
    assert bc(c, "BC_1")["temperature"] == {"field": "t.vtp"}
    validate(bc(c, "BC_1"))


def test_b8_immersed_wall_is_decide_for_wall():
    c = run(
        deck(
            BC_1={
                "zone": [1],
                "type": "immersed wall",
                "kind": "noslip",
                "immersed normal type": "immersed boundary vector",
            }
        )
    )
    assert decision_at(c, BCS + ("BC_1", "type")).options == ("wall",)
    assert bc(c, "BC_1")["kind"] == "no slip"
    assert "immersed wall normal type" not in bc(c, "BC_1")
    assert disposition(c, "BC_1", "immersed normal type").kind == "dropped"


def test_b9_radial_pressure_gradient_reference_radius():
    c = run(
        deck(
            BC_4={
                "zone": [4],
                "type": "outflow",
                "kind": "radial pressure gradient",
                "reference radius": 0.5,
                "condition": "IC_1",
            }
        )
    )
    assert bc(c, "BC_4")["reference radius"] == 0.5
    validate(bc(c, "BC_4"))


def test_b10_overset_becomes_interface():
    c = run(deck(BC_9={"zone": [9], "type": "overset"}))
    assert bc(c, "BC_9") == {"zones": [9], "type": "interface", "kind": "overset"}
    assert disposition(c, "BC_9", "type").kind == "respelled"
    validate(bc(c, "BC_9"))


def test_b11_periodic_linear_pair():
    c = run(
        deck(
            BC_4={
                "zone": [18],
                "type": "periodic",
                "kind": {"linear": {"vector": [0, 0, 0.24]}},
            },
            BC_5={
                "zone": [17],
                "type": "periodic",
                "kind": {"linear": {"vector": [0, 0, -0.24]}},
            },
        )
    )
    assert bc(c, "BC_4") == {
        "zones": [18],
        "type": "interface",
        "kind": "periodic",
        "connection": {"boundary": "BC_5"},
        "transform": {"type": "linear", "vector": [0, 0, 0.24]},
    }
    assert bc(c, "BC_5")["connection"] == {"boundary": "BC_4"}
    assert c.decisions == []
    validate(bc(c, "BC_4"))
    validate(bc(c, "BC_5"))


def test_b11_periodic_rotated_pair():
    def side(theta):
        return {
            "zone": [1],
            "type": "periodic",
            "kind": {
                "rotated": {"theta": theta, "axis": [1, 0, 0], "origin": [0, 0, 0]}
            },
        }

    c = run(deck(BC_1=side(0.5), BC_2=side(-0.5)))
    assert bc(c, "BC_1")["transform"] == {
        "type": "rotated",
        "theta": 0.5,
        "axis": [1, 0, 0],
        "origin": [0, 0, 0],
    }
    assert bc(c, "BC_2")["connection"] == {"boundary": "BC_1"}
    validate(bc(c, "BC_1"))


def test_b11_lone_periodic_is_decide():
    c = run(
        deck(
            BC_4={
                "zone": [18],
                "type": "periodic",
                "kind": {"linear": {"vector": [0, 0, 1]}},
            }
        )
    )
    decision_at(c, BCS + ("BC_4", "connection", "boundary"))
    assert bc(c, "BC_4")["transform"] == {"type": "linear", "vector": [0, 0, 1]}


def test_b11_three_periodic_is_decide():
    lin = {"zone": [1], "type": "periodic", "kind": {"linear": {"vector": [0, 0, 1]}}}
    c = run(deck(BC_1=lin, BC_2=lin, BC_3=lin))
    d = decision_at(c, BCS + ("BC_1", "connection", "boundary"))
    assert d.options == ("BC_2", "BC_3")
    decision_at(c, BCS + ("BC_2", "connection", "boundary"))
    decision_at(c, BCS + ("BC_3", "connection", "boundary"))


def test_b11_pair_that_is_not_inverse_is_decide():
    lin = {"zone": [1], "type": "periodic", "kind": {"linear": {"vector": [0, 0, 1]}}}
    c = run(deck(BC_1=lin, BC_2=lin))
    d = decision_at(c, BCS + ("BC_1", "connection", "boundary"))
    assert d.options == ("BC_2",)


SPONGE = {"distance": 2.0, "damping factor": 0.5, "condition": "IC_1"}
FARFIELD = {"type": "farfield", "kind": "riemann", "condition": "IC_1"}


def test_b12_open_question_7_sponge_on_one_farfield():
    tree = deck(BC_2=dict(FARFIELD, zone=[2]))
    tree["sponge layer damping"] = dict(SPONGE)
    c = run(tree)
    assert bc(c, "BC_2")["sponge layer damping"] == SPONGE
    validate(bc(c, "BC_2"))
    assert owned_unclaimed(c) == []


def test_b12_open_question_7_sponge_copied_to_every_farfield():
    tree = deck(BC_2=dict(FARFIELD, zone=[2]), BC_3=dict(FARFIELD, zone=[3]))
    tree["sponge layer damping"] = dict(SPONGE)
    c = run(tree)
    assert bc(c, "BC_2")["sponge layer damping"] == SPONGE
    assert bc(c, "BC_3")["sponge layer damping"] == SPONGE
    assert any("BC_2" in n.text and "BC_3" in n.text for n in c.notes)
    assert disposition(c, "sponge layer damping", "distance").kind == "moved"


def test_b12_open_question_7_sponge_with_no_farfield_is_decide():
    tree = deck(BC_1={"zone": [1], "type": "wall"})
    tree["sponge layer damping"] = dict(SPONGE)
    c = run(tree)
    d = decision_at(c, (UNCONVERTED, "sponge layer damping"))
    assert len(d.v1_paths) == 3
    assert resolve(c.v2[(UNCONVERTED, "sponge layer damping")]) == SPONGE
    assert owned_unclaimed(c) == []


MM = {
    "zone": [1, 2, 3],
    "file format": "nastran",
    "transform type": "rbf multiscale",
    "alpha": 2.0,
    "fluid force scaling": 1.0,
    "basis": "c2",
    "base point fraction": 0.1,
    "nastran casename": "wing",
    "mode mapping max distance": 0.1,
    "modal damping": [0.0, 0.0],
    "mode list": [1, 2],
    "mode mapping pureidw": False,
    "use n nearest neighbours": 8,
    "newmark alpha": 0.25,
}


def test_b13_modal_model_on_matching_wall():
    tree = deck(
        BC_1={"zone": [3, 2, 1], "type": "wall"},
        BC_2={"zone": [1, 2], "type": "wall"},
    )
    tree["modal model"] = {"MM_1": dict(MM)}
    c = run(tree)
    fsi = bc(c, "BC_1")["fsi"]
    assert fsi["transform type"] == "rbf_multiscale"
    assert fsi["support radius"] == 2.0
    assert fsi["mode mapping pure idw"] is False
    assert fsi["mode mapping n nearest"] == 8
    assert "zone" not in fsi
    assert "fsi" not in bc(c, "BC_2")
    assert disposition(c, "modal model", "MM_1", "zone").kind == "dropped"
    assert validate(bc(c, "BC_1")).fsi.type == "modal_model"
    assert owned_unclaimed(c) == []


def test_b13_modal_model_update_frequency_moves_to_the_fsi_block():
    tree = deck(BC_1={"zone": [1, 2, 3], "type": "wall"})
    tree["modal model"] = {"MM_1": dict(MM, **{"update frequency": 5})}
    c = run(tree)
    assert bc(c, "BC_1")["fsi"]["update frequency"] == 5
    assert validate(bc(c, "BC_1")).fsi.update_frequency == 5
    assert owned_unclaimed(c) == []


def test_b13_modal_model_v1_defaults_v2_requires_are_written():
    mm = {
        k: v
        for k, v in MM.items()
        if k
        not in (
            "file format",
            "transform type",
            "fluid force scaling",
            "base point fraction",
        )
    }
    tree = deck(BC_1={"zone": [1, 2, 3], "type": "wall"})
    tree["modal model"] = {"MM_1": mm}
    c = run(tree)
    fsi = bc(c, "BC_1")["fsi"]
    assert fsi["file format"] == "nastran"
    assert fsi["transform type"] == "rbf_multiscale"
    assert fsi["fluid force scaling"] == 1.0
    assert fsi["base point fraction"] == 0.1
    validate(bc(c, "BC_1"))


def test_b13_modal_model_matching_no_wall_is_decide():
    tree = deck(BC_1={"zone": [1, 2], "type": "wall"})
    tree["modal model"] = {"MM_1": dict(MM)}
    c = run(tree)
    d = decision_at(c, (UNCONVERTED, "modal model > MM_1"))
    assert "[1, 2, 3]" in d.details
    assert owned_unclaimed(c) == []
    assert "fsi" not in bc(c, "BC_1")


def test_b13_modal_model_on_part_of_a_wall_asks_to_split_the_wall():
    tree = deck(BC_1={"zone": [1, 2, 3, 4, 5], "type": "wall"})
    tree["modal model"] = {"MM_1": dict(MM)}
    c = run(tree)
    d = decision_at(c, (UNCONVERTED, "modal model > MM_1"))
    assert d.question.startswith("Split BC_1 into two walls")
    assert "[1, 2, 3]" in d.question and "[4, 5]" in d.question
    assert owned_unclaimed(c) == []
    assert "fsi" not in bc(c, "BC_1")


def test_b13_modal_greedy_rbf_is_decide():
    tree = deck(BC_1={"zone": [1, 2, 3], "type": "wall"})
    tree["modal model"] = {"MM_1": dict(MM, **{"transform type": "rbf"})}
    c = run(tree)
    d = decision_at(c, BCS + ("BC_1", "fsi", "transform type"))
    assert d.options == ("rbf_multiscale",)
    assert "1.0" in d.details


def test_b13_modal_idw_moves_idw_keys():
    mm = {k: v for k, v in MM.items() if k not in ("basis", "base point fraction")}
    mm.update(
        {
            "transform type": "idw",
            "power": [3.0, 5.0, 10.0],
            "n nearest": 50,
            "fixed zones": [7],
        }
    )
    tree = deck(BC_1={"zone": [1, 2, 3], "type": "wall"})
    tree["modal model"] = {"MM_1": mm}
    c = run(tree)
    fsi = bc(c, "BC_1")["fsi"]
    assert fsi["transform type"] == "idw"
    assert fsi["fixed zones"] == [7]
    validate(bc(c, "BC_1"))


FSI = {
    "type": "generic",
    "zone": [5],
    "transform type": "rbf multiscale",
    "alpha": 3.0,
    "max error": 0.2,
    "user variables": {"Coupler": "preCICE", "steps": 2},
}


def test_b14_top_level_fsi_on_matching_wall():
    tree = deck(BC_3={"zone": [5], "type": "wall"})
    tree["fsi"] = copy.deepcopy(FSI)
    c = run(tree)
    fsi = bc(c, "BC_3")["fsi"]
    assert fsi == {
        "type": "generic_fsi",
        "zone": [5],
        "transform type": "rbf_multiscale",
        "support radius": 3.0,
        "max error": 0.2,
        "user variables": {"Coupler": "preCICE", "steps": 2},
    }
    assert validate(bc(c, "BC_3")).fsi.type == "generic_fsi"
    assert owned_unclaimed(c) == []


def test_b14_fsi_type_without_generic_is_decide():
    tree = deck(BC_3={"zone": [5], "type": "wall"})
    tree["fsi"] = dict(copy.deepcopy(FSI), type="external")
    c = run(tree)
    d = decision_at(c, BCS + ("BC_3", "fsi", "type"))
    assert d.options == ("generic_fsi",)


def test_b14_fsi_matching_no_wall_is_decide():
    tree = deck(BC_3={"zone": [6], "type": "wall"})
    tree["fsi"] = copy.deepcopy(FSI)
    c = run(tree)
    decision_at(c, (UNCONVERTED, "fsi"))
    assert owned_unclaimed(c) == []


def test_b15_active_layers_copied_to_every_overset():
    tree = deck(
        BC_1={"zone": [1], "type": "overset"}, BC_2={"zone": [2], "type": "overset"}
    )
    tree["mapping"] = {"number of additional overset active layers": 2}
    c = run(tree)
    for label in ("BC_1", "BC_2"):
        assert bc(c, label)["additional active layers"] == 2
        validate(bc(c, label))
    assert c.decisions == []


def test_b15_active_layers_with_no_overset_is_decide():
    tree = deck(BC_1={"zone": [1], "type": "wall"})
    tree["mapping"] = {"number of additional overset active layers": 2}
    c = run(tree)
    decision_at(
        c, (UNCONVERTED, "mapping > number of additional overset active layers")
    )


def test_b15_active_layers_zero_dropped():
    tree = deck(BC_1={"zone": [1], "type": "overset"})
    tree["mapping"] = {"number of additional overset active layers": 0}
    c = run(tree)
    d = disposition(c, "mapping", "number of additional overset active layers")
    assert d.kind == "dropped"
    assert "additional active layers" not in bc(c, "BC_1")


def test_b15_overset_halos_copied_to_every_overset():
    tree = deck(
        BC_1={"zone": [1], "type": "overset"},
        BC_2={"zone": [2], "type": "overset"},
        BC_3={"zone": [3], "type": "wall"},
    )
    tree["create overset halos"] = False
    c = run(tree)
    for label in ("BC_1", "BC_2"):
        assert bc(c, label)["create overset halos"] is False
        validate(bc(c, label))
    assert "create overset halos" not in bc(c, "BC_3")
    assert not any(p[0] == "solver" for p in c.v2)
    assert disposition(c, "create overset halos").kind == "moved"


def test_b15_absent_overset_halos_on_an_overset_deck_gets_a_note():
    c = run(deck(BC_1={"zone": [1], "type": "Overset"}, BC_2={"type": "wall"}))
    assert "create overset halos" not in bc(c, "BC_1")
    [note] = [n for n in c.notes if "create overset halos" in n.text]
    assert "False" in note.text and "True" in note.text


def test_b15_absent_overset_halos_without_overset_has_no_note():
    c = run(deck(BC_1={"zone": [1], "type": "wall"}))
    assert not [n for n in c.notes if "create overset halos" in n.text]


def test_b15_overset_halos_with_no_overset_is_dropped():
    c = run(
        dict(deck(BC_1={"zone": [1], "type": "wall"}), **{"create overset halos": True})
    )
    assert disposition(c, "create overset halos").kind == "dropped"
    assert owned_unclaimed(c) == []


def test_b16_idw_from_cells_order():
    c = run(dict(deck(), mapping={"idw from cells order": 0}))
    assert disposition(c, "mapping", "idw from cells order").kind == "dropped"
    c = run(dict(deck(), mapping={"idw from cells order": 1}))
    assert owned_unclaimed(c) == [("mapping", "idw from cells order")]


def test_b17_wallsrc_is_decide():
    c = run(deck(BC_1={"zone": [1], "type": "wallsrc"}))
    decision_at(c, BCS + ("BC_1", "type"))
    assert bc(c, "BC_1")["zones"] == [1]


def test_b18_massflow_inflow_and_outflow():
    tree = deck(
        bc_1={"zone": [1], "type": "wall", "kind": "noslip"},
        bc_2={"zone": [2], "type": "inflow", "condition": "IC_3", "kind": "massflow"},
        bc_3={"zone": [3], "type": "outflow", "kind": "massflow", "condition": "IC_3"},
    )
    tree["ic_3"] = {"reference": "IC_1", "mass flow rate": 2.0}
    c = run(tree)
    assert bc(c, "bc_2") == {
        "zones": [2],
        "type": "inflow",
        "kind": "massflow",
        "reference": "IC_1",
    }
    assert bc(c, "bc_3")["kind"] == "massflow"
    assert bc(c, "bc_3")["reference"] == "IC_1"


def test_b19_labels_kept_as_spelt():
    c = run(deck(**{"Bc_Wing Wall": {"zone": [1], "type": "wall"}}))
    assert bc(c, "Bc_Wing Wall") == {"zones": [1], "type": "wall"}


def test_b20_numbering_gap_is_noted():
    c = run(
        deck(BC_1={"zone": [1], "type": "wall"}, BC_3={"zone": [3], "type": "symmetry"})
    )
    assert bc(c, "BC_3") == {"zones": [3], "type": "symmetry"}
    gap = [n for n in c.notes if "BC_2" in n.text]
    assert len(gap) == 1 and "BC_3" in gap[0].text


def test_b20_no_note_without_gap():
    c = run(
        deck(BC_1={"zone": [1], "type": "wall"}, BC_2={"zone": [2], "type": "wall"})
    )
    assert c.notes == []


def test_owned_keys_with_no_v2_home_are_left_unclaimed():
    mm = dict(
        MM,
        **{
            "use local rbf nodes": True,
            "relaxation factor": 0.5,
            "max error": 0.1,
        },
    )
    tree = deck(BC_1={"zone": [1, 2, 3], "type": "wall"})
    tree["modal model"] = {"MM_1": mm}
    c = run(tree)
    assert sorted(owned_unclaimed(c)) == sorted(
        norm_path("modal model", "MM_1", k)
        for k in (
            "use local rbf nodes",
            "relaxation factor",
            "max error",
        )
    )


def sweep_deck() -> dict:
    tree = deck(
        BC_1={"zone": [1], "type": "wallsrc"},
        BC_2={"zone": [2], "type": "immersed wall", "immersed normal type": "STL"},
        BC_3={
            "zone": [3],
            "type": "outflow",
            "kind": "radial pressure gradient",
            "reference radius": 1.0,
        },
        BC_4={
            "zone": [4],
            "type": "periodic",
            "kind": {"linear": {"vector": [1, 0, 0]}},
        },
        BC_5={
            "zone": [5],
            "type": "periodic",
            "kind": {"linear": {"vector": [-1, 0, 0]}},
        },
        BC_6={"zone": [6], "type": "wall"},
        BC_7={"zone": [7], "type": "wall"},
    )
    tree["modal model"] = {
        "MM_1": dict(
            MM,
            zone=[6],
            **{
                "transform type": "rbf",
                "fixed zones": [9],
                "forcing": {
                    "location": [0, 0, 0],
                    "force": [0, 0, 1],
                    "frequency": 2.0,
                },
                "mode mapping power": 2.0,
                "use local rbf nodes": True,
                "newmark delta": 0.5,
                "relaxation factor": 0.5,
                "max error": 0.1,
                "recalculate rbfs alpha fraction": 0.1,
            },
        )
    }
    return tree


@pytest.mark.parametrize(
    "keys, kind, v2_tail",
    [
        (("BC_1", "type"), "decide", ("BC_1", "type")),
        (("BC_2", "type"), "decide", ("BC_2", "type")),
        (("BC_3", "reference radius"), "moved", ("BC_3", "reference radius")),
        (("BC_4", "type"), "respelled", ("BC_4", "type")),
        (
            ("modal model", "MM_1", "transform type"),
            "decide",
            ("BC_6", "fsi", "transform type"),
        ),
        (("modal model", "MM_1", "fixed zones"), None, None),
        (
            ("modal model", "MM_1", "forcing", "location"),
            "moved",
            ("BC_6", "fsi", "harmonic initial state", "location"),
        ),
        (
            ("modal model", "MM_1", "forcing", "force"),
            "moved",
            ("BC_6", "fsi", "harmonic initial state", "force"),
        ),
        (
            ("modal model", "MM_1", "forcing", "frequency"),
            "moved",
            ("BC_6", "fsi", "harmonic initial state", "frequency"),
        ),
        (
            ("modal model", "MM_1", "mode mapping power"),
            "moved",
            ("BC_6", "fsi", "mode mapping power"),
        ),
        (
            ("modal model", "MM_1", "mode mapping pureidw"),
            "respelled",
            ("BC_6", "fsi", "mode mapping pure idw"),
        ),
        (
            ("modal model", "MM_1", "use n nearest neighbours"),
            "respelled",
            ("BC_6", "fsi", "mode mapping n nearest"),
        ),
        (("modal model", "MM_1", "use local rbf nodes"), None, None),
        (
            ("modal model", "MM_1", "newmark alpha"),
            "moved",
            ("BC_6", "fsi", "newmark alpha"),
        ),
        (
            ("modal model", "MM_1", "newmark delta"),
            "moved",
            ("BC_6", "fsi", "newmark delta"),
        ),
        (("modal model", "MM_1", "relaxation factor"), None, None),
        (("modal model", "MM_1", "max error"), None, None),
        (
            ("modal model", "MM_1", "recalculate rbfs alpha fraction"),
            "moved",
            ("BC_6", "fsi", "recalculate rbfs alpha fraction"),
        ),
    ],
)
def test_names_sweep(keys, kind, v2_tail):
    c = run(sweep_deck())
    if kind is None:
        assert norm_path(*keys) in owned_unclaimed(c)
        return
    d = disposition(c, *keys)
    assert d.kind == kind
    assert d.v2_path == BCS + v2_tail


def test_names_sweep_fsi_keys():
    tree = deck(BC_7={"zone": [7], "type": "wall"})
    tree["fsi"] = dict(copy.deepcopy(FSI), zone=[7], **{"fixed zones": [8]})
    c = run(tree)
    for keys, tail in [
        (("fsi", "type"), ("fsi", "type")),
        (("fsi", "fixed zones"), ("fsi", "fixed zones")),
        (("fsi", "max error"), ("fsi", "max error")),
        (("fsi", "user variables", "Coupler"), ("fsi", "user variables", "Coupler")),
    ]:
        assert disposition(c, *keys).v2_path == BCS + ("BC_7",) + tail
    validate(bc(c, "BC_7"))


def test_a_forcing_function_is_a_decision_on_the_force():
    def push(**kwargs):
        return {"force": (0, 0, 1)}

    tree = sweep_deck()
    tree["modal model"]["MM_1"]["forcing"]["force"] = push
    c = run(tree)
    fsi = BCS + ("BC_6", "fsi", "harmonic initial state")
    decision = decision_at(c, fsi + ("force",))
    assert "constant force vector" in decision.question
    assert "once, at time 0" in decision.details
    assert "'force'" in decision.details
    for key in ("location", "frequency"):
        d = disposition(c, "modal model", "MM_1", "forcing", key)
        assert (d.kind, d.v2_path) == ("moved", fsi + (key,))
    assert disposition(c, "modal model", "MM_1", "forcing", "force").kind == "decide"
