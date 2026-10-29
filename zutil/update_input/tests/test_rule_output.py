"""Rule package R6: output settings, reports and solver-wide settings."""

from __future__ import annotations

import copy

import pytest

from zutil.update_input.model import (
    Conversion,
    Leaf,
    Source,
    V1Deck,
    Value,
    Expr,
    norm_key,
)
from zutil.update_input.rules import output
from zutil.validate.schemas.output_schemas import OutputModel
from zutil.validate.schemas.solver_settings_schemas import (
    IncompressibleLinearSolverOptions,
    LinearSolverOptions,
    SolverSettingsPropertiesModel,
)
from zutil.validate.validation_utils import replace_spaces_in_keys

OUT = ("solver", "output settings")
SETTINGS = ("solver", "solver settings")


def flatten(tree, prefix=()):
    leaves = []
    for key, value in tree.items():
        keys = prefix + (key,)
        if isinstance(value, dict) and value:
            leaves.extend(flatten(value, keys))
        elif not isinstance(value, dict):
            leaves.append(Leaf(keys, value, repr(value)))
    return leaves


def convert(tree):
    tree = copy.deepcopy(tree)
    c = Conversion(V1Deck(flatten(tree), tree), "wing")
    output.convert(c)
    return c


def evaluate(value):
    if isinstance(value, Source):
        return value.leaf.value
    if isinstance(value, (Value, Expr)):
        return value.value
    raise TypeError(value)


def nest(c):
    tree = {}
    for path, value in c.v2.items():
        node = tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = evaluate(value)
    return tree


def unclaimed_paths(c):
    return {" > ".join(lf.path) for lf in c.unclaimed()}


def validated_output(c):
    fragment = nest(c)["solver"]["output settings"]
    fragment.setdefault("solution", {})
    fragment.setdefault("report", {"frequency": 1})
    return OutputModel.model_validate(replace_spaces_in_keys(fragment))


def test_module_contract():
    assert output.ORDER == 60
    assert sorted(output.OWNS) == sorted(
        [
            "write output",
            "report",
            "units",
            "partitioner",
            "safe",
            "linear solver options",
            "mesh quality remediation angle threshold",
            "reset face area",
            "append mesh name",
            "zone id to name",
            "variable name alias",
        ]
    )


def test_o1_solution_output_moves_with_an_integer_frequency():
    deck = {
        "write output": {
            "format": "vtk",
            "surface variables": ["p", "T"],
            "volume variables": ["V", "p"],
            "frequency": 500,
        }
    }
    c = convert(deck)
    assert c.unclaimed() == []
    solution = OUT + ("solution",)
    assert c.v2[solution + ("format",)] == Source(c.v1.leaf("write output", "format"))
    assert evaluate(c.v2[solution + ("frequency",)]) == 500
    assert nest(c)["solver"]["output settings"]["solution"]["surface variables"] == [
        "p",
        "T",
    ]
    model = validated_output(c)
    assert model.solution.frequency.volume_data == 500
    assert model.solution.frequency.checkpoint == 500


def test_o2_frequency_dict_moves_every_key_unchanged():
    keys = [
        "volume data",
        "surface data",
        "volume interpolate",
        "fwh interpolate",
        "fwh wall data",
        "checkpoint",
    ]
    frequency = {k: 100 + i for i, k in enumerate(keys)}
    frequency.update({f"{k} start": 5 + i for i, k in enumerate(keys)})
    c = convert({"write output": {"frequency": frequency}})
    assert c.unclaimed() == []
    assert set(nest(c)["solver"]["output settings"]["solution"]["frequency"]) == set(
        frequency
    )
    model = validated_output(c)
    assert model.solution.frequency.volume_data == 100
    assert model.solution.frequency.fwh_wall_data_start == 9


def test_o2_underscored_frequency_keys_fold_to_spaces():
    c = convert({"write output": {"frequency": {"volume_data": 7, "checkpoint": 9}}})
    assert OUT + ("solution", "frequency", "volume data") in c.v2


def test_o3_scripts_and_fwh_move_to_solution_and_immersed_boundary_drops():
    deck = {
        "write output": {
            "scripts": ["a.py"],
            "fwh wall data": True,
            "fwh interpolate": ["s.stl"],
            "immersed boundary surface": [{"zone": 3, "stl": "b.stl"}],
            "surface variables": ["cp", "immersed boundary vector"],
            "volume variables": ["V", "immersed boundary yplus"],
        }
    }
    c = convert(deck)
    assert c.unclaimed() == []
    solution = nest(c)["solver"]["output settings"]["solution"]
    assert set(solution) == {
        "scripts",
        "fwh wall data",
        "fwh interpolate",
        "surface variables",
        "volume variables",
    }
    assert solution["surface variables"] == ["cp"]
    assert solution["volume variables"] == ["V"]
    model = validated_output(c)
    assert model.solution.fwh_wall_data is True


def test_o3_fwh_interpolate_list_and_frequency_do_not_collide():
    deck = {
        "write output": {
            "fwh interpolate": ["s.stl"],
            "frequency": {"fwh interpolate": 10},
        }
    }
    c = convert(deck)
    assert c.unclaimed() == []
    assert OUT + ("solution", "fwh interpolate") in c.v2
    assert OUT + ("solution", "frequency", "fwh interpolate") in c.v2


def test_o4_averaging_moves_to_output_settings():
    deck = {
        "write output": {
            "compute average and rms": True,
            "average start time cycle": 400,
        }
    }
    c = convert(deck)
    assert c.unclaimed() == []
    assert evaluate(c.v2[OUT + ("compute average and rms",)]) is True
    assert evaluate(c.v2[OUT + ("average start time cycle",)]) == 400
    model = validated_output(c)
    assert model.average_start_time_cycle == 400


def test_o5_volume_interpolate_and_alias_have_no_v2_home():
    deck = {
        "write output": {
            "format": "vtk",
            "volume interpolate": ["pts.vtk"],
            "variable_name_alias": {"pressure": "p"},
        }
    }
    c = convert(deck)
    assert unclaimed_paths(c) == {
        "write output > volume interpolate",
        "write output > variable name alias > pressure",
    }
    assert not any("volume interpolate" in p for p in c.v2)


def test_o6_report_frequency_and_residual_scaling():
    c = convert({"report": {"frequency": 10, "scale residuals by volume": True}})
    assert c.unclaimed() == []
    assert evaluate(c.v2[OUT + ("report", "frequency")]) == 10
    assert evaluate(c.v2[OUT + ("report", "scale residuals by volume")]) is True
    assert validated_output(c).report.scale_residuals_by_volume is True


def test_o6_report_without_monitors_forces_or_massflow():
    c = convert({"report": {"frequency": 25}})
    assert set(c.v2) == {OUT + ("report", "frequency")}
    assert c.unclaimed() == []


def test_o7_monitors_keep_their_label_and_move_unchanged():
    deck = {
        "report": {
            "frequency": 1,
            "monitor": {
                "MR_1": {"name": "probe", "point": [0.0, 1.0, 2.0], "variables": ["p"]},
                "MR_2": {"name": "wake", "point": [1.0, 1.0, 2.0], "variables": ["V"]},
            },
        }
    }
    c = convert(deck)
    assert c.unclaimed() == []
    base = OUT + ("report", "monitor")
    assert evaluate(c.v2[base + ("MR_1", "name")]) == "probe"
    assert evaluate(c.v2[base + ("MR_2", "point")]) == [1.0, 1.0, 2.0]
    model = validated_output(c)
    assert set(model.report.monitor) == {"MR_1", "MR_2"}


def test_o8_forces_rename_zone_and_move_the_rest():
    def transform(*, vector):
        return vector

    deck = {
        "report": {
            "frequency": 1,
            "forces": {
                "FR_1": {
                    "name": "wing",
                    "zone": [3, 4],
                    "transform": transform,
                    "reference area": 2.0,
                    "reference length": 0.5,
                    "reference point": [0.25, 0.0, 0.0],
                    "reference pressure": 101325.0,
                    "dimensional": True,
                }
            },
        }
    }
    c = convert(deck)
    assert c.unclaimed() == []
    base = OUT + ("report", "forces", "FR_1")
    assert evaluate(c.v2[base + ("zones",)]) == [3, 4]
    assert base + ("zone",) not in c.v2
    assert evaluate(c.v2[base + ("transform",)]) is transform
    zone_leaf = c.v1.leaf("report", "forces", "FR_1", "zone")
    kinds = {d.v1_path: d.kind for d in c.dispositions}
    assert kinds[zone_leaf.path] == "respelled"
    model = validated_output(c)
    force = model.report.forces["FR_1"]
    assert force.zones == [3, 4]
    assert force.dimensional is True
    assert force.reference_pressure == 101325.0


def test_o8_a_positional_transform_is_a_decision_to_port_it():
    def wind_axes(x, y, z):
        return {"v1": x, "v2": y, "v3": z}

    deck = {"report": {"forces": {"FR_1": {"zone": [3], "transform": wind_axes}}}}
    c = convert(deck)
    [decision] = c.decisions
    assert decision.v2_path == OUT + ("report", "forces", "FR_1", "transform")
    assert decision.question == (
        "rewrite wind_axes as def wind_axes(*, vector), returning the "
        "transformed vector as three numbers"
    )
    assert "v1 called it as transform(x, y, z)" in decision.details
    assert "the transformed vector as three numbers" in decision.details
    assert c.v2[decision.v2_path].text == (
        "DECIDE: rewrite wind_axes as def wind_axes(*, vector), returning the "
        "transformed vector as three numbers "
        "(see conversion report, item 1)"
    )
    assert c.unclaimed() == []


def test_o8_force_without_optional_keys():
    deck = {
        "report": {
            "frequency": 1,
            "forces": {"FR_1": {"name": "all", "zone": [1], "dimensional": False}},
        }
    }
    c = convert(deck)
    assert c.unclaimed() == []
    base = OUT + ("report", "forces", "FR_1")
    assert {p[len(base)] for p in c.v2 if p[: len(base)] == base} == {
        "name",
        "zones",
        "dimensional",
    }


def test_open_question_6_massflow_name_becomes_the_key():
    deck = {
        "report": {
            "frequency": 1,
            "massflow": {
                "MF_1": {"name": "inlet", "zone": [4]},
                "MF_2": {"name": "outlet", "zone": [5, 6]},
            },
        }
    }
    c = convert(deck)
    assert c.unclaimed() == []
    base = OUT + ("report", "mass flow")
    assert evaluate(c.v2[base + ("inlet", "zones")]) == [4]
    assert evaluate(c.v2[base + ("outlet", "zones")]) == [5, 6]
    assert not any("MF_1" in p for p in c.v2)
    assert not any(p[-1] == "name" for p in c.v2 if p[: len(base)] == base)
    model = validated_output(c)
    assert set(model.report.mass_flow) == {"inlet", "outlet"}


def test_open_question_6_name_leaf_is_recorded_against_the_zones_path():
    deck = {
        "report": {"frequency": 1, "massflow": {"MF_1": {"name": "in", "zone": [4]}}}
    }
    c = convert(deck)
    zones = OUT + ("report", "mass flow", "in", "zones")
    by_path = {d.v1_path: d for d in c.dispositions}
    name = by_path[("report", "massflow", "mf 1", "name")]
    zone = by_path[("report", "massflow", "mf 1", "zone")]
    assert name.v2_path == zones and zone.v2_path == zones
    assert name.kind == "respelled" and "key" in name.reason


def test_open_question_6_duplicate_names_keep_labels_and_note_it():
    deck = {
        "report": {
            "frequency": 1,
            "massflow": {
                "MF_1": {"name": "flow", "zone": [4]},
                "MF_2": {"name": "flow", "zone": [5]},
            },
        }
    }
    c = convert(deck)
    assert c.unclaimed() == []
    base = OUT + ("report", "mass flow")
    assert evaluate(c.v2[base + ("MF_1", "zones")]) == [4]
    assert evaluate(c.v2[base + ("MF_2", "zones")]) == [5]
    assert any("flow" in n.text and "MF_1" in n.text for n in c.notes)


def test_g1_units_are_dropped():
    c = convert({"units": "SI"})
    assert c.v2 == {}
    assert c.unclaimed() == []
    [only] = c.dispositions
    assert only.kind == "dropped" and "SI" in only.reason


def test_g2_g3_partitioner_and_safe_move_to_solver_settings():
    c = convert({"partitioner": "scotch", "safe": True})
    assert c.unclaimed() == []
    assert evaluate(c.v2[SETTINGS + ("partitioner",)]) == "scotch"
    assert evaluate(c.v2[SETTINGS + ("safe",)]) is True
    model = SolverSettingsPropertiesModel.model_validate(
        {"type": "compressible", "partitioner": "scotch", "safe": True}
    )
    assert model.safe is True


def test_g2_metis_is_dropped_with_a_note_so_the_deck_uses_scotch():
    c = convert({"partitioner": "metis"})
    assert c.unclaimed() == []
    assert SETTINGS + ("partitioner",) not in c.v2
    (note,) = c.notes
    assert "research versions of zCFD" in note.text
    assert note.v1_paths == (("partitioner",),)


def test_g4_compressible_linear_solver_options_keep_petsc_spelling():
    deck = {
        "equations": "euler",
        "linear solver options": {
            "flow": {"-ksp_type": "gmres", "-ksp_rtol": "1.0e-4", "-ksp_monitor": ""},
            "turbulence": {"-ksp_type": "bcgs"},
            "rbf": {"-ksp_rtol": "1.0e-6"},
            "double precision": True,
        },
    }
    c = convert(deck)
    assert [lf for lf in c.unclaimed() if lf.path[0] == "linear solver options"] == []
    base = SETTINGS + ("linear solver options",)
    assert evaluate(c.v2[base + ("flow", "-ksp_type")]) == "gmres"
    assert evaluate(c.v2[base + ("double precision",)]) is True
    fragment = nest(c)["solver"]["solver settings"]["linear solver options"]
    model = LinearSolverOptions.model_validate(replace_spaces_in_keys(fragment))
    assert model.flow["-ksp_type"] == "gmres"
    assert model.double_precision is True


def test_g4_incompressible_linear_solver_options():
    deck = {
        "equations": "incomp viscous",
        "linear solver options": {
            "flow": {"-ksp_type": "fgmres", "-ksp_max_it": "30"},
            "turbulence": {"-ksp_type": "bcgs"},
            "rbf": {"-ksp_type": "fgmres"},
            "double precision": False,
        },
    }
    c = convert(deck)
    assert all(lf.path[0] != "linear solver options" for lf in c.unclaimed())
    fragment = nest(c)["solver"]["solver settings"]["linear solver options"]
    model = IncompressibleLinearSolverOptions.model_validate(
        replace_spaces_in_keys(fragment)
    )
    assert model.flow["-ksp_max_it"] == "30"
    assert model.double_precision is False


def test_g4_a_numeric_petsc_value_is_respelled_as_text():
    deck = {"linear solver options": {"flow": {"-ksp_rtol": 1e-4}}}
    c = convert(deck)
    path = SETTINGS + ("linear solver options", "flow", "-ksp_rtol")
    assert c.v2[path] == Value("0.0001")
    [d] = c.dispositions
    assert d.kind == "respelled"


def test_g6_remediation_threshold_moves():
    c = convert({"mesh quality remediation angle threshold": 75.0})
    assert c.unclaimed() == []
    path = SETTINGS + ("mesh quality remediation angle threshold",)
    assert evaluate(c.v2[path]) == 75.0
    model = SolverSettingsPropertiesModel.model_validate(
        {"type": "compressible", "mesh_quality_remediation_angle_threshold": 75.0}
    )
    assert model.mesh_quality_remediation_angle_threshold == 75.0


def test_g6_a_non_positive_threshold_is_dropped_with_a_reason():
    c = convert({"mesh quality remediation angle threshold": -1.0})
    assert c.v2 == {}
    [d] = c.dispositions
    assert d.kind == "dropped" and "ran without it" in d.reason


def test_g6_a_positive_threshold_on_an_incompressible_deck_is_a_decision():
    c = convert(
        {"equations": "incomp RANS", "mesh quality remediation angle threshold": 75.0}
    )
    path = SETTINGS + ("mesh quality remediation angle threshold",)
    [d] = c.decisions
    assert d.v2_path == path
    assert "compressible" in d.question


@pytest.mark.parametrize(
    "key, value",
    [
        ("reset face area", 1.0e-12),
        ("append mesh name", "_fine"),
        ("zone_id_to_name", {"3": "wing"}),
        ("variable_name_alias", {"pressure": "p"}),
    ],
)
def test_g7_to_g9_keys_with_no_v2_home_stay_unclaimed(key, value):
    c = convert({key: value, "partitioner": "scotch"})
    assert c.v2 == {SETTINGS + ("partitioner",): Source(c.v1.leaf("partitioner"))}
    assert {lf.path[0] for lf in c.unclaimed()} == {norm_key(key)}


def test_nothing_outside_the_owned_keys_is_claimed():
    deck = {
        "equations": "euler",
        "reference": "IC_1",
        "time marching": {"cfl": 1.0},
        "BC_1": {"type": "wall", "zone": [1]},
        "partitioner": "metis",
    }
    c = convert(deck)
    assert {lf.path[0] for lf in c.unclaimed()} == {
        "equations",
        "reference",
        "time marching",
        "bc 1",
    }


def test_every_v2_path_is_under_the_solver_block():
    deck = {
        "units": "SI",
        "partitioner": "metis",
        "safe": False,
        "write output": {"format": "vtk", "frequency": 10},
        "report": {"frequency": 1, "massflow": {"MF_1": {"name": "a", "zone": [1]}}},
    }
    c = convert(deck)
    assert c.unclaimed() == []
    assert all(p[0] == "solver" for p in c.v2)


def test_a_whole_output_deck_validates_against_the_v2_schema():
    deck = {
        "write output": {
            "format": "vtk",
            "surface variables": ["p"],
            "volume variables": ["p", "V"],
            "frequency": {"volume data": 100, "surface data": 200, "checkpoint": 300},
            "compute average and rms": True,
            "average start time cycle": 50,
        },
        "report": {
            "frequency": 5,
            "scale residuals by volume": True,
            "monitor": {"MR_1": {"name": "m", "point": [0, 0, 0], "variables": ["p"]}},
            "forces": {"FR_1": {"name": "f", "zone": [1], "dimensional": False}},
            "massflow": {"MF_1": {"name": "mf", "zone": [2]}},
        },
    }
    c = convert(deck)
    assert c.unclaimed() == []
    model = validated_output(c)
    assert model.solution.frequency.surface_data == 200
    assert model.report.frequency == 5
    assert list(model.report.mass_flow) == ["mf"]


def test_open_question_6_names_that_differ_only_in_case_keep_labels():
    deck = {
        "report": {
            "frequency": 1,
            "massflow": {
                "MF_1": {"name": "In", "zone": [4]},
                "MF_2": {"name": "in", "zone": [5]},
            },
        }
    }
    c = convert(deck)
    base = OUT + ("report", "mass flow")
    assert base + ("MF_1", "zones") in c.v2
    assert base + ("MF_2", "zones") in c.v2
