"""The golden overset groups, converted and compared with their develop decks.

The v1 decks are the c31f0bb0d^ testcases. Develop has since renamed models
and BCs and changed some physics, so these tests pin where the conversion
agrees with develop and spell out, as v1 facts, where develop moved on.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from zutil.update_input.convert import convert_deck

GOLDEN = Path(__file__).parent / "decks" / "golden"


@pytest.fixture
def golden(tmp_path, stand_in_mesh) -> Path:
    for name in (
        "euler_rotating_cylinder_group",
        "traintunnel_group",
        "overset_rotations_groups",
    ):
        shutil.copytree(GOLDEN / name, tmp_path / name)
    for mesh in ("cylinder_origin.h5", "cylinder_y_rot.h5"):
        stand_in_mesh(tmp_path / "overset_rotations_groups" / mesh)
    return tmp_path


def convert(path: Path, run_deck):
    result = convert_deck(path)
    return result, run_deck(result.text, path.parent)


def bcs(model: dict) -> list[tuple]:
    return sorted(
        (tuple(bc["zones"]), bc["type"], bc.get("kind"))
        for bc in model["boundary conditions"].values()
    )


def test_euler_rotating_cylinder(golden, run_deck):
    result, p = convert(
        golden / "euler_rotating_cylinder_group" / "overset.py", run_deck
    )
    assert list(p["model"]) == ["background", "cylinder"]
    assert p["model"]["background"]["mesh"] == "backgroundIB.h5"
    assert p["model"]["cylinder"]["mesh"] == "eulerCyl.h5"
    assert bcs(p["model"]["background"]) == [
        ((1, 2, 3, 4), "farfield", "riemann"),
        ((5, 6), "symmetry", None),
    ]
    assert bcs(p["model"]["cylinder"]) == [
        ((11, 12), "symmetry", None),
        ((13,), "interface", "overset"),
        ((14,), "wall", "slip"),
    ]
    motion = p["model"]["cylinder"]["transforms"]["motion"]
    assert motion["type"] == "rotating"
    assert motion["moving mesh"] is True
    assert motion["omega"] == pytest.approx(2.0 * math.pi / 0.36)
    assert "fluid zones" not in p["model"]["cylinder"]
    assert p["solver"]["time settings"]["time step"] == 0.001
    [decision] = result.conversion.decisions
    assert decision.v2_path == ("solver", "time settings", "total time")
    assert decision.options == ("2.4", "0.002")
    solution = p["model"]["cylinder"]["output settings"]["solution"]
    assert solution["volume variables"][-1] == "cellzone"
    assert p["model"]["cylinder"]["output settings"]["report"]["forces"] == {
        "FR_1": {"name": "cyl", "zones": [10]}
    }
    background = p["model"]["background"]["output settings"]
    assert background["solution"]["frequency"] == 5
    assert "frequency" not in p["solver"]["output settings"]["solution"]
    for block, settings in p["solver"]["output settings"].items():
        assert settings.items() <= background[block].items()


def test_traintunnel(golden, run_deck):
    result, p = convert(golden / "traintunnel_group" / "override_roe.py", run_deck)
    assert list(p["model"]) == ["tunnel_roe", "train_roe"]
    assert p["solver"]["time settings"] == {
        "type": "unsteady",
        "total time": 3.6,
        "time step": 0.001,
        "kind": "dual time stepping",
        "order": 1,
        "start": 0,
    }
    assert p["solver"]["convergence control"]["cycles"] == 5
    tunnel = p["model"]["tunnel_roe"]["boundary conditions"]["BC_2"]
    assert tunnel["sponge layer damping"] == {
        "distance": 20.0,
        "damping factor": pytest.approx(2.037495),
        "condition": "IC_1",
    }
    train = p["model"]["train_roe"]
    assert train["transforms"]["transform"][0] == [1.0, 0.0, 0.0, 110.0]
    motion = train["transforms"]["motion"]
    assert motion["type"] == "translating"
    assert motion["moving mesh"] is True
    assert motion["velocity"] == pytest.approx([-69.4, 0, 0])
    assert "fluid zones" not in train
    assert bcs(train) == [
        ((8,), "wall", "slip"),
        ((9,), "interface", "overset"),
        ((10,), "symmetry", None),
    ]
    [decision] = result.conversion.decisions
    assert decision.v2_path == (
        "model",
        "train_roe",
        "transforms",
        "motion",
        "ramp time",
    )
    assert decision.question.startswith(
        "A v2 control deck carries no functions. 'linear_accel' was the "
        "translation function of fluid zone 'FZ_1'."
    )
    assert motion["ramp time"].startswith("DECIDE: ")
    assert result.validated is False
    assert result.conversion.rewrites == []
    assert "def linear_accel" not in result.text
    assert train["output settings"]["solution"]["volume variables"] == [
        "V",
        "p",
        "T",
        "rho",
        "m",
    ]


@pytest.mark.parametrize("case", ["A", "B", "C", "D"])
def test_overset_rotations(golden, case, run_deck):
    result, p = convert(
        golden / "overset_rotations_groups" / f"case{case}.py", run_deck
    )
    assert result.conversion.decisions == []
    assert result.validated is True
    labels = [f"background{case}"] + [f"cylinder{case}{n}" for n in range(1, 5)]
    assert list(p["model"]) == labels

    background = p["model"][labels[0]]
    assert background["mesh"] == "background.h5"
    assert background["boundary conditions"]["BC_1"]["sponge layer damping"] == {
        "distance": 5.0,
        "damping factor": 5.0,
        "condition": "IC_2",
    }
    angle = 0.0 if case == "A" else 25.0
    vector = [math.cos(math.radians(angle)), 0.0, -math.sin(math.radians(angle))]
    assert p["solver"]["reference conditions"]["IC_2"]["v"]["vector"] == pytest.approx(
        vector
    )
    steady = case in "AB"
    assert (p["solver"]["time settings"]["type"] == "steady") is steady

    for label in labels[1:]:
        model = p["model"][label]
        assert model["boundary conditions"]["BC_1"] == {
            "zones": [7],
            "type": "interface",
            "kind": "overset",
            "additional active layers": 2,
        }
        rotation = model["transforms"]["motion"]
        assert rotation["type"] == "rotating"
        assert rotation["moving mesh"] is (case == "D")
        assert "zones" not in rotation
        assert "fluid zones" not in model
        assert abs(rotation["omega"]) == pytest.approx(2.0 * math.pi * 1000.0 / 60.0)
        assert "sponge layer damping" not in str(model)

    dropped = {
        d.v1_path[0]: d.reason
        for d in result.conversion.dispositions
        if d.kind == "dropped"
        and d.v1_path[1:2] in (("mapping",), ("sponge layer damping",))
    }
    assert set(dropped) == {label.lower() for label in labels}
    assert "has no overset interface" in dropped[labels[0].lower()]
    assert "has no farfield boundary" in dropped[labels[1].lower()]


@pytest.mark.parametrize(
    "case, n, reverse, y_rot",
    [
        ("A", 2, False, False),
        ("B", 2, True, True),
        ("B", 3, True, False),
        ("C", 4, False, True),
    ],
)
def test_overset_rotations_keeps_v1_spin_and_placement(
    golden, case, n, reverse, y_rot, run_deck
):
    """v1 reversed a cylinder's spin with ``reverse`` and rotated every mesh by ``angle``.

    Develop later stopped spinning some cylinders and stopped rotating the
    pre-rotated cylinder_y_rot mesh; the converter keeps what v1 ran.
    """
    _, p = convert(golden / "overset_rotations_groups" / f"case{case}.py", run_deck)
    model = p["model"][f"cylinder{case}{n}"]
    assert model["mesh"] == ("cylinder_y_rot.h5" if y_rot else "cylinder_origin.h5")
    assert (model["transforms"]["motion"]["omega"] < 0) is reverse
    angle = 0.0 if case == "A" else math.radians(25.0)
    assert model["transforms"]["transform"][0][0] == pytest.approx(math.cos(angle))


def test_overset_rotations_without_meshes_asks_for_each_frame_at_motion_zones(
    golden,
):
    for mesh in (golden / "overset_rotations_groups").glob("*.h5"):
        mesh.unlink()
    result = convert_deck(golden / "overset_rotations_groups" / "caseA.py")
    decisions = result.conversion.decisions
    assert sorted(d.v2_path for d in decisions) == [
        ("model", f"cylinderA{n}", "transforms", "motion", "zones") for n in range(1, 5)
    ]
    assert all(" rotates " in d.question for d in decisions)
