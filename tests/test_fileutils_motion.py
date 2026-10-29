"""zCFD_Result.rotating / .translating read the motion in a model's transforms."""

import pprint
from pathlib import Path

import pytest

from zutil.fileutils import zCFD_Result


def _write_case(directory: Path, models: dict) -> str:
    parameters = {
        "config version": 2,
        "solver": {"time settings": {"type": "unsteady"}},
        "model": models,
    }
    path = directory / "case.py"
    path.write_text("parameters = " + pprint.pformat(parameters) + "\n")
    return str(path)


_ROTATION = {"axis": [0.0, 0.0, 1.0], "origin": [0.0, 0.0, 0.0], "omega": 100.0}


def _moving(mesh, motion, moving_mesh):
    return {
        "mesh": mesh,
        "transforms": {"motion": {**motion, "moving mesh": moving_mesh}},
    }


_ROTATING = {"type": "rotating", **_ROTATION}
_TRANSLATING = {"type": "translating", "velocity": [-69.4, 0.0, 0.0]}


def _rotating_motion():
    return _moving("rotor.h5", _ROTATING, True)


def _still():
    return {"mesh": "stator.h5"}


@pytest.mark.parametrize(
    "models, rotating, translating",
    [
        ({"rotor": _rotating_motion()}, True, False),
        ({"train": _moving("train.h5", _TRANSLATING, True)}, False, True),
        ({"rotor": _moving("rotor.h5", _ROTATING, False)}, True, False),
        ({"plate": _moving("plate.h5", _TRANSLATING, False)}, False, True),
        ({"stator": _still()}, False, False),
        (
            {"stator": {"mesh": "stator.h5", "transforms": {"scale": [2.0, 2.0, 2.0]}}},
            False,
            False,
        ),
    ],
)
def test_motion_sets_the_flags(tmp_path, models, rotating, translating):
    result = zCFD_Result(_write_case(tmp_path, models))
    assert (result.rotating, result.translating) == (rotating, translating)


@pytest.mark.parametrize(
    "model",
    [
        {"mesh": "rotor.h5", "motion": _ROTATING},
        {
            "mesh": "rotor.h5",
            "fluid zones": {
                "FZ_1": {"type": "rotating frame", "zones": [0], **_ROTATION}
            },
        },
    ],
)
def test_retired_motion_shapes_are_not_read(tmp_path, model):
    result = zCFD_Result(_write_case(tmp_path, {"rotor": model}))
    assert result.rotating is False


def test_target_model_scopes_the_flags(tmp_path):
    control = _write_case(tmp_path, {"stator": _still(), "rotor": _rotating_motion()})
    assert zCFD_Result(control, model="stator").rotating is False
    assert zCFD_Result(control, model="rotor").rotating is True
    assert zCFD_Result(control).rotating is True
