"""``transforms > motion`` on a model, one block for a moving mesh and a
frame on the whole mesh, and the rotating fluid zone, a frame on cell zones of
a still mesh: their compatibility rules, and the errors that name what
replaced the retired motion and frame settings."""

import copy

import pytest
from pydantic import TypeAdapter, ValidationError

from zutil.validate.error_mapping import map_errors_to_nodes
from zutil.validate.schemas.motion_schemas import (
    Motion,
    RotatingMotion,
    TranslatingMotion,
)
from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive
from zutil.validate.validation_utils import replace_spaces_in_keys
from zutil.validate.zcfd_schema import ZCFDInputModelBase

ROTATION = {"axis": [0.0, 0.0, 1.0], "origin": [0.0, 0.0, 0.0], "omega": 10.0}
STEADY = {"type": "steady"}
UNSTEADY = {
    "type": "unsteady",
    "kind": "dual time stepping",
    "total time": 1.0,
    "time step": 0.01,
}


def _rotating(**settings):
    return {"type": "rotating", **ROTATION, **settings}


def _config(
    motion=None,
    zones=None,
    velocity=None,
    interface=True,
    time_settings=UNSTEADY,
    incompressible=False,
    transforms=None,
):
    rotor = {
        "mesh": "rotor.h5",
        "boundary conditions": {
            "wall": {
                "zones": [1],
                "type": "wall",
                "kind": "slip",
                **({"velocity": velocity} if velocity else {}),
            },
        },
    }
    models = {"rotor": rotor}
    if interface:
        rotor["boundary conditions"]["slide"] = {
            "zones": [20],
            "type": "interface",
            "kind": "sliding",
            "connection": {"model": "stator", "boundary": "slide"},
        }
        models["stator"] = {
            "mesh": "stator.h5",
            "boundary conditions": {
                "slide": {
                    "zones": [21],
                    "type": "interface",
                    "kind": "sliding",
                    "connection": {"model": "rotor", "boundary": "slide"},
                },
            },
        }
    if transforms is not None or motion is not None:
        rotor["transforms"] = dict(transforms or {})
        if motion is not None:
            rotor["transforms"]["motion"] = motion
    if zones is not None:
        rotor["fluid zones"] = zones
    solver = {
        "solver settings": {"type": "compressible"},
        "equations": {"type": "euler"},
        "numerical scheme": {"order": "second"},
        "fluid properties": {"material": "air"},
        "reference conditions": {
            "IC_1": {
                "temperature": 300.0,
                "pressure": 101325.0,
                "v": [1.0, 0.0, 0.0],
            },
        },
        "initialisation": {"initial conditions": "IC_1"},
        "time settings": time_settings,
        "convergence control": {
            "cycles": 10,
            "scheme": {"name": "implicit euler", "stage": 1},
        },
        "output settings": {
            "solution": {"frequency": 10},
            "report": {"frequency": 10},
        },
    }
    if incompressible:
        solver["solver settings"] = {"type": "incompressible"}
        solver["equations"] = {"type": "viscous"}
        solver["reference conditions"]["IC_1"]["reynolds no"] = 100.0
        solver["convergence control"] = {"cycles": 10, "scheme": {"name": "coupled"}}
    return {"config_version": 2, "solver": solver, "model": models}


def _nodes(config):
    _model, nodes = validate_zcfd_comprehensive(copy.deepcopy(config))
    return nodes


def _errors(config):
    return [
        (n["field_path"], n["display_message"])
        for n in _nodes(config)
        if n.get("severity", "error") == "error"
    ]


def _full_model_message(config):
    deck = replace_spaces_in_keys(copy.deepcopy(config))
    with pytest.raises(ValidationError) as excinfo:
        ZCFDInputModelBase.model_validate(deck)
    return "\n".join(
        n["message"] for n in map_errors_to_nodes(excinfo.value.errors(), deck)
    )


def _cpp(config):
    return ZCFDInputModelBase.model_validate(
        replace_spaces_in_keys(copy.deepcopy(config))
    ).to_cpp_dict()


MOTION = "model.rotor.transforms.motion"

MOVING_MESH_NEEDS_AN_UNSTEADY_INTERFACE = (
    "'moving mesh': True only applies when the model has an overset or sliding "
    "interface and the run is unsteady and compressible. For a steady rotor, or "
    "an incompressible run, set 'moving mesh': False to solve the flow in the "
    "moving frame."
)
RAMP_NEEDS_A_MOVING_MESH = (
    "'ramp time' only applies when 'moving mesh' is True. A frame runs at its "
    "full rate from the first step: remove 'ramp time'."
)
ROTATE_NODES_NEEDS_A_MOVING_MESH = (
    "'rotate nodes' only applies when 'moving mesh' is True. A frame never moves "
    "the nodes: remove 'rotate nodes'."
)

MODEL_MOTION = (
    "Model 'rotor' sets 'motion', which is not a setting of a model: a mesh's "
    'motion is given in its \'transforms\', as "transforms": {"motion": '
    "{...}}. Set 'moving mesh': True in it for a mesh that moves through the run."
)
TRANSFORM_MATRIX = (
    "'transform matrix' is not a setting of 'transforms': the placement is "
    "'transform', a 4x4 matrix or a function def place(*, location), returning "
    "the placed position as three numbers. 'transform' acts after 'scale', so a "
    "matrix's translation is in metres. If the deck also sets 'scale', give the "
    "translation in metres, or for a non-uniform scale use S @ M @ inv(S), with "
    "S = np.diag([*scale, 1])."
)
ROTATING_ZONE_FRAGMENT = (
    '"fluid zones": {"<name>": {"type": "rotating", "cell zones": [...], "axis": '
    '[...], "origin": [...], "omega": ...}}'
)
ROTATING_FRAME = (
    "Fluid zone 'FZ_1' has type 'rotating frame', which is not a fluid-zone "
    "type. A rotating frame on part of the mesh is a 'rotating' fluid zone on "
    f"its cell zones: {ROTATING_ZONE_FRAGMENT}. A rotating frame on the whole "
    'mesh is given in the model\'s \'transforms\': "transforms": {"motion": '
    '{"type": "rotating", "axis": [...], "origin": [...], "omega": ..., '
    '"moving mesh": False}}.'
)
TRANSLATING_FRAME = (
    "Fluid zone 'FZ_1' has type 'translating frame', which is not a fluid-zone "
    "type. A frame is the motion of its mesh, given in the model's 'transforms': "
    '"transforms": {"motion": {"type": "translating", "velocity": [...], '
    '"moving mesh": False}}. A frame covers the whole mesh.'
)
TRANSLATING_ZONE = (
    "Fluid zone 'FZ_1' has type 'translating', which is not a fluid-zone type. A "
    "mesh's motion is given in the model's 'transforms': \"transforms\": "
    '{"motion": {"type": "translating", "velocity": [...], "moving mesh": ...}}, '
    "with 'moving mesh' True for a mesh that moves through the run, or False for "
    "a frame."
)
MOTION_ZONES = (
    "'zones' is not a setting of 'motion': 'motion' covers the whole mesh. A "
    "rotating frame on part of the mesh is a 'rotating' fluid zone on its cell "
    f"zones: {ROTATING_ZONE_FRAGMENT}. A region that moves through the run is a "
    "mesh of its own, joined to the rest by a sliding or overset interface."
)
TRANSLATION_FUNCTION = (
    "'translation function' is not a setting: a control deck carries no "
    "functions. Give 'velocity', the full value in m/s, and 'ramp time', the "
    "seconds to reach it from rest."
)


def _origin(key):
    return (
        f"'{key}' is not a setting here: the centre of rotation is 'origin', as "
        "on every rotation."
    )


class TestMotionShape:
    def test_the_type_tag_picks_the_motion(self):
        adapter = TypeAdapter(Motion)
        assert isinstance(adapter.validate_python(_rotating()), RotatingMotion)
        assert isinstance(
            adapter.validate_python({"type": "translating", "velocity": [1, 0, 0]}),
            TranslatingMotion,
        )

    def test_rotating_and_translating_at_once_cannot_be_written(self):
        with pytest.raises(ValidationError):
            TypeAdapter(Motion).validate_python(_rotating(velocity=[1.0, 0.0, 0.0]))

    def test_a_motion_is_a_frame_until_the_mesh_is_said_to_move(self):
        motion = RotatingMotion(type="rotating", **ROTATION)
        assert motion.moving_mesh is False
        assert motion.ramp_time is None
        assert motion.rotate_nodes is False
        assert "rotate_nodes" not in TranslatingMotion.model_fields

    def test_a_moving_mesh_reaches_the_solver_dict(self):
        config = _config(
            motion=_rotating(**{"moving mesh": True, "ramp time": 0.05}),
        )
        assert _errors(config) == []
        rotor = _cpp(config)["model"]["rotor"]
        assert "motion" not in rotor
        motion = rotor["transforms"]["motion"]
        assert motion["type"] == "rotating"
        assert motion["omega"] == 10.0
        assert motion["moving_mesh"] is True
        assert motion["ramp_time"] == 0.05
        assert motion["rotate_nodes"] is False
        assert rotor["transforms"]["scale"] == [1.0, 1.0, 1.0]


class TestMovingMeshNeedsAnUnsteadyCompressibleInterface:
    @pytest.mark.parametrize(
        "settings",
        [
            {"time_settings": STEADY},
            {"incompressible": True},
            {"interface": False},
        ],
        ids=["steady", "incompressible", "no interface"],
    )
    def test_a_moving_mesh_is_rejected(self, settings):
        config = _config(motion=_rotating(**{"moving mesh": True}), **settings)
        assert _errors(config) == [
            (f"{MOTION}.moving_mesh", MOVING_MESH_NEEDS_AN_UNSTEADY_INTERFACE)
        ]
        assert (
            f"{MOTION}.moving_mesh: {MOVING_MESH_NEEDS_AN_UNSTEADY_INTERFACE}"
            in _full_model_message(config)
        )

    @pytest.mark.parametrize(
        "motion",
        [
            _rotating(**{"moving mesh": True}),
            _rotating(**{"moving mesh": True, "rotate nodes": True, "ramp time": 0.05}),
            {
                "type": "translating",
                "velocity": [1.0, 0.0, 0.0],
                "moving mesh": True,
                "ramp time": 1.5,
            },
        ],
        ids=["rotor", "rotor with every setting", "translation"],
    )
    def test_an_unsteady_compressible_interface_mesh_may_move(self, motion):
        assert _errors(_config(motion=motion)) == []

    @pytest.mark.parametrize(
        "settings",
        [
            {"time_settings": STEADY},
            {"incompressible": True},
            {"interface": False},
            {},
        ],
        ids=["steady", "incompressible", "no interface", "unsteady interface"],
    )
    def test_a_whole_mesh_frame_needs_nothing(self, settings):
        assert _errors(_config(motion=_rotating(), **settings)) == []


class TestMovingMeshOnlySettings:
    def test_ramp_time_on_a_frame_is_rejected(self):
        config = _config(motion=_rotating(**{"ramp time": 1.0}))
        assert _errors(config) == [(f"{MOTION}.ramp_time", RAMP_NEEDS_A_MOVING_MESH)]

    @pytest.mark.parametrize("value", [True, False])
    def test_rotate_nodes_on_a_frame_is_rejected(self, value):
        config = _config(motion=_rotating(**{"rotate nodes": value}))
        assert _errors(config) == [
            (f"{MOTION}.rotate_nodes", ROTATE_NODES_NEEDS_A_MOVING_MESH)
        ]

    def test_rotate_nodes_on_a_translation(self):
        config = _config(
            motion={
                "type": "translating",
                "velocity": [1.0, 0.0, 0.0],
                "moving mesh": True,
                "rotate nodes": True,
            }
        )
        assert _errors(config) == [
            (
                f"{MOTION}.rotate_nodes",
                "'rotate nodes' is not a setting of translating 'motion': a "
                "translating mesh always moves its nodes.",
            )
        ]


IDLE_MOTION = (
    "'motion' has a zero rate, so the mesh never moves but its interfaces are "
    "re-coupled every step, which slows the run: give a non-zero 'omega' or "
    "'velocity', or remove 'motion'."
)


class TestIdleMotionWarns:
    def _warnings(self, config):
        nodes = _nodes(config)
        assert all(n["severity"] == "warning" for n in nodes)
        return [(n["field_path"], n["display_message"]) for n in nodes]

    def test_rotation_at_zero_omega(self):
        config = _config(motion=_rotating(omega=0.0, **{"moving mesh": True}))
        assert self._warnings(config) == [(MOTION, IDLE_MOTION)]

    def test_translation_at_zero_velocity(self):
        config = _config(
            motion={
                "type": "translating",
                "velocity": [0.0, 0.0, 0.0],
                "moving mesh": True,
            }
        )
        assert self._warnings(config) == [(MOTION, IDLE_MOTION)]

    def test_a_placement_matrix_does_not_move_the_mesh(self):
        config = _config(
            motion=_rotating(omega=0.0, **{"moving mesh": True}),
            transforms={
                "transform": [[1, 0, 0, 2.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            },
        )
        assert self._warnings(config) == [(MOTION, IDLE_MOTION)]

    @pytest.mark.parametrize(
        "motion",
        [_rotating(**{"moving mesh": True}), _rotating(omega=0.0)],
        ids=["turning", "zero-rate frame"],
    )
    def test_quiet(self, motion):
        assert self._warnings(_config(motion=motion)) == []

    def test_the_run_goes_ahead_with_the_warning_logged(self, caplog):
        from zutil.validate.deck import check_deck

        report = check_deck(
            _config(motion=_rotating(omega=0.0, **{"moving mesh": True}))
        )
        assert report.ok
        with caplog.at_level("WARNING", logger="zutil.validate"):
            report.log_warnings()
        assert IDLE_MOTION in caplog.text


class TestRetiredKeys:
    def test_model_level_motion(self):
        config = _config()
        config["model"]["rotor"]["motion"] = _rotating(**{"moving mesh": True})
        assert _errors(config) == [("model.rotor.motion", MODEL_MOTION)]
        assert MODEL_MOTION in _full_model_message(config)

    def test_transform_matrix(self):
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        config = _config(transforms={"transform matrix": identity})
        assert _errors(config) == [
            ("model.rotor.transforms.transform_matrix", TRANSFORM_MATRIX)
        ]
        assert TRANSFORM_MATRIX in _full_model_message(config)

    @pytest.mark.parametrize(
        "zone, text",
        [
            ({"type": "rotating frame", "zones": [0], **ROTATION}, ROTATING_FRAME),
            (
                {"type": "translating frame", "zones": [0], "velocity": [1, 0, 0]},
                TRANSLATING_FRAME,
            ),
            (
                {"type": "translating", "zones": [0], "velocity": [1, 0, 0]},
                TRANSLATING_ZONE,
            ),
        ],
        ids=["rotating frame", "translating frame", "translating"],
    )
    def test_fluid_zone_types_name_the_motion_block(self, zone, text):
        config = _config(zones={"FZ_1": zone})
        assert _errors(config) == [("model.rotor.fluid_zones.FZ_1.type", text)]
        assert text in _full_model_message(config)

    @pytest.mark.parametrize(
        "motion, key, message",
        [
            (
                _rotating(**{"move mesh": True}),
                "move_mesh",
                "'move mesh' is not a setting of 'motion': the flag is 'moving mesh'.",
            ),
            (
                _rotating(ramp={"time": 1.0}),
                "ramp",
                "'ramp' is not a setting of 'motion': a mesh that starts from rest "
                "is given 'ramp time', the seconds to reach its full rate, with "
                "'moving mesh': True.",
            ),
            (
                {
                    "type": "translating",
                    "velocity": [1.0, 0.0, 0.0],
                    "translation function": lambda *, time, time_step: {
                        "velocity": (10.0 * time, 0.0, 0.0)
                    },
                },
                "translation_function",
                TRANSLATION_FUNCTION,
            ),
            (_rotating(centre=[0.0, 0.0, 0.0]), "centre", _origin("centre")),
            (_rotating(center=[0.0, 0.0, 0.0]), "center", _origin("center")),
            (_rotating(spin=True), "spin", 'Unknown key "spin".'),
        ],
        ids=[
            "move mesh",
            "ramp",
            "translation function",
            "centre",
            "center",
            "unknown key",
        ],
    )
    def test_a_retired_motion_key(self, motion, key, message):
        assert _errors(_config(motion=motion)) == [(f"{MOTION}.{key}", message)]

    @pytest.mark.parametrize("moving_mesh", [False, True])
    def test_zones_are_not_a_setting_of_motion(self, moving_mesh):
        config = _config(
            motion=_rotating(**{"moving mesh": moving_mesh, "zones": [17]}),
            interface=False,
        )
        assert _errors(config) == [(f"{MOTION}.zones", MOTION_ZONES)]

    def test_center_on_a_wall_velocity(self):
        velocity = {**_rotating(omega=1.0), "center": [0.0, 0.0, 0.0]}
        config = _config(velocity=velocity)
        assert _errors(config) == [
            ("model.rotor.boundary_conditions.wall.velocity.center", _origin("center"))
        ]
        assert (
            f"model.rotor.boundary_conditions.wall.velocity.center: {_origin('center')}"
        ) in _full_model_message(config)


class TestMotionValues:
    def test_an_unknown_motion_type(self):
        config = _config(motion={"type": "spinning", **ROTATION})
        assert _errors(config) == [
            (MOTION, "Type must be one of 'rotating', 'translating'.")
        ]

    @pytest.mark.parametrize("ramp_time", [0.0, -1.0])
    def test_ramp_time_must_be_positive(self, ramp_time):
        config = _config(
            motion=_rotating(**{"moving mesh": True, "ramp time": ramp_time})
        )
        assert _errors(config) == [
            (f"{MOTION}.ramp_time", "Ramp time must be greater than 0 s.")
        ]
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            RotatingMotion(type="rotating", **ROTATION, ramp_time=ramp_time)


def _wheel(**settings):
    return {"type": "rotating", "cell zones": [17], **ROTATION, **settings}


ONE_FRAME = (
    "'type': 'rotating' only applies when the model has no 'transforms > "
    "motion' and no overset or sliding interface. A model has one frame: "
    "rotating fluid zones on its cell zones, or 'transforms > motion' for the "
    "whole mesh. A mesh with an overset or sliding interface turns as a whole, "
    "so it takes 'transforms > motion' only. Remove 'motion' from "
    "'transforms', or remove the rotating fluid zones and turn the whole mesh "
    'with "transforms": {"motion": {"type": "rotating", "axis": [...], '
    '"origin": [...], "omega": ..., "moving mesh": False}}.'
)
STILL_MESH = (
    "'{key}' is not a setting of a rotating fluid zone: it is a frame on a "
    "still mesh. A mesh that moves through the run is given in the model's "
    '\'transforms\', as "transforms": {"motion": {"type": "rotating", "axis": '
    '[...], "origin": [...], "omega": ..., "moving mesh": True}}.'
)
FULL_RATE = (
    "'{key}' is not a setting of a rotating fluid zone: it is a frame on a "
    "still mesh and turns at its full 'omega' from the first step. A mesh that "
    "starts from rest is given in the model's 'transforms', as \"transforms\": "
    '{"motion": {"type": "rotating", "axis": [...], "origin": [...], "omega": '
    '..., "moving mesh": True, "ramp time": ...}}.'
)
ZONE = "model.rotor.fluid_zones.wheel"


class TestRotatingFluidZone:
    @pytest.mark.parametrize("incompressible", [False, True])
    def test_a_frame_on_cell_zones_of_a_still_mesh(self, incompressible):
        config = _config(
            zones={"wheel": _wheel(), "fan": _wheel(**{"cell zones": [3, 4]})},
            interface=False,
            incompressible=incompressible,
        )
        assert _errors(config) == []
        zones = _cpp(config)["model"]["rotor"]["fluid_zones"]
        assert zones["wheel"] == {
            "type": "rotating",
            "cell_zones": [17],
            "axis": [0.0, 0.0, 1.0],
            "origin": [0.0, 0.0, 0.0],
            "omega": 10.0,
        }

    @pytest.mark.parametrize(
        "motion",
        [_rotating(), _rotating(**{"moving mesh": True})],
        ids=["frame", "moving mesh"],
    )
    def test_a_model_takes_one_frame(self, motion):
        moving = motion.get("moving mesh", False)
        config = _config(zones={"wheel": _wheel()}, motion=motion, interface=moving)
        assert _errors(config) == [(f"{ZONE}.type", ONE_FRAME)]

    @pytest.mark.parametrize("kind", ["sliding", "overset"])
    def test_a_mesh_with_an_interface_takes_no_rotating_zone(self, kind):
        config = _config(zones={"wheel": _wheel()})
        for model in config["model"].values():
            interface = model["boundary conditions"]["slide"]
            interface["kind"] = kind
            if kind == "overset":
                del interface["connection"]
        assert _errors(config) == [(f"{ZONE}.type", ONE_FRAME)]

    def test_a_cell_zone_turns_with_one_rotating_zone(self):
        config = _config(
            zones={"wheel": _wheel(), "hub": _wheel(**{"cell zones": [3, 17]})},
            interface=False,
        )
        assert _errors(config) == [
            (
                "model.rotor.fluid_zones.hub.cell_zones",
                "Cell zone 17 is in 'cell zones' here and on 'wheel'. Each cell zone "
                "belongs to one rotating fluid zone at most: remove it from one "
                "of them.",
            )
        ]

    def test_a_cell_zone_of_another_model_is_its_own(self):
        config = _config(zones={"wheel": _wheel()}, interface=False)
        config["model"]["stator"] = {
            "mesh": "stator.h5",
            "boundary conditions": {
                "wall": {"zones": [1], "type": "wall", "kind": "slip"}
            },
            "fluid zones": {"vane": _wheel()},
        }
        assert _errors(config) == []

    def test_a_source_may_share_a_rotating_zones_cells(self):
        config = _config(
            zones={
                "wheel": _wheel(),
                "fan": {"type": "porous", "cell zones": [17]},
            },
            interface=False,
        )
        assert _errors(config) == []

    @pytest.mark.parametrize("cell_zones", [None, []], ids=["missing", "empty"])
    def test_cell_zones_are_required(self, cell_zones):
        zone = _wheel()
        if cell_zones is None:
            del zone["cell zones"]
        else:
            zone["cell zones"] = cell_zones
        [(path, _)] = _errors(_config(zones={"wheel": zone}, interface=False))
        assert path == f"{ZONE}.cell_zones"

    @pytest.mark.parametrize(
        "setting, key, message",
        [
            ({"moving mesh": True}, "moving_mesh", STILL_MESH),
            ({"move mesh": True}, "move_mesh", STILL_MESH),
            ({"ramp": {"time": 1.0}}, "ramp", FULL_RATE),
            ({"ramp time": 1.0}, "ramp_time", FULL_RATE),
            (
                {"definition": "wheel.vtp"},
                "definition",
                "'definition' is not a setting of a rotating fluid zone: it "
                "selects its cells with 'cell zones', the mesh cell zone numbers, "
                'as "cell zones": [...].',
            ),
            ({"centre": [0.0, 0.0, 0.0]}, "centre", _origin("centre")),
            ({"center": [0.0, 0.0, 0.0]}, "center", _origin("center")),
        ],
        ids=[
            "moving mesh",
            "move mesh",
            "ramp",
            "ramp time",
            "definition",
            "centre",
            "center",
        ],
    )
    def test_a_frame_on_a_still_mesh_names_what_replaces(self, setting, key, message):
        config = _config(zones={"wheel": _wheel(**setting)}, interface=False)
        text = message.replace("{key}", key.replace("_", " "))
        assert _errors(config) == [(f"{ZONE}.{key}", text)]
        assert text in _full_model_message(config)
