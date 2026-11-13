import copy
import math

import pytest

from pydantic import ValidationError
from zutil.validate.schemas.boundary_conditions_schemas import (
    OversetInterface,
    PeriodicInterface,
    SlidingInterface,
)
from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive
from zutil.validate.zcfd_schema import ZCFDInputModelBase

UNSTEADY = {
    "type": "unsteady",
    "kind": "dual time stepping",
    "total_time": 1.0,
    "time_step": 0.01,
}


def _config(models, time_settings=None):
    return {
        "config_version": 2,
        "solver": {
            "solver_settings": {"type": "compressible"},
            "equations": {"type": "euler"},
            "numerical_scheme": {"order": "second"},
            "fluid_properties": {"material": "air"},
            "reference_conditions": {
                "IC_1": {
                    "temperature": 300.0,
                    "pressure": 101325.0,
                    "v": [1.0, 0.0, 0.0],
                },
            },
            "initialisation": {"initial_conditions": "IC_1"},
            "time_settings": time_settings or UNSTEADY,
            "convergence_control": {
                "cycles": 10,
                "scheme": {"name": "implicit euler", "stage": 1},
            },
            "output_settings": {
                "solution": {"frequency": 10},
                "report": {"frequency": 10},
            },
        },
        "model": {
            name: {
                "mesh": f"{name}.h5",
                "boundary_conditions": {
                    "wall": {"zones": [1], "type": "wall", "kind": "slip"},
                    **bcs,
                },
                **({"fluid_zones": zones} if zones else {}),
            }
            for name, (bcs, zones) in models.items()
        },
    }


def _sliding(model, boundary, zones=(20,)):
    return {
        "zones": list(zones),
        "type": "interface",
        "kind": "sliding",
        "connection": {"model": model, "boundary": boundary},
    }


def _periodic(boundary, transform, zones=(5,)):
    return {
        "zones": list(zones),
        "type": "interface",
        "kind": "periodic",
        "connection": {"boundary": boundary},
        "transform": transform,
    }


def _overset(models=None):
    bc = {"zones": [10], "type": "interface", "kind": "overset"}
    if models is not None:
        bc["connection"] = {"model": models}
    return bc


def _rotating(**overrides):
    return {
        "type": "rotating",
        "omega": 10.0,
        "axis": [0.0, 0.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        **overrides,
    }


def _moving(config, motion, model="rotor"):
    config["model"][model]["transforms"] = {"motion": motion}
    return config


def _rotor_stator(rotor_bc=None, stator_bc=None):
    return {
        "rotor": (
            {"slide": rotor_bc or _sliding("stator", "slide")},
            None,
        ),
        "stator": (
            {"slide": stator_bc or _sliding("rotor", "slide")},
            None,
        ),
    }


def _rotated(theta, axis=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0)):
    return {
        "type": "rotated",
        "theta": theta,
        "axis": list(axis),
        "origin": list(origin),
    }


def _linear(vector):
    return {"type": "linear", "vector": list(vector)}


def _blade(first, second):
    return {
        "blade": (
            {"lo": _periodic("hi", first), "hi": _periodic("lo", second, zones=(6,))},
            None,
        )
    }


def _valid(config):
    ZCFDInputModelBase.model_validate(copy.deepcopy(config))


def _error(config):
    with pytest.raises(ValidationError) as excinfo:
        ZCFDInputModelBase.model_validate(copy.deepcopy(config))
    return "\n".join(
        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in excinfo.value.errors()
    )


class TestConnectionSchemaExport:
    @pytest.mark.parametrize(
        "cls, expected",
        [
            (
                SlidingInterface,
                {
                    "model": "model",
                    "boundary": "boundary_conditions",
                    "other_model": True,
                    "reciprocal": True,
                },
            ),
            (
                PeriodicInterface,
                {
                    "model": "model",
                    "boundary": "boundary_conditions",
                    "other_model": False,
                    "reciprocal": True,
                },
            ),
            (
                OversetInterface,
                {
                    "model": "model",
                    "boundary": None,
                    "other_model": True,
                    "reciprocal": False,
                },
            ),
        ],
    )
    def test_references_are_exported(self, cls, expected):
        node = cls.model_json_schema()["properties"]["connection"]
        assert node["zcfd"]["references"] == expected


class TestSlidingConnection:
    def test_reciprocal_pair_is_accepted(self):
        _valid(_config(_rotor_stator()))

    def test_model_that_does_not_exist(self):
        message = _error(_config(_rotor_stator(rotor_bc=_sliding("statr", "slide"))))
        assert (
            "model.rotor.boundary_conditions.slide.connection: 'connection' names "
            "model 'statr', which is not a model in this case (models: rotor, "
            "stator)" in message
        )

    def test_own_model(self):
        message = _error(_config(_rotor_stator(rotor_bc=_sliding("rotor", "wall"))))
        assert "'connection' must name another model, not 'rotor' itself" in message

    def test_boundary_that_does_not_exist(self):
        message = _error(_config(_rotor_stator(rotor_bc=_sliding("stator", "slid"))))
        assert (
            "'connection' names boundary 'slid' on 'stator', which does not exist "
            "(boundaries on stator: wall, slide)" in message
        )

    def test_boundary_of_another_kind(self):
        message = _error(_config(_rotor_stator(rotor_bc=_sliding("stator", "wall"))))
        assert (
            "'connection' names 'wall' on 'stator', which is not a sliding "
            "interface" in message
        )

    def test_boundary_that_does_not_connect_back(self):
        models = _rotor_stator(stator_bc=_sliding("hub", "slide"))
        models["hub"] = ({"slide": _sliding("stator", "slide")}, None)
        message = _error(_config(models))
        assert (
            "model.rotor.boundary_conditions.slide.connection: 'connection' names "
            "'slide' on 'stator', whose connection is to 'slide' on 'hub', not "
            "back to 'slide' on 'rotor'" in message
        )

    @pytest.mark.parametrize("field", ["axis", "origin"])
    def test_axis_comes_from_the_motion_or_frame_not_the_interface(self, field):
        bc = {**_sliding("stator", "slide"), field: [0.0, 0.0, 1.0]}
        message = _error(_config(_rotor_stator(rotor_bc=bc)))
        assert field in message

    def test_every_problem_is_reported_together(self):
        models = _rotor_stator(
            rotor_bc=_sliding("statr", "slide"), stator_bc=_sliding("rotor", "slid")
        )
        message = _error(_config(models))
        assert "names model 'statr'" in message
        assert "names boundary 'slid' on 'rotor'" in message


class TestPeriodicConnection:
    @pytest.mark.parametrize(
        "first, second",
        [
            (_linear([0.0, 0.0, 1.0]), _linear([0.0, 0.0, -1.0])),
            (_rotated(math.radians(30.0)), _rotated(math.radians(-30.0))),
            (
                _rotated(0.5, axis=(0.0, 0.0, 1.0)),
                _rotated(0.5, axis=(0.0, 0.0, -1.0), origin=(0.0, 0.0, 4.0)),
            ),
        ],
    )
    def test_inverse_pair_is_accepted(self, first, second):
        _valid(_config(_blade(first, second)))

    def test_boundary_that_does_not_exist(self):
        models = _blade(_linear([1.0, 0.0, 0.0]), _linear([-1.0, 0.0, 0.0]))
        models["blade"][0]["lo"]["connection"] = {"boundary": "hj"}
        message = _error(_config(models))
        assert (
            "'connection' names boundary 'hj' on 'blade', which does not exist "
            "(boundaries on blade: wall, lo, hi)" in message
        )

    def test_own_boundary(self):
        models = _blade(_linear([1.0, 0.0, 0.0]), _linear([-1.0, 0.0, 0.0]))
        models["blade"][0]["lo"]["connection"] = {"boundary": "lo"}
        message = _error(_config(models))
        assert (
            "'connection' names its own boundary 'lo'; it must name the boundary "
            "on the other side" in message
        )

    def test_translations_that_are_not_inverses(self):
        message = _error(
            _config(_blade(_linear([0.0, 0.0, 1.0]), _linear([0.0, 0.0, 1.0])))
        )
        assert (
            "the transforms on 'hi' and 'lo' are not inverses of each other: a "
            "translation of (0, 0, 1) on 'hi' needs (0, 0, -1) on 'lo', which "
            "gives (0, 0, 1)" in message
        )
        assert message.count("are not inverses") == 1

    def test_rotations_that_are_not_inverses(self):
        message = _error(_config(_blade(_rotated(0.5), _rotated(0.5))))
        assert (
            "the transforms on 'hi' and 'lo' are not inverses of each other: a "
            "rotation of 0.5 rad about (0, 0, 1) through (0, 0, 0) on 'hi' needs "
            "-0.5 rad about the same axis on 'lo', which gives 0.5 rad about "
            "(0, 0, 1) through (0, 0, 0)" in message
        )

    def test_rotations_about_different_axes(self):
        message = _error(
            _config(_blade(_rotated(0.5), _rotated(-0.5, origin=(1.0, 0.0, 0.0))))
        )
        assert "are not inverses of each other" in message

    def test_rotation_paired_with_translation(self):
        message = _error(_config(_blade(_rotated(0.5), _linear([1.0, 0.0, 0.0]))))
        assert (
            "the transforms on 'hi' and 'lo' are not inverses of each other: "
            "'hi' is linear and 'lo' is rotated" in message
        )


class TestOversetConnection:
    def test_without_connection_is_accepted(self):
        _valid(_config({"base": ({}, None), "collar": ({"fringe": _overset()}, None)}))

    def test_named_models_are_accepted(self):
        _valid(
            _config(
                {"base": ({}, None), "collar": ({"fringe": _overset(["base"])}, None)}
            )
        )

    def test_model_that_does_not_exist(self):
        message = _error(
            _config(
                {"base": ({}, None), "collar": ({"fringe": _overset(["bsae"])}, None)}
            )
        )
        assert (
            "'connection' names model 'bsae', which is not a model in this case "
            "(models: base, collar)" in message
        )

    def test_own_model(self):
        message = _error(
            _config(
                {"base": ({}, None), "collar": ({"fringe": _overset("collar")}, None)}
            )
        )
        assert "'connection' must name another model, not 'collar' itself" in message


def _collar(**fringes):
    return _config({"base": ({}, None), "collar": (fringes, None)})


def _halos(value=None, zones=(10,)):
    bc = {"zones": list(zones), "type": "interface", "kind": "overset"}
    if value is not None:
        bc["create_overset_halos"] = value
    return bc


def _section_errors(config):
    _model, nodes = validate_zcfd_comprehensive(copy.deepcopy(config))
    return [
        (n["field_path"], n["display_message"])
        for n in nodes
        if n.get("severity", "error") == "error"
    ]


SOLVER_HALOS = (
    "'create overset halos' is not a setting of the solver: it is a setting of "
    'each overset interface, as "boundary conditions": {"<name>": {"type": '
    '"interface", "kind": "overset", "create overset halos": ...}}. Every '
    "overset interface on one model gives the same value."
)


class TestCreateOversetHalos:
    def test_defaults_to_true_on_the_interface(self):
        assert OversetInterface.model_validate(_halos()).create_overset_halos is True

    def test_reaches_the_cpp_dict_on_the_interface(self):
        config = _collar(fringe=_halos(False))
        cpp = ZCFDInputModelBase.model_validate(copy.deepcopy(config)).to_cpp_dict()
        fringe = cpp["model"]["collar"]["boundary_conditions"]["fringe"]
        assert fringe["create_overset_halos"] is False
        assert "create_overset_halos" not in cpp["solver"]

    def test_is_exported_as_model_wide(self):
        node = OversetInterface.model_json_schema()["properties"][
            "create_overset_halos"
        ]
        assert node["zcfd"]["model_wide"] is True

    def test_interfaces_that_agree_are_accepted(self):
        _valid(_collar(a=_halos(False), b=_halos(False, zones=(11,))))

    def test_interfaces_that_disagree_are_rejected(self):
        message = _error(_collar(a=_halos(False), b=_halos(True, zones=(11,))))
        assert (
            "model.collar.boundary_conditions.b.create_overset_halos: 'create "
            "overset halos' is True here and False on 'a'. The solver applies one "
            "value to the whole of model 'collar', so every boundary on it gives "
            "the same one." in message
        )

    def test_an_unset_interface_takes_the_default_in_the_comparison(self):
        message = _error(_collar(a=_halos(False), b=_halos(zones=(11,))))
        assert "is True (the default) here and False on 'a'" in message

    def test_interfaces_on_different_models_may_differ(self):
        config = _collar(fringe=_halos(False))
        config["model"]["base"]["boundary_conditions"]["hole"] = _halos(True, (20,))
        _valid(config)

    def test_the_solver_level_key_is_rejected(self):
        config = _collar(fringe=_halos())
        config["solver"]["create_overset_halos"] = False
        assert _section_errors(config) == [
            ("solver.create_overset_halos", SOLVER_HALOS)
        ]


MOVING_MESH_NEEDS_AN_UNSTEADY_INTERFACE = (
    "'moving mesh': True only applies when the model has an overset or sliding "
    "interface and the run is unsteady and compressible. For a steady rotor, or "
    "an incompressible run, set 'moving mesh': False to solve the flow in the "
    "moving frame."
)


class TestMotionOnInterfaceMeshes:
    """Which interfaces let a mesh move."""

    def test_an_overset_mesh_may_move(self):
        config = _config({"base": ({}, None), "collar": ({"fringe": _overset()}, None)})
        _valid(_moving(config, _rotating(moving_mesh=True), model="collar"))

    def test_a_periodic_interface_does_not_let_a_mesh_move(self):
        config = _config(_blade(_linear([1.0, 0.0, 0.0]), _linear([-1.0, 0.0, 0.0])))
        model = next(iter(config["model"]))
        message = _error(_moving(config, _rotating(moving_mesh=True), model=model))
        assert (
            f"model.{model}.transforms.motion.moving_mesh: "
            f"{MOVING_MESH_NEEDS_AN_UNSTEADY_INTERFACE}"
        ) in message

    def test_a_whole_mesh_frame_on_a_sliding_mesh_is_accepted(self):
        _valid(
            _moving(
                _config(_rotor_stator(), time_settings={"type": "steady"}),
                _rotating(),
            )
        )


class TestCoupledMeshesNeedAnImplicitScheme:
    @pytest.mark.parametrize(
        "scheme", [{"name": "runge kutta", "stage": 5}, {"name": "euler"}]
    )
    def test_explicit_scheme_is_rejected_with_a_sliding_interface(self, scheme):
        config = _config(_rotor_stator())
        config["solver"]["convergence_control"]["scheme"] = scheme
        message = _error(config)
        assert (
            "overset or sliding interfaces are declared on 'rotor' and 'stator'"
            in message
        )

    def test_explicit_scheme_is_accepted_with_only_periodic_interfaces(self):
        config = _config(_blade(_linear([1.0, 0.0, 0.0]), _linear([-1.0, 0.0, 0.0])))
        config["solver"]["convergence_control"]["scheme"] = {
            "name": "runge kutta",
            "stage": 5,
        }
        _valid(config)
