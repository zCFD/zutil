"""
Schema→consumer contract for the actuator-disc model dispatch.

zTurbine.assign_controller must compare the controller "type" VALUE against
the schema Literal "tsr curve", not "tsr_curve": replace_spaces_in_keys()
underscores KEYS but only lowercases type/kind VALUES. A mismatch silently
loads the no-op ControllerBase (omega=0), zeroing thrust/power on every
curve-driven turbine zone while the run reports COMPLETED. These tests pin
the dispatch to the schema's Literal values and the loud-failure contract
for unknown types.
"""

import pytest

pytest.importorskip("mpi4py")

from pydantic import TypeAdapter

from zutil.validate.schemas.fluid_zones_schemas import ActuatorDiscZone
from zutil.zWind import zActuatorDisc
from zutil.zWind.zController import FixedController, TSRController
from zutil.zWind.zTurbine import zTurbine


def _tsr_curve_disc_zone():
    return {
        "type": "disc",
        "name": "T001",
        "controller": {
            "type": "tsr curve",
            "tip_speed_ratio_curve": [[0.0, 0.0], [4.0, 7.0], [25.0, 3.0]],
        },
        "geometry": {
            "centre": [0.0, 0.0, 90.0],
            "normal": [-1.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "inner_radius": 2.0,
            "outer_radius": 60.0,
        },
        "discretisation": {"type": "disc", "number_of_elements": 12},
        "model": {
            "type": "simple",
            "kind": "turbine",
            "power_model": "curve",
            "power": 0.0,
            "thrust_coefficient": 0.0,
            "power_curve": [[0.0, 0.0], [10.0, 2.0e6], [25.0, 2.0e6]],
            "thrust_coefficient_curve": [[0.0, 0.0], [10.0, 0.8], [25.0, 0.3]],
        },
        "reference_plane": True,
        "reference_point": [-120.0, 0.0, 90.0],
        "update_frequency": 1,
        "rotation_direction": "clockwise",
    }


def _validated_dump(zone_dict):
    model = TypeAdapter(ActuatorDiscZone).validate_python(zone_dict)
    return model.model_dump(exclude_none=True, mode="python")


class TestCurveDrivenDispatch:
    def test_tsr_curve_controller_resolves(self):
        turbine = zTurbine(_validated_dump(_tsr_curve_disc_zone()))
        assert isinstance(turbine.controller, TSRController)

    def test_curve_driven_disc_produces_force(self):
        zone = _validated_dump(_tsr_curve_disc_zone())
        info = {}
        zActuatorDisc.main(zone, 10.0, 1.225, info, "T001")
        assert info["T001_omega"] > 0.0
        assert info["T001_thrust"] > 0.0
        assert info["T001_power"] > 0.0

    def test_fixed_controller_still_resolves(self):
        zone = _tsr_curve_disc_zone()
        zone["controller"] = {"type": "fixed", "pitch": 0.0, "omega": 1.5}
        turbine = zTurbine(_validated_dump(zone))
        assert isinstance(turbine.controller, FixedController)


class TestUnknownTypesFailLoudly:
    def test_unknown_controller_type_raises(self):
        zone = _validated_dump(_tsr_curve_disc_zone())
        zone["controller"]["type"] = "tsr_curve"
        with pytest.raises(ValueError, match="controller type"):
            zTurbine(zone)

    def test_none_controller_type_raises(self):
        zone = _validated_dump(_tsr_curve_disc_zone())
        zone["controller"]["type"] = "none"
        with pytest.raises(ValueError, match="controller type"):
            zTurbine(zone)

    def test_unmatched_model_raises(self):
        zone = _validated_dump(_tsr_curve_disc_zone())
        zone["model"]["type"] = "simple"
        zone["model"]["power_model"] = "fixed"
        zone["model"].pop("kind")
        with pytest.raises(ValueError, match="force model"):
            zTurbine(zone)
