"""
Turbine sub-blocks (TurbineController/Geometry/Discretisation/Model) carry
strict field sets rather than extra="allow".

ActuatorDiscZone's controller/geometry/discretisation/model/name/
rotation_direction are required: a bare {"type": "disc"} would otherwise
validate and then crash the C++ consumer, which does an unguarded .getDict().
"""

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.fluid_zones_schemas import (
    ActuatorDiscZone,
    Fluid_Zone_Base_Model,
    TurbineController,
    TurbineDiscretisation,
    TurbineGeometry,
    TurbineModel,
)
from pydantic import TypeAdapter


def _minimal_actuator_disc(**overrides):
    data = dict(
        type="disc",
        controller={"type": "fixed", "omega": 1.0},
        geometry={
            "centre": [0.0, 0.0, 0.0],
            "normal": [1.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "inner_radius": 1.0,
            "outer_radius": 5.0,
        },
        discretisation={"type": "disc", "number_of_elements": 10},
        model={"type": "simple", "power_model": "fixed"},
        name="turbine_1",
        rotation_direction="clockwise",
    )
    data.update(overrides)
    return data


class TestTurbineSubBlocksStrict:
    def test_turbine_controller_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            TurbineController(type="fixed", made_up=1)

    def test_turbine_controller_accepts_known_fields(self):
        c = TurbineController(
            type="tsr curve",
            tip_speed_ratio=8.0,
            tip_speed_ratio_curve=[[0.0, 0.0], [1.0, 1.0]],
        )
        assert c.type == "tsr curve"

    def test_turbine_controller_rejects_none_type(self):
        with pytest.raises(ValidationError):
            TurbineController(type="none")

    def test_turbine_geometry_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            TurbineGeometry(centre=[0, 0, 0], normal=[1, 0, 0], up=[0, 0, 1], made_up=1)

    def test_turbine_discretisation_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            TurbineDiscretisation(type="disc", made_up=1)

    def test_turbine_model_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            TurbineModel(type="simple", made_up=1)

    def test_turbine_model_type_enum_enforced(self):
        with pytest.raises(ValidationError):
            TurbineModel(type="not_a_real_type")

    def test_turbine_model_accepts_lowercased_bet(self):
        """'BET' is lowercased to 'bet' upstream by replace_spaces_in_keys()
        (any key literally named 'type' gets its string value lowercased),
        so the Literal must accept the lowercase form."""
        m = TurbineModel(type="bet", kind="propellor")
        assert m.type == "bet"


class TestActuatorDiscZoneRequiredFields:
    def test_bare_disc_type_rejected(self):
        """A bare {'type': 'disc'} must not validate."""
        with pytest.raises(ValidationError):
            ActuatorDiscZone(type="disc")

    def test_fully_specified_disc_passes(self):
        model = ActuatorDiscZone(**_minimal_actuator_disc())
        assert model.name == "turbine_1"
        assert model.rotation_direction == "clockwise"

    @pytest.mark.parametrize(
        "missing_field",
        [
            "controller",
            "geometry",
            "discretisation",
            "model",
            "name",
            "rotation_direction",
        ],
    )
    def test_each_required_field_enforced(self, missing_field):
        data = _minimal_actuator_disc()
        del data[missing_field]
        with pytest.raises(ValidationError):
            ActuatorDiscZone(**data)

    def test_zones_rejected(self):
        """A disc selects its cells with its geometry and 'definition'."""
        with pytest.raises(ValidationError, match="zones"):
            ActuatorDiscZone(
                **_minimal_actuator_disc(definition="turbine.vtp", zones=[0]),
            )


class TestRealWorldTurbineShapesStillValidate:
    """Regression guard using the real field shapes seen in testcases/ (hornsrev,
    turbine_spec/ffplane/2p5D, propellor*) after replace_spaces_in_keys() normalises
    space-separated legacy keys to snake_case."""

    def test_hornsrev_style_bet_turbine_zone(self):
        data = {
            "type": "disc",
            "name": "T1",
            "rotation_direction": "clockwise",
            "definition": "turbine_vtp/T1.vtp",
            "update_frequency": 1,
            "controller": {
                "type": "tsr curve",
                "tip_speed_ratio": 9.94,
                "tip_speed_ratio_curve": [[0.0, 0.0], [1.0, 0.0]],
            },
            "geometry": {
                "centre": [423974, 6151447, 70.0],
                "inner_radius": 4.0,
                "outer_radius": 40.0,
                "normal": [-0.98, 0.17, 0.0],
                "up": [0.0, 0.0, 1.0],
            },
            "discretisation": {"type": "disc", "number_of_elements": 10},
            "model": {
                "type": "simple",
                "power_model": "curve",
                "power_curve": [[0.0, 0.0], [1.0, 100.0]],
                "thrust_coefficient": 0.87,
                "thrust_coefficient_curve": [[0.0, 0.88], [1.0, 0.92]],
            },
        }
        model = TypeAdapter(Fluid_Zone_Base_Model).validate_python(data)
        assert isinstance(model, ActuatorDiscZone)

    def test_propellor_style_bet_zone(self):
        data = {
            "type": "disc",
            "name": "prop_1",
            "rotation_direction": "clockwise",
            "reference_plane": False,
            "controller": {"type": "fixed", "omega": 500.0, "pitch": 0.0},
            "geometry": {
                "centre": [0.0, 0.0, 0.0],
                "inner_radius": 0.05,
                "outer_radius": 0.5,
                "normal": [0.0, 0.0, 1.0],
                "up": [0.0, 1.0, 0.0],
            },
            "discretisation": {"type": "disc", "number_of_elements": 20},
            "model": {
                "type": "bet",  # lowercased upstream from 'BET'
                "kind": "propellor",
                "aerofoils": {"aerofoil1": {"cd": [[0, 0.01]], "cl": [[0, 1.0]]}},
                "aerofoil_positions": [[0.0, "aerofoil1"], [1.0, "aerofoil1"]],
                "blade_chord": [[0.2, 0.1]],
                "blade_twist": [[0.2, 28.0]],
                "number_of_sections": 24,
                "number_of_blades": 2,
                "tip_loss_correction": "rstar",
                "tip_loss_correction_radius": 0.8,
            },
        }
        model = TypeAdapter(Fluid_Zone_Base_Model).validate_python(data)
        assert isinstance(model, ActuatorDiscZone)
        assert model.model.kind == "propellor"
