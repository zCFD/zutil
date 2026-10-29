"""
Body-force and buoyancy keys on ``fluid_properties``: which solver accepts
each one, and the error an engineer sees when a deck sets one where it has
no effect.

Every consumer is on the incompressible solver. ``Reference::set`` keeps
gravity at zero on the compressible solver, Coriolis takes its vertical from
gravity and is skipped without it, and Boussinesq buoyancy acts only on the
solved temperature field. The Boussinesq reference density had no reader and
is no longer a key.
"""

import copy

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.fluid_model_schemas import (
    FluidModelBase,
    ViscousFluidModel,
)
from zutil.validate.tests.test_buoyancy_placement import (
    _incompressible_buoyancy_config,
)
from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive


def _compressible_config(fluid_properties):
    return {
        "config_version": 2,
        "solver": {
            "solver_settings": {"type": "compressible"},
            "equations": {"type": "euler"},
            "numerical_scheme": {"order": "second"},
            "fluid_properties": {"material": "ideal gas", **fluid_properties},
            "reference_conditions": {
                "IC_1": {"temperature": 300.0, "pressure": 101325.0},
            },
            "initialisation": {"initial_conditions": "IC_1"},
            "time_settings": {"type": "steady"},
            "convergence_control": {
                "cycles": 10,
                "cfl": 1.0,
                "scheme": {"name": "runge kutta"},
            },
            "output_settings": {
                "solution": {"frequency": 10},
                "report": {"frequency": 10},
            },
        },
        "model": {
            "base": {
                "mesh": "base.h5",
                "boundary_conditions": {
                    "bc1": {
                        "type": "farfield",
                        "kind": "riemann",
                        "condition": "IC_1",
                        "zones": [1],
                    },
                },
            }
        },
    }


def _incompressible_config(**fluid_overrides):
    config = _incompressible_buoyancy_config()
    fluid = config["solver"]["fluid_properties"]
    for key, value in fluid_overrides.items():
        if value is None:
            fluid.pop(key, None)
        else:
            fluid[key] = value
    return config


def _errors(config):
    _, nodes = validate_zcfd_comprehensive(copy.deepcopy(config))
    return [
        str(n.get("message"))
        for n in nodes
        if (n.get("severity") or "error") == "error"
    ]


class TestIncompressibleBuoyancyDeck:
    def test_full_buoyancy_and_coriolis_deck_is_valid(self):
        assert _errors(_incompressible_config()) == []

    def test_gravity_alone_is_valid(self):
        config = _incompressible_config(
            latitude=None, planet_rotation=None, thermal_expansion_coeff=None
        )
        assert _errors(config) == []


class TestGravityIsIncompressibleOnly:
    def test_compressible_deck_setting_gravity_is_rejected(self):
        errors = _errors(_compressible_config({"gravity": [0.0, 0.0, -9.81]}))
        assert len(errors) == 1
        assert "'gravity' only applies when" in errors[0]
        assert "'incompressible'" in errors[0]

    def test_compressible_deck_without_gravity_is_valid(self):
        assert _errors(_compressible_config({})) == []


class TestLatitudeNeedsGravity:
    def test_latitude_without_gravity_is_rejected(self):
        config = _incompressible_config(gravity=None, thermal_expansion_coeff=None)
        errors = _errors(config)
        assert len(errors) == 1
        assert "'latitude' only applies when" in errors[0]
        assert "gravity is set" in errors[0]

    def test_latitude_on_compressible_is_rejected(self):
        errors = _errors(_compressible_config({"latitude": 45.0}))
        assert len(errors) == 1
        assert "'latitude' only applies when" in errors[0]


class TestZeroGravityIsNoGravity:
    """A zero gravity vector is a legal way to switch gravity off, so the keys
    that need gravity treat it as absent: Coriolis has no vertical and the
    Boussinesq source is multiplied by it."""

    def test_zero_gravity_alone_is_valid(self):
        config = _incompressible_config(
            gravity=[0.0, 0.0, 0.0],
            latitude=None,
            planet_rotation=None,
            thermal_expansion_coeff=None,
        )
        assert _errors(config) == []

    def test_latitude_with_zero_gravity_is_rejected(self):
        config = _incompressible_config(
            gravity=[0.0, 0.0, 0.0], thermal_expansion_coeff=None
        )
        errors = _errors(config)
        assert len(errors) == 1
        assert "'latitude' only applies when" in errors[0]
        assert "gravity is set and non-zero" in errors[0]

    def test_thermal_expansion_coeff_with_zero_gravity_is_rejected(self):
        config = _incompressible_config(
            gravity=[0.0, 0.0, 0.0], latitude=None, planet_rotation=None
        )
        errors = _errors(config)
        assert len(errors) == 1
        assert "'thermal expansion coeff' only applies when" in errors[0]
        assert "gravity is set and non-zero" in errors[0]


class TestPlanetRotationNeedsLatitude:
    def test_planet_rotation_without_latitude_is_rejected(self):
        errors = _errors(_incompressible_config(latitude=None))
        assert len(errors) == 1
        assert "'planet rotation' only applies when" in errors[0]
        assert "latitude is set and non-zero" in errors[0]

    def test_planet_rotation_at_the_equator_is_rejected(self):
        errors = _errors(_incompressible_config(latitude=0.0))
        assert len(errors) == 1
        assert "'planet rotation' only applies when" in errors[0]


class TestGeostrophicWindNeedsLatitude:
    def test_geostrophic_wind_with_latitude_is_valid(self):
        config = _incompressible_config()
        config["solver"]["equations"]["geostrophic_wind"] = [10.0, 0.0, 0.0]
        assert _errors(config) == []

    def test_geostrophic_wind_without_latitude_is_rejected(self):
        config = _incompressible_config(latitude=None, planet_rotation=None)
        config["solver"]["equations"]["geostrophic_wind"] = [10.0, 0.0, 0.0]
        errors = _errors(config)
        assert len(errors) == 1
        assert "'geostrophic wind' only applies when" in errors[0]
        assert "latitude is set and non-zero" in errors[0]


class TestThermalExpansionCoeff:
    def test_rejected_without_solve_energy(self):
        config = _incompressible_config()
        config["solver"]["equations"]["solve_energy"] = False
        errors = _errors(config)
        assert len(errors) == 1
        assert "'thermal expansion coeff' only applies when" in errors[0]
        assert "solve energy is True" in errors[0]

    def test_rejected_on_compressible(self):
        errors = _errors(_compressible_config({"thermal_expansion_coeff": 0.003}))
        assert len(errors) == 1
        assert "'thermal expansion coeff' only applies when" in errors[0]
        assert "'incompressible'" in errors[0]


class TestReferenceDensityIsGone:
    @pytest.mark.parametrize(
        "key",
        [
            "boussinesq_buoyancy_reference_density",
            "bousinessq_buoyancy_reference_density",
        ],
    )
    def test_key_is_rejected_as_unknown(self, key):
        with pytest.raises(ValidationError):
            ViscousFluidModel.model_validate({key: 1.2})

    def test_compressible_deck_setting_it_fails(self):
        errors = _errors(
            _compressible_config({"boussinesq_buoyancy_reference_density": 1.225})
        )
        assert len(errors) == 1
        assert "boussinesq_buoyancy_reference_density" in errors[0]


class TestSchemaMetadata:
    @pytest.mark.parametrize(
        "field, units",
        [
            ("gravity", "m/s^2"),
            ("latitude", "deg"),
            ("planet_rotation", "rad/s"),
            ("thermal_expansion_coeff", "1/K"),
        ],
    )
    def test_units(self, field, units):
        schema = ViscousFluidModel.model_json_schema()
        assert schema["properties"][field]["zcfd"]["units"] == units

    def test_latitude_clause_names_non_zero_gravity(self):
        schema = FluidModelBase.model_json_schema()
        clause = schema["properties"]["latitude"]["zcfd"]["applies_when"]
        assert clause["solver.fluid_properties.gravity"] == {
            "not": [None, 0.0, [0.0, 0.0, 0.0]]
        }
        assert clause["solver.solver_settings.type"] == ["incompressible"]
