"""
BaseState surface-state resolution and profile-parameter mutex.

Surface temperature/pressure come from a NAMED reference condition --
``base_state.reference``, falling back to ``initialisation.reference`` and
then ``initialisation.initial_conditions`` -- so the surface state is written
once per deck instead of duplicated between the IC and the base state
(IncompressibleSolverBlockModel.resolve_base_state_surface). Explicit values
override. The profile parameter belongs to its
kind: BaseState is a union discriminated on ``kind``, so the wrong-kind
parameter is rejected as an unknown field. The C++
(Coupled.cxx initialiseBaseState) reads every base_state key unconditionally,
so both validators must leave the emitted dict fully populated.
"""

import pytest
from pydantic import ValidationError

from zutil.validate import validate_zcfd_input
from zutil.validate.error_handling import ValidationErrorHandler


def _config(base_state, initialisation=None):
    return {
        "config_version": 2,
        "solver": {
            "solver_settings": {"type": "incompressible"},
            "equations": {
                "type": "viscous",
                "solve_energy": True,
                "energy_variable": "potential temperature",
                "base_state": base_state,
            },
            "fluid_properties": {
                "material": "air",
                "gravity": [0.0, 0.0, -9.81],
            },
            "reference_conditions": {
                "ic_1": {
                    "temperature": 300.0,
                    "pressure": 101325.0,
                    "v": [1.0, 0.0, 0.0],
                    "reynolds_no": 100.0,
                },
                "ic_2": {
                    "temperature": 250.0,
                    "pressure": 90000.0,
                    "v": [0.0, 0.0, 0.0],
                    "reynolds_no": 100.0,
                },
            },
            "initialisation": initialisation or {"initial_conditions": "ic_1"},
            "time_settings": {"type": "steady"},
            "convergence_control": {"cycles": 10},
            "numerical_scheme": {"order": "second"},
            "output_settings": {"solution": {}, "report": {"frequency": 10}},
        },
        "model": {
            "mesh1": {
                "mesh": "mesh1.h5",
                "boundary_conditions": {
                    "bc_wall": {"type": "wall", "zones": [1]},
                },
            },
        },
    }


def _base_state_dict(base_state, initialisation=None):
    model = validate_zcfd_input(_config(base_state, initialisation))
    return model.to_cpp_dict()["solver"]["equations"]["base_state"]


def _rejection(base_state, initialisation=None):
    with pytest.raises((ValidationError, ValidationErrorHandler)) as excinfo:
        validate_zcfd_input(_config(base_state, initialisation))
    return str(excinfo.value)


class TestSurfaceResolution:
    def test_falls_back_to_the_initial_conditions_reference(self):
        bs = _base_state_dict({"kind": "constant lapse rate"})
        assert bs["surface_temperature"] == 300.0
        assert bs["surface_pressure"] == 101325.0

    def test_initialisation_reference_wins_over_initial_conditions(self):
        bs = _base_state_dict(
            {"kind": "constant lapse rate"},
            {"reference": "ic_2", "initial_conditions": "ic_1"},
        )
        assert bs["surface_temperature"] == 250.0
        assert bs["surface_pressure"] == 90000.0

    def test_named_reference_wins_over_everything(self):
        bs = _base_state_dict(
            {"kind": "constant lapse rate", "reference": "ic_2"},
            {"reference": "ic_1", "initial_conditions": "ic_1"},
        )
        assert bs["surface_temperature"] == 250.0
        assert bs["surface_pressure"] == 90000.0

    def test_explicit_surface_values_override_the_reference(self):
        bs = _base_state_dict(
            {
                "kind": "constant lapse rate",
                "surface_temperature": 288.15,
                "surface_pressure": 100000.0,
            }
        )
        assert bs["surface_temperature"] == 288.15
        assert bs["surface_pressure"] == 100000.0

    def test_partial_override_fills_the_rest_from_the_reference(self):
        bs = _base_state_dict(
            {"kind": "constant lapse rate", "surface_temperature": 288.15}
        )
        assert bs["surface_temperature"] == 288.15
        assert bs["surface_pressure"] == 101325.0

    def test_unknown_reference_name_rejected(self):
        assert "ic_9" in _rejection(
            {"kind": "constant lapse rate", "reference": "ic_9"}
        )

    def test_emitted_dict_always_carries_both_surface_keys(self):
        bs = _base_state_dict({"kind": "constant lapse rate"})
        assert "surface_temperature" in bs and "surface_pressure" in bs


class TestProfileParameterMutex:
    def test_brunt_rejected_for_a_lapse_rate_profile(self):
        msg = _rejection(
            {"kind": "constant lapse rate", "brunt_vaisala_frequency": 0.02}
        )
        assert "brunt vaisala frequency" in msg

    def test_lapse_rate_rejected_for_a_brunt_profile(self):
        msg = _rejection({"kind": "constant brunt vaisala", "lapse_rate": 0.0098})
        assert "lapse rate" in msg

    def test_both_rejected_for_isothermal(self):
        assert "lapse rate" in _rejection({"kind": "isothermal", "lapse_rate": 0.0065})
        assert "brunt vaisala frequency" in _rejection(
            {"kind": "isothermal", "brunt_vaisala_frequency": 0.01}
        )

    def test_matching_parameter_accepted_and_the_other_absent(self):
        bs = _base_state_dict(
            {"kind": "constant brunt vaisala", "brunt_vaisala_frequency": 0.02}
        )
        assert bs["brunt_vaisala_frequency"] == 0.02
        # Discriminated union: the other kinds' parameters do not exist on this
        # variant, and the C++ reads the profile parameter by kind.
        assert "lapse_rate" not in bs

    def test_missing_kind_rejected(self):
        assert "kind" in _rejection({}).lower()


class TestBaseStateNeedsGravity:
    """initialiseBaseState integrates the hydrostatic profile along gravity
    and throws on a zero vector, so validation rejects the deck first."""

    @pytest.mark.parametrize("gravity", [None, [0.0, 0.0, 0.0]])
    def test_rejected_without_non_zero_gravity(self, gravity):
        config = _config({"kind": "isothermal"})
        fluid = config["solver"]["fluid_properties"]
        if gravity is None:
            del fluid["gravity"]
        else:
            fluid["gravity"] = gravity
        with pytest.raises((ValidationError, ValidationErrorHandler)) as excinfo:
            validate_zcfd_input(config)
        assert (
            "'base state' only applies when "
            "solver.fluid properties.gravity is set and non-zero"
        ) in str(excinfo.value)

    def test_accepted_with_gravity(self):
        assert _base_state_dict({"kind": "isothermal"})["kind"] == "isothermal"
