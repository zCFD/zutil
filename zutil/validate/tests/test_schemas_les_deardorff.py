"""The Deardorff SGS model must be reachable through the v2 schema.

`les_model_types` carried only "none" and "wale" while the whole stable-boundary-layer capability
(GABLS1) depends on `deardorff` -- the Deardorff (1980) 1.5-order prognostic SGS-TKE closure. It was
added to the OLD src/python/zcfd_validate schema and missed when the schemas moved to
zutil/validate/schemas, so it was silently unreachable here and had no test at all.

Why it matters rather than being a cosmetic omission: WALE is stratification-BLIND. Its k equation
has no buoyancy term and its length scale is never limited by the buoyancy frequency, so it mixes a
stable layer as if it were neutral. Selecting it where `deardorff` was intended does not fail, it
quietly gives the wrong physics.
"""

import pytest
from pydantic import ValidationError

from zutil.validate import validate_zcfd_input
from zutil.validate.error_handling import ValidationErrorHandler
from zutil.validate.schemas.equations_schemas import LESModel, les_model_types


def _config(les_model):
    return {
        "config_version": 2,
        "solver": {
            "solver_settings": {"type": "incompressible"},
            "equations": {
                "type": "les",
                "turbulence": {"les": les_model},
            },
            "fluid_properties": {"material": "air"},
            "reference_conditions": {
                "ic_1": {
                    "temperature": 265.0,
                    "pressure": 101325.0,
                    "v": [8.0, 0.0, 0.0],
                    "reynolds_no": 100.0,
                },
            },
            "initialisation": {"initial_conditions": "ic_1"},
            "time_settings": {"type": "steady"},
            "convergence_control": {"cycles": 10},
            "numerical_scheme": {"order": "second"},
            "output_settings": {"solution": {}, "report": {"frequency": 10}},
        },
        "model": {
            "mesh1": {
                "mesh": "mesh1.h5",
                "boundary_conditions": {"bc_wall": {"type": "wall", "zones": [1]}},
            },
        },
    }


def _config_with_relax(les_model, relax):
    c = _config(les_model)
    c["solver"]["equations"]["turbulence"]["relax"] = relax
    return c


def _turbulence_dict(les_model):
    model = validate_zcfd_input(_config(les_model))
    return model.to_cpp_dict()["solver"]["equations"]["turbulence"]


class TestDeardorffReachable:
    def test_deardorff_is_in_the_vocabulary(self):
        """The literal itself, so a schema move cannot drop it silently again."""
        assert "deardorff" in les_model_types.__args__

    def test_deardorff_validates(self):
        assert _turbulence_dict("deardorff")["les"] == "deardorff"

    @pytest.mark.parametrize("les_model", ["none", "wale", "deardorff"])
    def test_every_vocabulary_member_round_trips_to_the_cpp_dict(self, les_model):
        """The C++ reads this key, so each spelling must survive to_cpp_dict unchanged."""
        assert _turbulence_dict(les_model)["les"] == les_model

    def test_turbulence_relax_is_always_emitted(self):
        """The C++ reads turbulence.relax UNCONDITIONALLY (IncompSolverData.h initKernels), so a
        deck that omits it must still produce it. It carried a 0.5 default in the legacy schema and
        was lost when the schemas moved, which left the deardorff path unable to supply it."""
        assert "relax" in LESModel.model_fields
        assert _turbulence_dict("deardorff")["relax"] == 0.5

    def test_turbulence_relax_round_trips_when_given(self):
        model = validate_zcfd_input(_config_with_relax("deardorff", 0.3))
        assert model.to_cpp_dict()["solver"]["equations"]["turbulence"]["relax"] == 0.3

    def test_turbulence_relax_range_is_enforced(self):
        with pytest.raises((ValidationError, ValidationErrorHandler)):
            validate_zcfd_input(_config_with_relax("deardorff", 2.0))

    def test_an_unknown_sgs_model_is_still_rejected(self):
        """Guard against widening the vocabulary to anything."""
        with pytest.raises((ValidationError, ValidationErrorHandler)) as excinfo:
            validate_zcfd_input(_config("smagorinsky"))
        assert "smagorinsky" in str(excinfo.value) or "les" in str(excinfo.value)
