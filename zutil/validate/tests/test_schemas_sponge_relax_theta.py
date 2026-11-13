"""The sponge relaxes THETA as well as velocity, and that must be switchable.

EnergyEqn.cu applies the sponge to potential temperature whenever a sponge and a base state are both
present -- the deck's "target velocity" wording does not convey this, and it makes the upper boundary
a heat sink. `relax_theta` exposes it. The default is True so existing decks are unchanged; False
gives a velocity-only sponge, which is the single-variable way to test the sponge's effect on a
stratified case (sigma_max = 0 would also disable the velocity damping and let gravity waves reflect
off the lid, changing two things at once).
"""

import pytest
from pydantic import ValidationError

from zutil.validate import validate_zcfd_input
from zutil.validate.error_handling import ValidationErrorHandler
from zutil.validate.schemas.equations_schemas import SpongeLayer


def _config(sponge):
    return {
        "config version": 2,
        "solver": {
            "solver settings": {"type": "incompressible"},
            "equations": {
                "type": "les",
                "solve energy": True,
                "energy variable": "potential temperature",
                "base state": {
                    "kind": "constant brunt vaisala",
                    "brunt vaisala frequency": 0.01,
                },
                "turbulence": {"les": "deardorff"},
                "sponge layer": sponge,
            },
            "fluid properties": {"material": "air", "gravity": [0.0, -9.80665, 0.0]},
            "reference conditions": {
                "IC_1": {
                    "temperature": 265.0,
                    "pressure": 101325.0,
                    "v": [8.0, 0.0, 0.0],
                    "reynolds no": 1.0e6,
                },
            },
            "initialisation": {"reference": "IC_1", "initial conditions": "IC_1"},
            "time settings": {"type": "steady"},
            "convergence control": {"cycles": 10},
            "numerical scheme": {"order": "second"},
            "output settings": {"solution": {}, "report": {"frequency": 10}},
        },
        "model": {
            "m1": {
                "mesh": "m.h5",
                "boundary conditions": {"BC_1": {"type": "wall", "zones": [1]}},
            },
        },
    }


def _sponge_dict(sponge):
    model = validate_zcfd_input(_config(sponge))
    return model.to_cpp_dict()["solver"]["equations"]["sponge_layer"]


BASE = {"sigma max": 0.05, "start height": 300.0, "target velocity": [8.0, 0.0, 0.0]}


class TestSpongeRelaxTheta:
    def test_defaults_true_so_existing_decks_are_unchanged(self):
        assert SpongeLayer.model_fields["relax_theta"].default is True
        assert _sponge_dict(dict(BASE))["relax_theta"] is True

    def test_can_be_disabled_for_a_velocity_only_sponge(self):
        assert _sponge_dict({**BASE, "relax theta": False})["relax_theta"] is False

    def test_survives_to_the_cpp_dict_under_the_snake_case_name(self):
        """The C++ reads relax_theta; the deck spells it with a space."""
        d = _sponge_dict({**BASE, "relax theta": False})
        assert "relax_theta" in d and "relax theta" not in d

    def test_rejects_a_non_boolean(self):
        with pytest.raises((ValidationError, ValidationErrorHandler)):
            validate_zcfd_input(_config({**BASE, "relax theta": "sometimes"}))
