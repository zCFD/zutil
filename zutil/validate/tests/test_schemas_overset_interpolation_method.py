"""
The overset interpolation method selects the donor reconstruction: the nearest
cell's value, an inverse-distance average, a Green-Gauss reconstruction or a
least-squares reconstruction. Anything else has no implementation behind it, so
the schema restricts the field to those names.
"""

import pytest
from pydantic import ValidationError

from zutil.validate.error_handling import ValidationErrorHandler
from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _config(method):
    """Two-mesh compressible overset config with a given interpolation method."""
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
            "time_settings": {"type": "steady"},
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
            "base": {
                "mesh": "base.h5",
                "boundary_conditions": {
                    "bc1": {"zones": [1], "type": "wall", "kind": "no slip"},
                },
            },
            "collar": {
                "mesh": "collar.h5",
                "boundary_conditions": {
                    "bc1": {
                        "zones": [10],
                        "type": "interface",
                        "kind": "overset",
                        "connection": {"model": ["base"]},
                        "interpolation_method": method,
                    },
                },
            },
        },
    }


class TestOversetInterpolationMethod:
    @pytest.mark.parametrize(
        "method",
        ["nearest cell", "inverse distance", "green-gauss", "least-squares"],
    )
    def test_implemented_methods_are_accepted(self, method):
        ZCFDInputModelBase.model_validate(_config(method))

    @pytest.mark.parametrize("method", ["cubic", "idw", "least squares", ""])
    def test_an_unknown_method_is_rejected(self, method):
        with pytest.raises((ValidationError, ValidationErrorHandler)):
            ZCFDInputModelBase.model_validate(_config(method))
