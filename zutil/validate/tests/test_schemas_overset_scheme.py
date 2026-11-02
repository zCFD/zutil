"""
Overset cases must use an implicit pseudo-time scheme.

The overset interface is a Dirichlet boundary whose value is interpolated from
another mesh by ``_exchange_mapping_data`` once per pseudo-time cycle
(``SimulationContext._execute_inner_step``), not once per Runge-Kutta stage.
An explicit multi-stage march therefore advances the interior several times
against interface data frozen for the whole cycle.

Whether a case is overset is a boundary-condition fact and the scheme is a
solver (or per-mesh ``convergence_control``) fact, so no single-block
applies_when clause can express the rule -- it is a hand-written cross-block
validator on ``ZCFDInputModelBase``.
"""

import pytest
from pydantic import ValidationError

from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _config(scheme, *, overset=True, mesh_scheme=None):
    """Two-mesh compressible config; 'collar' oversets onto 'base'."""
    collar_bc = (
        {
            "zones": [10],
            "type": "interface",
            "kind": "overset",
            "connection": {"model": ["base"]},
        }
        if overset
        else {"zones": [10], "type": "symmetry"}
    )
    collar = {
        "mesh": "collar.h5",
        "boundary_conditions": {"bc1": collar_bc},
    }
    if mesh_scheme is not None:
        collar["convergence_control"] = {"cycles": 5, "scheme": mesh_scheme}

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
            "convergence_control": {"cycles": 10, "scheme": scheme},
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
            "collar": collar,
        },
    }


class TestOversetRequiresImplicitScheme:
    @pytest.mark.parametrize(
        "scheme",
        [{"name": "implicit euler", "stage": 1}, {"name": "lu-sgs"}],
    )
    def test_implicit_schemes_are_accepted(self, scheme):
        ZCFDInputModelBase.model_validate(_config(scheme))

    @pytest.mark.parametrize(
        "scheme",
        [{"name": "runge kutta", "stage": 5}, {"name": "euler"}],
    )
    def test_explicit_schemes_are_rejected(self, scheme):
        with pytest.raises((ValidationError, Exception)) as excinfo:
            ZCFDInputModelBase.model_validate(_config(scheme))
        assert "explicit pseudo-time scheme" in str(excinfo.value)

    def test_the_message_names_the_meshes_that_are_overset(self):
        with pytest.raises(Exception) as excinfo:
            ZCFDInputModelBase.model_validate(
                _config({"name": "runge kutta", "stage": 5})
            )
        assert "collar" in str(excinfo.value)

    def test_explicit_is_still_fine_without_an_overset_boundary(self):
        ZCFDInputModelBase.model_validate(
            _config({"name": "runge kutta", "stage": 5}, overset=False)
        )

    def test_a_per_mesh_override_cannot_smuggle_an_explicit_scheme_in(self):
        """The solver default is implicit; the collar overrides it to explicit."""
        with pytest.raises(Exception) as excinfo:
            ZCFDInputModelBase.model_validate(
                _config(
                    {"name": "implicit euler", "stage": 1},
                    mesh_scheme={"name": "runge kutta", "stage": 5},
                )
            )
        assert "explicit pseudo-time scheme" in str(excinfo.value)
