"""Output-variable names are checked against what the solver will accept.

`_validate_output_variables` runs on the cleaned input dict, before Pydantic
builds any model, so every test here drives it through the same public entry
point a deck goes through. It used to read `output_settings.solution` as an
attribute, which a dict never has, so no deck's variable list was checked at
all and a misspelled name reached the solver — which logs a warning, drops the
variable and writes the file without it.
"""

import pytest

from zutil.validate.schemas.output_schemas import (
    _closest_variable,
    _validate_output_variables,
)
from zutil.validate.schemas.output_variables import (
    get_valid_variables_for_equation_type,
)
from zutil.validate.validate_input import validate_zcfd_input


def _deck(surface_variables=None, volume_variables=None, **solver_overrides):
    solution = {"format": "vtk"}
    if surface_variables is not None:
        solution["surface variables"] = surface_variables
    if volume_variables is not None:
        solution["volume variables"] = volume_variables

    solver = {
        "equations": {"type": "viscous"},
        "solver settings": {"type": "compressible"},
        "fluid properties": {"material": "air"},
        "reference conditions": {
            "IC_1": {
                "temperature": 288.15,
                "pressure": 101325.0,
                "v": {"vector": [1.0, 0.0, 0.0], "mach": 0.2},
                "viscosity": 1.81e-05,
            }
        },
        "initialisation": {"initial conditions": "IC_1"},
        "numerical scheme": {"order": "second", "limiter": "vanalbada"},
        "time settings": {"type": "steady"},
        "convergence control": {"cycles": 10, "cfl": {"cfl": 2.0}},
        "output settings": {"solution": solution, "report": {"frequency": 10}},
    }
    solver.update(solver_overrides)
    return {
        "config version": 2,
        "solver": solver,
        "model": {
            "block": {
                "mesh": "block.h5",
                "boundary conditions": {
                    "BC_1": {
                        "zones": [1],
                        "type": "farfield",
                        "kind": "riemann",
                        "condition": "IC_1",
                    }
                },
            }
        },
    }


class TestADeckIsActuallyChecked:
    """The regression: validation reached the deck's variable lists at all."""

    def test_a_valid_deck_passes(self):
        validate_zcfd_input(_deck(surface_variables=["p", "T", "cp"]))

    def test_an_unknown_surface_variable_is_rejected(self):
        with pytest.raises(Exception, match="not_a_variable"):
            validate_zcfd_input(_deck(surface_variables=["p", "not_a_variable"]))

    def test_an_unknown_volume_variable_is_rejected(self):
        with pytest.raises(Exception, match="not_a_variable"):
            validate_zcfd_input(_deck(volume_variables=["p", "not_a_variable"]))

    def test_a_variable_the_equation_type_does_not_offer_is_rejected(self):
        deck = _deck(volume_variables=["lesregion"])
        deck["solver"]["equations"] = {"type": "viscous"}
        with pytest.raises(Exception, match="lesregion"):
            validate_zcfd_input(deck)

    def test_the_dict_the_early_check_is_given_is_validated(self):
        """The shape the production caller passes, checked directly.

        `_validate_output_variables` is called with plain dicts from
        `ZCFDInputModelBase.model_validate`; an attribute-only read here
        validates nothing and raises nothing.
        """
        output_settings = {
            "solution": {"surface_variables": ["not_a_variable"]},
            "report": {"frequency": 10},
        }
        with pytest.raises(ValueError, match="not_a_variable"):
            _validate_output_variables(output_settings, "viscous")


class TestMisspellingsAreNamed:
    """A wrong spelling costs the whole variable, so say what was meant."""

    def test_spaces_for_underscores_are_suggested(self):
        with pytest.raises(ValueError, match="gauge_pressure"):
            _validate_output_variables(
                {"solution": {"surface_variables": ["gauge pressure"]}},
                "viscous",
            )

    def test_the_wrong_case_is_suggested(self):
        with pytest.raises(ValueError, match="lesregion"):
            _validate_output_variables(
                {"solution": {"volume_variables": ["LESRegion"]}}, "rans"
            )

    def test_a_near_miss_is_suggested(self):
        with pytest.raises(ValueError) as raised:
            _validate_output_variables(
                {"solution": {"volume_variables": ["walldistance"]}}, "rans"
            )
        assert "Did you mean 'walldistance' -> 'walldist'?" in str(raised.value)

    @pytest.mark.parametrize(
        "written, meant",
        [
            ("walldistance", "walldist"),
            ("walldistancezon", "walldistancezone"),
            ("pressuregra", "pressuregrad"),
            ("presure", "pressure"),
            ("xyzzy", None),
        ],
    )
    def test_the_nearest_name_is_the_one_suggested(self, written, meant):
        """Fewest edits wins; between equally near names, the one the written
        name spells out beats the one that extends it."""
        valid = get_valid_variables_for_equation_type("rans", "volume")
        assert _closest_variable(written, valid) == meant


class TestTurbulenceModelDependentVariables:
    def test_the_transition_fields_need_the_transition_model(self):
        deck = _deck(volume_variables=["turbulenceintermittency"])
        deck["solver"]["equations"] = {
            "type": "rans",
            "turbulence": {"model": "sst"},
        }
        with pytest.raises(Exception, match="turbulenceintermittency"):
            validate_zcfd_input(deck)

    def test_the_transition_model_offers_them(self):
        deck = _deck(volume_variables=["turbulenceintermittency"])
        deck["solver"]["equations"] = {
            "type": "rans",
            "turbulence": {"model": "sst-transition"},
        }
        validate_zcfd_input(deck)

    def test_the_equation_count_follows_the_turbulence_model(self):
        """sa-neg solves six equations, so var_7 is not one of them."""
        deck = _deck(surface_variables=["var_7"])
        deck["solver"]["equations"] = {
            "type": "rans",
            "turbulence": {"model": "sa-neg"},
        }
        with pytest.raises(Exception, match="var_7"):
            validate_zcfd_input(deck)

        deck["solver"]["equations"]["turbulence"]["model"] = "sst"
        validate_zcfd_input(deck)


class TestFSIGating:
    def test_modal_fields_are_offered_when_a_mesh_declares_an_fsi_zone(self):
        """FSI zones live under `model.<mesh>`, not at the top level."""
        deck = _deck(surface_variables=["modaldeformation"])
        with pytest.raises(Exception, match="modaldeformation"):
            validate_zcfd_input(deck)

        deck["model"]["block"]["boundary conditions"]["BC_2"] = {
            "zones": [2],
            "type": "wall",
            "kind": "slip",
            "fsi": {"type": "generic_fsi", "zone": [2], "transform type": "idw"},
        }
        validate_zcfd_input(deck)

    def _framed(self, **solution):
        deck = _deck(**solution)
        deck["model"]["block"]["transforms"] = {
            "motion": {
                "type": "translating",
                "velocity": [10.0, 0.0, 0.0],
            }
        }
        return deck

    def test_a_frame_offers_mesh_velocity(self):
        validate_zcfd_input(self._framed(volume_variables=["meshvelocity"]))
        validate_zcfd_input(self._framed(surface_variables=["meshvelocity"]))

    @pytest.mark.parametrize(
        "name", ["modaldeformation", "modeshapedisplacement_1", "modeshaperotation_1"]
    )
    def test_a_frame_without_fsi_refuses_the_modal_fields(self, name):
        """Only a modal FSI model writes these; with none the solver writes an unset buffer."""
        with pytest.raises(Exception, match=name):
            validate_zcfd_input(self._framed(surface_variables=[name]))

    def test_mesh_velocity_is_refused_on_a_still_mesh(self):
        with pytest.raises(Exception, match="meshvelocity"):
            validate_zcfd_input(_deck(volume_variables=["meshvelocity"]))


class TestValidSetCoversWhatTheSolverWrites:
    @pytest.mark.parametrize("name", ["mu", "nu", "viscosity"])
    def test_viscosity_is_offered_in_both_contexts(self, name):
        for context in ("volume", "surface"):
            assert name in get_valid_variables_for_equation_type("viscous", context)

    @pytest.mark.parametrize(
        "name", ["V_avg", "V_rms", "p_avg", "p_rms", "T_avg", "T_rms"]
    )
    def test_the_running_average_fields_are_offered(self, name):
        assert name in get_valid_variables_for_equation_type("viscous", "volume")
