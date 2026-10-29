"""Blocks whose version is decided elsewhere in the deck.

Each case is checked three ways, because engineers reach the schema three
ways: the solver's check (validate_zcfd_comprehensive), a script that builds
the model directly, and a JSON round trip from zui.
"""

import copy
import json

import pytest
from pydantic import ValidationError

from zutil.validate.error_mapping import format_errors_as_text
from zutil.validate.schemas import fluid_model_schemas as fld
from zutil.validate.schemas import block_versions as versions
from zutil.validate.schemas import numerical_schemes_schemas as num
from zutil.validate.schemas.variants import Variants
from zutil.validate.schemas.model_schemas import ModelSettings
from zutil.validate.tests.test_buoyancy_placement import _incompressible_buoyancy_config
from zutil.validate.tests.test_core_orchestration import _minimal_valid_config
from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive
from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _report(deck):
    _, nodes = validate_zcfd_comprehensive(copy.deepcopy(deck))
    return format_errors_as_text(nodes, deck) if nodes else ""


def _direct_errors(deck):
    out = {}
    for path, build in (
        ("Model(**d)", lambda d: ZCFDInputModelBase(**d)),
        (
            "model_validate_json",
            lambda d: ZCFDInputModelBase.model_validate_json(json.dumps(d)),
        ),
    ):
        try:
            build(copy.deepcopy(deck))
            out[path] = []
        except ValidationError as e:
            out[path] = [(e["loc"], e["msg"]) for e in e.errors()]
    return out


def _mesh_scheme(order):
    deck = _minimal_valid_config()
    deck["model"]["base"]["numerical_scheme"] = {"order": order}
    return deck


class TestMeshOverrideFollowsTheSolver:
    def test_central_on_a_compressible_deck_is_rejected_everywhere(self):
        deck = _mesh_scheme("central")
        assert "'central' is only available when" in _report(deck)
        assert "solver settings › type is 'incompressible'" in _report(deck)
        for errors in _direct_errors(deck).values():
            assert [loc for loc, _ in errors] == [
                ("model", "base", "numerical_scheme", "order")
            ]

    def test_valid_override_is_held_as_the_compressible_scheme_everywhere(self):
        deck = _mesh_scheme("second")
        assert _report(deck) == ""
        for build in (
            lambda d: ZCFDInputModelBase.model_validate(d),
            lambda d: ZCFDInputModelBase(**d),
            lambda d: ZCFDInputModelBase.model_validate_json(json.dumps(d)),
        ):
            scheme = build(copy.deepcopy(deck)).model["base"].numerical_scheme
            assert isinstance(scheme, num.FVCompressibleEulerNumericalScheme)

    def test_a_wrong_value_gets_one_error_with_this_solvers_options(self):
        deck = _mesh_scheme("bogus")
        for errors in _direct_errors(deck).values():
            assert len(errors) == 1
        assert "'central'" not in _report(deck)

    def test_central_on_an_incompressible_deck_is_accepted(self):
        deck = _incompressible_buoyancy_config()
        mesh = next(iter(deck["model"]))
        deck["model"][mesh]["numerical_scheme"] = {"order": "central"}
        assert _report(deck) == ""


class TestDecidingSettingTypo:
    def test_only_the_misspelt_setting_is_reported(self):
        deck = _minimal_valid_config()
        deck["solver"]["equations"]["type"] = "rnas"
        for errors in _direct_errors(deck).values():
            assert {loc[:2] for loc, _ in errors} == {("solver", "equations")}


class TestSettingFromAnotherEquationSet:
    def test_turbulence_setting_in_a_viscous_deck_names_the_equations_that_use_it(self):
        deck = _minimal_valid_config()
        deck["solver"]["equations"]["type"] = "viscous"
        condition = deck["solver"]["reference_conditions"]["IC_1"]
        condition.update(viscosity=1.8e-5, turbulence_intensity=0.01)
        assert (
            "'turbulence_intensity' is only used when solver › equations › type is 'rans' or 'les'"
            in _report(deck)
        )
        for errors in _direct_errors(deck).values():
            assert [loc for loc, _ in errors] == [
                ("solver", "reference_conditions", "IC_1", "turbulence_intensity")
            ]


class TestIncompressibleGlobalTimeStepping:
    def test_the_reason_is_given_before_any_missing_field(self):
        deck = _incompressible_buoyancy_config()
        deck["solver"]["time_settings"] = {
            "type": "unsteady",
            "kind": "global timestepping",
            "total_time": 1.0,
            "time_step": 0.01,
        }
        report = _report(deck)
        assert "cannot run under global time stepping" in report
        assert "Scheme" not in report


class TestBlockOnItsOwn:
    def test_a_mesh_block_with_a_scheme_override_needs_its_deck(self):
        with pytest.raises(
            ValidationError, match="only be checked as part of a whole deck"
        ):
            ModelSettings.model_validate(
                {
                    "mesh": "base.h5",
                    "boundary_conditions": {},
                    "numerical_scheme": {"order": "second"},
                }
            )


def _tables():
    return {
        name: table
        for name, table in vars(versions).items()
        if name.isupper() and isinstance(table, list)
    }


def _combinations():
    return [
        (when[versions.SOLVER], when[versions.EQUATIONS])
        for when, _ in versions.EQUATIONS_BLOCK
    ]


class TestEveryCaseHasAVersion:
    @pytest.mark.parametrize("solver, equations", _combinations())
    def test_every_table_has_a_version_for_every_solver_and_equations(
        self, solver, equations
    ):
        deck = {
            "solver": {
                "equations": {"type": equations},
                "solver_settings": {"type": solver},
            }
        }
        for name, table in _tables().items():
            deciding = {p for when, _ in table for p in when}
            if deciding <= {versions.SOLVER, versions.EQUATIONS}:
                assert Variants(table).version_for(deck) is not None, name

    def test_incompressible_fluid_properties_use_the_incompressible_model(self):
        deck = {
            "solver": {
                "equations": {"type": "rans"},
                "solver_settings": {"type": "incompressible"},
            }
        }
        assert (
            Variants(versions.FLUID_PROPERTIES).version_for(deck)
            is fld.IncompressibleFluidModel
        )
