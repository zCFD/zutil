"""
Interpolated-restart requirements.

An interpolated restart needs 'restart': True, a named restart case and
initial conditions for the variables the restart case does not solve. The
restart case records its own mesh, so 'restart meshname' is optional, and a
trailing '.h5' is stripped from it.
"""

import pytest
from pydantic import ValidationError

from zutil.validate import validate_zcfd_comprehensive
from zutil.validate.schemas.initialisation_schemas import InitialisationModel
from zutil.validate.tests.test_core_orchestration import _minimal_valid_config
from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _deck(initialisation):
    deck = _minimal_valid_config()
    deck["solver"]["initialisation"] = initialisation
    return deck


def _interpolated(**overrides):
    settings = dict(
        restart=True,
        interpolate_restart=True,
        restart_casename="case1",
        initial_conditions="IC_1",
    )
    settings.update(overrides)
    return InitialisationModel(**settings)


class TestInterpolateRestartRequirements:
    def test_casename_and_initial_conditions_suffice(self):
        model = _interpolated()
        assert model.restart_meshname is None
        assert model.interpolation == "linear"
        assert model.interpolate_outside_limit == 0.001

    def test_without_casename_raises(self):
        with pytest.raises(ValidationError, match="restart_casename"):
            _interpolated(restart_casename=None)

    def test_without_restart_raises(self):
        with pytest.raises(ValidationError, match="'restart': True"):
            _interpolated(restart=False)

    def test_without_initial_conditions_raises(self):
        with pytest.raises(ValidationError, match="initial_conditions"):
            _interpolated(initial_conditions=None)

    def test_plain_restart_needs_no_initial_conditions(self):
        model = InitialisationModel(restart=True)
        assert model.initial_conditions is None

    def test_injection_accepted(self):
        assert _interpolated(interpolation="injection").interpolation == "injection"

    def test_unknown_interpolation_rejected(self):
        with pytest.raises(ValidationError):
            _interpolated(interpolation="cubic")

    def test_outside_limit_is_a_fraction(self):
        with pytest.raises(ValidationError):
            _interpolated(interpolate_outside_limit=1.5)

    def test_restart_models_maps_names(self):
        model = _interpolated(restart_models={"wing_fine": "wing"})
        assert model.restart_models == {"wing_fine": "wing"}


class TestRestartMeshnameH5Stripping:
    def test_trailing_h5_suffix_stripped(self):
        assert _interpolated(restart_meshname="mesh1.h5").restart_meshname == "mesh1"

    def test_no_suffix_left_unchanged(self):
        assert _interpolated(restart_meshname="mesh1").restart_meshname == "mesh1"

    def test_h5_stripped_even_without_interpolate_restart(self):
        """The suffix strip applies to the field itself, independent of interpolate_restart."""
        model = InitialisationModel(restart=True, restart_meshname="other.h5")
        assert model.restart_meshname == "other"


class TestRestartNeedsAReference:
    """A restart takes its reference state from 'reference' or, failing that,
    'initial conditions'; the solver aborts when it has neither."""

    def test_restart_with_neither_rejected(self):
        with pytest.raises(ValidationError) as info:
            ZCFDInputModelBase.model_validate(_deck({"restart": True}))
        assert (
            "Set 'reference' or 'initial conditions'; one is required when "
            "restart is True."
        ) in str(info.value)

    @pytest.mark.parametrize(
        "given", [{"reference": "IC_1"}, {"initial_conditions": "IC_1"}]
    )
    def test_restart_with_either_accepted(self, given):
        model = ZCFDInputModelBase.model_validate(_deck({"restart": True, **given}))
        assert model.solver.initialisation.restart is True


RESTART_ONLY = {
    "restart_casename": "case1",
    "restart_meshname": "mesh1",
    "restart_ignore_history": True,
    "interpolation": "injection",
    "restart_models": {"wing_fine": "wing"},
    "interpolate_outside_limit": 0.01,
}


class TestRestartOnlySettings:
    """A restart setting applies only while 'restart' is True."""

    @pytest.mark.parametrize("key", sorted(RESTART_ONLY) + ["interpolate_restart"])
    def test_declares_that_it_applies_only_on_restart(self, key):
        node = InitialisationModel.model_json_schema()["properties"][key]
        assert node["zcfd"]["applies_when"] == {"<self>.restart": [True]}

    @pytest.mark.parametrize("key", sorted(RESTART_ONLY))
    def test_set_without_restart_is_an_error_on_that_setting(self, key):
        deck = _deck({"initial_conditions": "IC_1", key: RESTART_ONLY[key]})
        _, nodes = validate_zcfd_comprehensive(deck)
        errors = [n for n in nodes if n.get("severity", "error") == "error"]
        assert [n["field_path"] for n in errors] == [f"solver.initialisation.{key}"]

    @pytest.mark.parametrize("key", sorted(RESTART_ONLY))
    def test_left_at_its_default_without_restart_is_accepted(self, key):
        default = InitialisationModel.model_fields[key].default
        deck = _deck({"initial_conditions": "IC_1", key: default})
        model, _ = validate_zcfd_comprehensive(deck)
        assert model is not None
