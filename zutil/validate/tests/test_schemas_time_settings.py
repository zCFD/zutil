"""
Accept/reject coverage for the time_settings version table: a deck's
time_settings is checked as Steady, GTS or DTS according to its 'type'
and 'kind'.

Also covers GTSTimeSettings._fill_default_scheme_name's actual default-fill
behaviour — test_convergence_schemes.py's TestGTSSchemeNarrowing covers the
narrowing to the three-stage TVD Runge-Kutta, this file covers the
omitted-name path for this class specifically (the sibling
ConvergenceControlBase copy was covered).
"""

import pytest

from pydantic import ValidationError

from zutil.validate.schemas.time_settings_schemas import (
    DTSTimeSettings,
    GTSTimeSettings,
    SteadyTimeSettings,
)
from zutil.validate.zcfd_schema import ZCFDInputModelBase

from zutil.validate.tests.test_core_orchestration import _minimal_valid_config


def _resolve_time_settings(time_settings):
    deck = _minimal_valid_config()
    deck["solver"]["time_settings"] = time_settings
    if time_settings.get("kind") == "global timestepping":
        deck["solver"].pop("convergence_control")
    try:
        return ZCFDInputModelBase.__pydantic_validator__.validate_python(
            deck
        ).solver.time_settings
    except ValidationError as e:
        raise ValueError(" | ".join(err["msg"] for err in e.errors())) from None


class TestResolveTimeSettingsRouting:
    def test_steady_routes_to_steady_model(self):
        result = _resolve_time_settings({"type": "steady"})
        assert isinstance(result, SteadyTimeSettings)

    def test_gts_routes_to_gts_model(self):
        result = _resolve_time_settings(
            {
                "type": "unsteady",
                "kind": "global timestepping",
                "total_time": 1.0,
                "time_step": 0.01,
                "scheme": {"name": "runge kutta"},
            }
        )
        assert isinstance(result, GTSTimeSettings)

    def test_dts_routes_to_dts_model(self):
        result = _resolve_time_settings(
            {
                "type": "unsteady",
                "kind": "dual time stepping",
                "total_time": 1.0,
                "time_step": 0.01,
            }
        )
        assert isinstance(result, DTSTimeSettings)

    def test_missing_type_is_steady(self):
        assert isinstance(_resolve_time_settings({}), SteadyTimeSettings)

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="Input should be 'steady', 'unsteady'"):
            _resolve_time_settings({"type": "quasi-steady"})

    def test_unsteady_missing_kind_rejected(self):
        with pytest.raises(
            ValueError,
            match="Input should be 'global timestepping', 'dual time stepping'",
        ):
            _resolve_time_settings({"type": "unsteady"})

    def test_unsteady_unknown_kind_rejected(self):
        with pytest.raises(
            ValueError,
            match="Input should be 'global timestepping', 'dual time stepping'",
        ):
            _resolve_time_settings({"type": "unsteady", "kind": "sideways"})


class TestGTSDefaultSchemeName:
    """GTSTimeSettings._fill_default_scheme_name: a scheme dict without
    'name' defaults to 'runge kutta', matching ConvergenceControlBase's
    identical default-fill behaviour."""

    _BASE = {
        "kind": "global timestepping",
        "total_time": 0.2,
        "time_step": 1.0e-5,
    }

    def test_empty_scheme_dict_defaults_to_runge_kutta(self):
        gts = GTSTimeSettings.model_validate({**self._BASE, "scheme": {}})
        assert gts.scheme.name == "runge kutta"

    def test_scheme_dict_without_name_defaults_to_runge_kutta(self):
        gts = GTSTimeSettings.model_validate({**self._BASE, "scheme": {"stage": 3}})
        assert gts.scheme.name == "runge kutta"
        assert gts.scheme.stage == 3

    def test_explicit_name_not_overridden(self):
        """The fill only supplies a missing 'name'; an explicit one is left
        alone — and then rejected, because GTS admits runge kutta only."""
        with pytest.raises(ValueError, match="three-stage TVD"):
            GTSTimeSettings.model_validate({**self._BASE, "scheme": {"name": "euler"}})
