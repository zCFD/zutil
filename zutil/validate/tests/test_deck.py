"""check_deck: one route from a deck to a verdict for every caller.

The solver, live reload, the CLI and zui all render the same DeckReport, so
these tests pin what that report says for the decks that used to get
different answers depending on who asked.
"""

import copy
import logging
from pathlib import Path

import pytest

from zutil.Parameters import DeckReloader, DeckReloadError, Parameters
from zutil.fileutils import get_parameters_from_file
from zutil.validate import validate_zcfd_input
from zutil.validate.deck import DeckReport, check_deck, route_deck
from zutil.validate.error_handling import ValidationErrorHandler

HERE = Path(__file__).resolve().parent
DECKS = HERE / "decks"
TESTCASES = HERE.parents[3] / "testcases" / "testcases"
WARNING_ONLY = TESTCASES / "plate" / "plate_coarse.py"
LEGACY = TESTCASES / "cavity" / "buoyancy_cavity.py"
VALID = TESTCASES / "cavity" / "re400.py"
NO_CONFIG_VERSION = DECKS / "no_config_version.py"
UNUSED_IC_2 = "Reference condition 'IC_2' is not used anywhere in the deck."


def _solver_read(deck: Path) -> tuple:
    """What the launcher does: Parameters.read, then validate_parameters."""
    params = Parameters()
    raw = params.read(str(deck.with_suffix("")), str(deck))
    return params.validate_parameters(raw, origin=str(deck))


def _raw(deck: Path) -> dict:
    return get_parameters_from_file(deck)


class TestWarningOnlyDeck:
    """(a) A deck whose only node is a warning is valid, and says so."""

    def test_path_source_is_ok_with_the_warning(self):
        report = check_deck(WARNING_ONLY)

        assert report.ok
        assert report.errors == ()
        assert [w["display_message"] for w in report.warnings] == [UNUSED_IC_2]
        assert report.warnings[0]["severity"] == "warning"
        assert report.model is not None
        assert report.warning_lines() == [f"solver.reference_conditions: {UNUSED_IC_2}"]

    def test_dict_source_gives_the_same_report(self):
        by_path = check_deck(WARNING_ONLY)
        by_dict = check_deck(_raw(WARNING_ONLY), origin=WARNING_ONLY)

        assert by_dict.ok
        assert by_dict.warnings == by_path.warnings
        assert by_dict.model.to_cpp_dict() == by_path.model.to_cpp_dict()

    def test_solver_path_accepts_it_and_logs_the_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="zutil.validate"):
            valid, result = _solver_read(WARNING_ONLY)

        assert valid is True
        assert set(result) == {"_validated_model"}
        assert f"solver.reference_conditions: {UNUSED_IC_2}" in caplog.messages

    def test_raising_view_returns_the_model(self):
        assert validate_zcfd_input(WARNING_ONLY) is not None


class TestDeckWithoutConfigVersion:
    """(b) A deck with no config_version gets the solver's answer from every source."""

    def test_solver_path_rejects_it_as_legacy(self):
        valid, message = _solver_read(NO_CONFIG_VERSION)

        assert valid is False
        assert message.startswith("Legacy configuration detected (Config Version 0).")
        assert "'no_config_version.py'" in message

    def test_path_source_matches_the_solver(self):
        _, message = _solver_read(NO_CONFIG_VERSION)
        report = check_deck(NO_CONFIG_VERSION)

        assert not report.ok
        assert report.text() == message
        assert report.parameters is None
        assert report.model is None

    def test_dict_with_origin_matches_the_solver(self):
        _, message = _solver_read(NO_CONFIG_VERSION)
        report = check_deck(_raw(NO_CONFIG_VERSION), origin=NO_CONFIG_VERSION)

        assert report.text() == message

    def test_dict_without_origin_is_rejected_the_same_way(self):
        """zui sends a bare dict: same verdict, the deck is called 'input'."""
        _, message = _solver_read(NO_CONFIG_VERSION)
        report = check_deck(_raw(NO_CONFIG_VERSION))

        assert not report.ok
        assert report.text() == message.replace(
            "'no_config_version.py'", "'input'"
        ).replace("update_input no_config_version.py", "update_input <deck>.py")

    def test_legacy_validator_sees_the_deck_as_written(self):
        """The legacy validator sees the deck's own spaced key spellings, so it
        reports the deck's real legacy error rather than a normalised key."""
        _, message = _solver_read(LEGACY)
        cli = check_deck(LEGACY)

        assert message == cli.text()
        assert "data['incomp viscous']['solve energy']" in message
        assert "time_marching" not in message


class TestRouting:
    def test_normalises_keys_once_for_the_schema(self):
        parameters, origin = route_deck(VALID)

        assert origin == VALID.resolve()
        assert "numerical_scheme" in parameters["solver"]
        assert "numerical scheme" not in parameters["solver"]

    def test_valid_deck_reports_its_parameters_and_model(self):
        report = check_deck(VALID)

        assert report.ok
        assert report.warnings == ()
        assert report.parameters == route_deck(VALID)[0]
        assert report.text() == "Configuration is valid."

    def test_does_not_modify_the_callers_dict(self):
        raw = _raw(VALID)
        before = copy.deepcopy(raw)
        check_deck(raw)
        assert raw == before

    def test_config_version_spelling_is_normalised(self):
        raw = _raw(VALID)
        raw["Config Version"] = raw.pop("config version")

        assert check_deck(raw).ok

    def test_unsupported_version(self):
        report = check_deck(DECKS / "unsupported_version.py")

        assert report.text() == (
            "Unsupported configuration version: 3. Supported versions are: 2"
        )
        assert isinstance(report.cause, ValueError)

    @pytest.mark.parametrize("key", ["frpm", "FRPM settings", "caa"])
    def test_deprecated_codepath_keys(self, key):
        raw = _raw(VALID)
        raw[key] = {}
        report = check_deck(raw)

        assert isinstance(report.cause, ValidationErrorHandler)
        assert "high order and CAA codepaths are deprecated" in report.text()

    def test_deprecated_dg_equations(self):
        report = check_deck(DECKS / "deprecated_dg.py")

        assert isinstance(report.cause, ValidationErrorHandler)

    def test_override_file(self):
        report = check_deck(DECKS / "override_case.py")

        assert report.text().startswith("Overset configuration detected")
        assert "background.h5 -> missing_case.py: file not found" in report.text()

    def test_override_dict(self):
        report = check_deck({"mesh_case_pair": [("a.h5", "a.py")]})

        assert report.text().startswith("Overset configuration detected")

    def test_duplicate_spellings_are_an_error_not_a_crash(self):
        report = check_deck(DECKS / "duplicate_keys.py")

        assert not report.ok
        assert report.text().startswith("Duplicate key: 'numerical scheme'")

    def test_unreadable_deck_keeps_the_loading_exception(self):
        report = check_deck(DECKS / "does_not_exist.py")

        assert not report.ok
        assert isinstance(report.cause, ValueError)
        assert report.errors[0]["severity"] == "error"

    def test_not_a_deck(self):
        report = check_deck(42)

        assert isinstance(report.cause, TypeError)


@pytest.mark.parametrize(
    "deck", ["unsupported_version.py", "override_case.py", "no_config_version.py"]
)
def test_a_deck_the_schema_never_sees_raises_both_error_types(deck):
    """Callers catching either ValueError or ZCFDValidationError catch it."""
    with pytest.raises(ValidationErrorHandler) as excinfo:
        validate_zcfd_input(DECKS / deck)
    assert isinstance(excinfo.value, ValueError)


class TestModelOrRaise:
    def test_reraises_the_routing_exception(self):
        report = check_deck(DECKS / "unsupported_version.py")

        with pytest.raises(ValueError) as excinfo:
            report.model_or_raise()
        assert excinfo.value is report.cause

    def test_schema_errors_raise_with_the_text(self):
        report = check_deck({"config_version": 2})

        with pytest.raises(ValidationErrorHandler) as excinfo:
            report.model_or_raise()
        assert str(excinfo.value) == report.text()

    def test_report_with_no_errors_counts_as_ok(self):
        assert DeckReport().ok


class TestDeckReloader:
    def test_prime_takes_the_validated_dict_without_logging_warnings(self, caplog):
        reloader = DeckReloader(str(WARNING_ONLY.with_suffix("")), str(WARNING_ONLY))
        with caplog.at_level(logging.WARNING, logger="zutil.validate"):
            baseline = reloader.prime()

        assert baseline == check_deck(WARNING_ONLY).model.to_cpp_dict()
        assert f"solver.reference_conditions: {UNUSED_IC_2}" not in caplog.messages

    def test_a_legacy_deck_will_not_prime(self):
        reloader = DeckReloader(str(LEGACY.with_suffix("")), str(LEGACY))

        with pytest.raises(DeckReloadError, match="Legacy configuration detected"):
            reloader.prime()
