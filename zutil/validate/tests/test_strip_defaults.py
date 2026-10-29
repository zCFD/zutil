"""strip_defaults drops what a deck sets to its default, against the class each block validated as."""

import copy
import json
import re
from pathlib import Path

import pytest

from zutil.docs.examples import _execute
from zutil.validate import (
    ZCFDInputModelBase,
    check_deck,
    strip_defaults,
    validate_zcfd_comprehensive,
)
from zutil.validate.tests.test_buoyancy_placement import (
    _incompressible_buoyancy_config,
)
from zutil.validate.tests.test_output_variable_names import _deck
from zutil.validate.validation_utils import replace_spaces_in_keys

TESTCASES = Path(__file__).resolve().parents[4] / "testcases" / "testcases"


def _compressible(scheme):
    deck = _deck(volume_variables=["p"])
    deck["solver"]["numerical scheme"] = scheme
    return deck


def _incompressible(scheme):
    deck = _incompressible_buoyancy_config()
    deck["solver"]["numerical_scheme"] = scheme
    return deck


def _dump(deck):
    model, _ = validate_zcfd_comprehensive(copy.deepcopy(deck))
    return json.loads(json.dumps(model.model_dump(), sort_keys=True, default=repr))


def _cpp(deck):
    deck = replace_spaces_in_keys(copy.deepcopy(deck))
    return ZCFDInputModelBase.model_validate(deck).to_cpp_dict()


class TestStrict:
    def test_keeps_the_config_version_check_deck_routes_on(self):
        deck = _compressible({"order": "second"})
        stripped = strip_defaults(deck, "strict")
        assert stripped["config_version"] == 2
        assert check_deck(stripped).ok

    def test_strips_against_the_scheme_the_deck_selects(self):
        deck = _compressible(
            {"order": "second", "linear gradients": False, "entropy fix": 0.1}
        )
        stripped = strip_defaults(deck, "strict")
        assert stripped["solver"]["numerical_scheme"] == {"entropy_fix": 0.1}

    def test_strips_a_choice_at_its_default(self):
        deck = _compressible({"order": "first", "limiter": "van albada"})
        stripped = strip_defaults(deck, "strict")
        assert stripped["solver"]["numerical_scheme"] == {"order": "first"}

    def test_keeps_the_keys_that_select_a_version(self):
        stripped = strip_defaults(_compressible({"order": "second"}), "strict")
        assert stripped["solver"]["solver_settings"] == {"type": "compressible"}
        assert stripped["solver"]["equations"]["type"] == "viscous"
        assert stripped["solver"]["time_settings"] == {}

    def test_keeps_discriminator_literals(self):
        deck = _incompressible({"order": "second"})
        deck["solver"]["convergence_control"]["scheme"]["implicit_relax"] = 0.9
        stripped = strip_defaults(deck, "strict")
        assert stripped["solver"]["convergence_control"]["scheme"] == {
            "name": "coupled",
            "implicit_relax": 0.9,
        }
        assert stripped["model"]["base"]["boundary_conditions"]["bc1"]["type"] == (
            "wall"
        )

    def test_drops_a_block_left_empty_when_it_has_a_default(self):
        deck = _compressible({"order": "second"})
        deck["solver"]["output settings"]["report"]["scale residuals by volume"] = False
        stripped = strip_defaults(deck, "strict")
        assert (
            "scale_residuals_by_volume"
            not in (stripped["solver"]["output_settings"]["report"])
        )

    def test_the_stripped_deck_solves_the_same_case(self):
        deck = _compressible(
            {"order": "second", "first order cycles": 0, "limiter coefficient": 5.0}
        )
        assert _cpp(strip_defaults(deck, "strict")) == _cpp(deck)

    def test_normalises_the_keys_it_strips(self):
        deck = _compressible({"order": "first", "limiter coefficient": 5.0})
        stripped = strip_defaults(deck, "strict")
        assert "numerical scheme" not in stripped["solver"]
        assert stripped["solver"]["numerical_scheme"] == {"order": "first"}

    def test_a_velocity_written_as_its_default_keeps_its_form(self):
        deck = _compressible({"order": "first"})
        deck["solver"]["reference conditions"]["IC_1"]["v"] = {"vector": [0, 0, 0]}
        assert _dump(strip_defaults(deck, "strict")) == _dump(deck)


class TestModerate:
    def test_keeps_plain_defaults_and_strips_advanced_ones(self):
        deck = _compressible(
            {"order": "second", "first order cycles": 0, "limiter coefficient": 5.0}
        )
        stripped = strip_defaults(deck, "moderate")
        assert stripped["solver"]["numerical_scheme"] == {
            "order": "second",
            "first_order_cycles": 0,
        }

    def test_a_field_advanced_only_in_the_compressible_variant(self):
        compressible = strip_defaults(
            _compressible({"order": "second", "viscous time step factor": 1.0}),
            "moderate",
        )
        incompressible = strip_defaults(
            _incompressible({"order": "second", "viscous_time_step_factor": 1.0}),
            "moderate",
        )
        assert (
            "viscous_time_step_factor"
            not in (compressible["solver"]["numerical_scheme"])
        )
        assert (
            incompressible["solver"]["numerical_scheme"]["viscous_time_step_factor"]
            == 1.0
        )


class TestUnchanged:
    def test_none_strips_nothing(self):
        deck = _compressible({"order": "second", "limiter coefficient": 5.0})
        assert strip_defaults(deck, "none") == deck

    def test_an_invalid_deck_comes_back_unchanged(self):
        deck = _compressible({"order": "second", "limiter coefficient": 5.0})
        del deck["solver"]["reference conditions"]["IC_1"]["pressure"]
        assert strip_defaults(copy.deepcopy(deck), "strict") == deck

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="lenient"):
            strip_defaults(_compressible({"order": "second"}), "lenient")


class TestMeshBlocks:
    def _deck(self, mesh_scheme):
        deck = _compressible({"order": "second", "limiter coefficient": 3.0})
        deck["model"]["block"]["numerical scheme"] = mesh_scheme
        return deck

    def test_a_mesh_value_at_the_schema_default_is_kept(self):
        stripped = strip_defaults(self._deck({"limiter_coefficient": 5.0}), "strict")
        assert stripped["model"]["block"]["numerical_scheme"] == {
            "limiter_coefficient": 5.0
        }

    def test_a_mesh_value_the_solver_already_sets_is_dropped(self):
        deck = self._deck({"limiter_coefficient": 3.0, "order": "first"})
        stripped = strip_defaults(deck, "strict")
        assert stripped["model"]["block"]["numerical_scheme"] == {"order": "first"}
        assert _cpp(stripped) == _cpp(deck)


def _corpus():
    if not TESTCASES.is_dir():
        return []
    controlfile = re.compile(r'^\s*-?\s*controlfile:\s*"?([^"\s]+\.py)"?\s*$', re.M)
    return sorted(
        {
            yml.parent / name
            for yml in TESTCASES.rglob("*.zcfd-test.yml")
            for name in controlfile.findall(yml.read_text())
        }
    )


@pytest.mark.parametrize(
    "path", _corpus(), ids=lambda p: p.relative_to(TESTCASES).as_posix()
)
def test_a_testcase_deck_validates_the_same_once_stripped(path):
    deck = _execute(path)
    stripped = strip_defaults(copy.deepcopy(deck))
    assert _dump(stripped) == _dump(deck)
    assert check_deck(stripped).ok == check_deck(deck).ok
