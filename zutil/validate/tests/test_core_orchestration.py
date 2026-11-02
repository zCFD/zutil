"""
Core orchestration coverage: full-model validation failures must never be
swallowed silently, capitalised user keys must survive in error paths,
exception handling must stay narrow, pre-errors must format as structured
text, warning nodes must be severity-tagged and free of shared mutable
state, and callable reference conditions must be probe-validated.
"""

import json
import logging

import pytest
from pydantic import ValidationError

from zutil.validate import error_handling
from zutil.validate.error_handling import ValidationErrorHandler, ZCFDValidationError
from zutil.validate.error_mapping import _clean_loc_parts, map_errors_to_nodes
from zutil.validate.validate_input import validate_zcfd_input
from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive
from zutil.validate.zcfd_schema import ZCFDInputModelBase


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _minimal_valid_config(bc_name="bc1", ic_name="IC_1", extra_ic=None):
    """A minimal config that passes every section check *and* the final
    full-model ZCFDInputModelBase.model_validate() pass.

    Args:
        bc_name: Name to give the single boundary condition (so tests can
                 probe capitalised/user-supplied BC names).
        ic_name: Name to give the single reference condition (ditto, for
                 IC names).
        extra_ic: Optional dict of extra {name: reference_condition} to add.
    """
    reference_conditions = {
        ic_name: {"temperature": 300, "pressure": 101325},
    }
    if extra_ic:
        reference_conditions.update(extra_ic)

    return {
        "config_version": 2,
        "solver": {
            "solver_settings": {"type": "compressible"},
            "initialisation": {"initial_conditions": ic_name},
            "equations": {"type": "euler"},
            "numerical_scheme": {"order": "second"},
            "fluid_properties": {},
            "reference_conditions": reference_conditions,
            "output_settings": {
                "solution": {"frequency": 10},
                "report": {"frequency": 10},
            },
            "time_settings": {"type": "steady"},
            "convergence_control": {
                "cycles": 100,
                "cfl": 1.0,
                "scheme": {"name": "runge kutta"},
            },
        },
        "model": {
            "base": {
                "mesh": "base.h5",
                "boundary_conditions": {
                    bc_name: {
                        "type": "farfield",
                        "condition": ic_name,
                        "kind": "riemann",
                        "zones": [1],
                    }
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# silent false success in comprehensive validation
# ---------------------------------------------------------------------------


class TestValidationNeverSwallowed:
    """validate_zcfd_comprehensive() must never turn a failure into a
    silent (None, []): callers take an empty list to mean the deck passed.
    """

    def test_full_model_unexpected_exception_surfaces_not_swallowed(
        self, monkeypatch, caplog
    ):
        """A genuinely unexpected exception (not ValidationError or
        ValidationErrorHandler) in the full-model pass must still produce
        a concrete error node — never an empty error list — while being
        logged for diagnosis rather than propagating as a raw traceback.
        """
        data = _minimal_valid_config()

        def _boom(cls, _data):
            raise RuntimeError("something truly unexpected broke")

        monkeypatch.setattr(ZCFDInputModelBase, "model_validate", classmethod(_boom))

        with caplog.at_level(logging.ERROR, logger="zutil.validate"):
            model, nodes = validate_zcfd_comprehensive(data)

        assert model is None
        assert nodes, "unexpected exceptions must not be swallowed either"

    def test_unexpected_failure_raises_via_validate_zcfd_input(self, monkeypatch):
        """validate_zcfd_input() must raise (not silently return None) when
        validation itself breaks.
        """
        data = _minimal_valid_config()

        def _boom(cls, _data):
            raise RuntimeError("something truly unexpected broke")

        monkeypatch.setattr(ZCFDInputModelBase, "model_validate", classmethod(_boom))

        with pytest.raises(ValidationErrorHandler):
            validate_zcfd_input(data)

    def test_valid_config_still_returns_model_with_no_errors(self):
        """Sanity check: the happy path still validates cleanly."""
        data = _minimal_valid_config()
        model, nodes = validate_zcfd_comprehensive(data)
        assert model is not None
        assert nodes == []


# ---------------------------------------------------------------------------
# capitalised user keys stripped from error paths
# ---------------------------------------------------------------------------


class TestCleanLocPartsKeepsUserKeys:
    """_clean_loc_parts must identify pydantic union-artifact segments
    structurally (against the real set of schema model class names), not
    by string shape — so a BC named 'Inlet' or an IC named 'IC1' survives
    in the error path instead of being mistaken for a class name.
    """

    def test_capitalised_bc_name_is_kept(self):
        parts = _clean_loc_parts(("boundary_conditions", "Inlet", "farfield", "kindd"))
        assert "Inlet" in parts

    def test_capitalised_ic_name_is_kept(self):
        parts = _clean_loc_parts(("reference_conditions", "IC1", "pressure"))
        assert "IC1" in parts

    def test_real_model_class_name_is_still_dropped(self):
        """Genuine union-discrimination artifacts (bare model class names
        inserted into loc by non-discriminated unions) must still be
        stripped.
        """
        parts = _clean_loc_parts(("FluidModelBase", "bogus_field"))
        assert "FluidModelBase" not in parts
        assert "bogus_field" in parts

    def test_function_wrapper_segments_still_dropped(self):
        parts = _clean_loc_parts(
            (
                "function-after[validate_compatibility(), CompressibleSolverBlockModel]",
                "equations",
            )
        )
        assert parts == ["equations"]

    def test_bc_named_inlet_end_to_end(self):
        """A boundary condition named 'Inlet' with an invalid field must
        keep 'Inlet' in the mapped error node's path — it must not
        disappear while a union-tag artifact leaks through instead.
        """
        from zutil.validate.schemas.model_schemas import ModelSettings

        try:
            ModelSettings.model_validate(
                {
                    "mesh": "test.h5",
                    "boundary_conditions": {
                        "Inlet": {"zones": [1], "type": "farfield", "kindd": "riemann"},
                    },
                }
            )
        except ValidationError as e:
            errors = e.errors()
            for err in errors:
                err["loc"] = ("model", "base") + tuple(err["loc"])
        else:
            pytest.fail("Expected a ValidationError for the malformed BC")

        nodes = map_errors_to_nodes(errors)
        assert nodes, "expected at least one mapped error node"
        assert any("Inlet" in n["nodes"] for n in nodes), (
            f"'Inlet' should appear in a mapped node's tree path, got: {nodes}"
        )

    def test_ic_named_ic1_end_to_end(self):
        """A reference condition named 'IC1' with an invalid field must
        keep 'IC1' in the mapped error node's path.
        """
        from zutil.validate.schemas.reference_conditions_schemas import (
            EulerReferenceConditions,
        )

        try:
            EulerReferenceConditions.model_validate({"temperature": -1, "pressure": 1})
        except ValidationError as e:
            errors = e.errors()
            for err in errors:
                err["loc"] = ("solver", "reference_conditions", "IC1") + tuple(
                    err["loc"]
                )
        else:
            pytest.fail("Expected a ValidationError for negative temperature")

        nodes = map_errors_to_nodes(errors)
        assert nodes, "expected at least one mapped error node"
        assert any("IC1" in n["nodes"] for n in nodes), (
            f"'IC1' should appear in a mapped node's tree path, got: {nodes}"
        )


# ---------------------------------------------------------------------------
# one validation pass
# ---------------------------------------------------------------------------


class TestOneValidationPass:
    """validate_zcfd_comprehensive validates the deck once, against
    ZCFDInputModelBase, and still reports every error at its own path."""

    def _errors(self, data):
        model, nodes = validate_zcfd_comprehensive(data)
        assert model is None
        return [
            (n["field_path"], n["display_message"])
            for n in nodes
            if n.get("severity", "error") == "error"
        ]

    def test_errors_in_several_routed_settings_are_all_reported(self):
        data = _minimal_valid_config(
            extra_ic={"IC_2": {"temperature": -1, "pressure": 101325}}
        )
        data["solver"]["reference_conditions"]["IC_1"]["pressure"] = -5
        data["solver"]["output_settings"]["solution"]["frequency"] = -3
        data["model"]["base"]["numerical_scheme"] = {"order": "fifth"}
        paths = [path for path, _ in self._errors(data)]
        assert "solver.reference_conditions.IC_1.pressure" in paths
        assert "solver.reference_conditions.IC_2.temperature" in paths
        assert "solver.output_settings.solution.frequency" in paths
        assert "model.base.numerical_scheme.order" in paths

    def test_an_unknown_solver_key_is_one_error_at_its_path(self):
        data = _minimal_valid_config()
        data["solver"]["bogus"] = 1
        assert self._errors(data) == [("solver.bogus", 'Unknown key "bogus".')]

    def test_a_decision_in_a_kind_reads_as_a_decision(self):
        data = _minimal_valid_config()
        data["model"]["base"]["boundary_conditions"]["bc1"]["kind"] = "DECIDE: kind"
        assert self._errors(data) == [
            (
                "model.base.boundary_conditions.bc1.kind",
                "Open decision left by update_input: DECIDE: kind",
            )
        ]

    @pytest.mark.parametrize("key", ["model", "solver"])
    def test_a_block_given_as_a_list_is_named(self, key):
        data = _minimal_valid_config()
        data[key] = [1]
        assert [path for path, _ in self._errors(data)] == [key]

    def test_routing_holds_for_every_whole_deck_entry_point(self):
        data = _minimal_valid_config()
        data["solver"]["equations"] = {"type": "rans", "turbulence": {"model": "sst"}}
        with pytest.raises(ValidationError, match="reynolds_no"):
            ZCFDInputModelBase(**data)
        with pytest.raises(ValidationError, match="reynolds_no"):
            ZCFDInputModelBase.model_validate_json(json.dumps(data))

    def test_an_incompressible_deck_reports_no_compressible_alternatives(self):
        data = _minimal_valid_config()
        data["solver"]["solver_settings"]["type"] = "incompressible"
        data["solver"]["bogus"] = 1
        paths = [path for path, _ in self._errors(data)]
        assert "solver.bogus" in paths
        assert not [p for p in paths if "ompressible" in p]


# ---------------------------------------------------------------------------
# structured pre-error accumulation, no regex header surgery
# ---------------------------------------------------------------------------


class TestCrossBlockErrorsAtTheirField:
    """A rule that reads two blocks reports at the setting it concerns, with
    that setting's path, tree node and reference anchor, under one header.
    """

    def _ghost(self):
        data = _minimal_valid_config()
        data["model"]["base"]["boundary_conditions"]["bc1"]["condition"] = "GHOST"
        return data

    def test_an_undefined_reference_is_reported_at_the_boundary_that_names_it(self):
        model, nodes = validate_zcfd_comprehensive(self._ghost())
        assert model is None
        (node,) = nodes
        assert node["field_path"] == "model.base.boundary_conditions.bc1.condition"
        assert node["nodes"] == ["boundary conditions", "bc1"]
        assert node["field_name"] == "condition"
        assert node["display_message"] == (
            "There is no reference condition 'GHOST'. "
            "The reference conditions are 'IC_1'."
        )
        assert node["anchor"].endswith("#condition")

    def test_the_report_has_one_header(self):
        with pytest.raises(ValidationErrorHandler) as excinfo:
            validate_zcfd_input(self._ghost())
        text = str(excinfo.value)
        assert text.count("Configuration validation failed") == 1
        assert text.startswith(
            "Configuration validation failed with 1 error:\n\n"
            "  model › base › boundary conditions › bc1 › condition\n"
        )
        assert "Value error" not in text

    def test_block_errors_come_before_cross_block_rules(self):
        """The cross-block rules read validated blocks, so they run once
        every block is valid; until then only the block errors are shown."""
        data = self._ghost()
        data["solver"]["equations"] = {"type": "bogus"}
        _model, nodes = validate_zcfd_comprehensive(data)
        assert [n["field_path"] for n in nodes] == ["solver.equations.type"]


# ---------------------------------------------------------------------------
# severity-tagged warning nodes, no shared mutable class state
# ---------------------------------------------------------------------------


class TestWarningsContract:
    """_validation_warnings must be gone entirely; unused reference
    conditions surface as severity="warning" nodes, and a config with
    only warnings still returns a validated model.
    """

    def test_validation_warnings_attribute_removed(self):
        assert not hasattr(ZCFDInputModelBase, "_validation_warnings")

    def test_unused_reference_condition_is_a_warning_node_not_an_error(self):
        data = _minimal_valid_config(
            extra_ic={"IC_Unused": {"temperature": 400, "pressure": 200000}}
        )

        model, nodes = validate_zcfd_comprehensive(data)

        assert model is not None, "warning-only config must still validate"
        assert not any(n.get("severity", "error") == "error" for n in nodes)
        warning_nodes = [n for n in nodes if n.get("severity") == "warning"]
        assert warning_nodes
        assert "IC_Unused" in warning_nodes[0]["display_message"]

    def test_validate_zcfd_input_does_not_raise_for_warnings_only(self):
        data = _minimal_valid_config(
            extra_ic={"IC_Unused": {"temperature": 400, "pressure": 200000}}
        )
        model = validate_zcfd_input(data)
        assert model is not None

    def test_validate_zcfd_input_logs_warnings(self, caplog):
        data = _minimal_valid_config(
            extra_ic={"IC_Unused": {"temperature": 400, "pressure": 200000}}
        )
        with caplog.at_level(logging.WARNING, logger="zutil.validate"):
            validate_zcfd_input(data)
        assert any("'IC_Unused' is not used" in r.message for r in caplog.records)

    def test_validate_zcfd_input_still_raises_when_any_node_is_an_error(self):
        data = _minimal_valid_config(
            extra_ic={"IC_Unused": {"temperature": 400, "pressure": 200000}}
        )
        # Introduce a real error alongside the warning.
        data["model"]["base"]["boundary_conditions"]["bc1"]["condition"] = "GHOST"
        with pytest.raises(ValidationErrorHandler):
            validate_zcfd_input(data)

    def test_a_script_building_the_model_gets_the_warning_logged(self, caplog):
        data = _minimal_valid_config(
            extra_ic={"IC_Unused": {"temperature": 400, "pressure": 200000}}
        )
        with caplog.at_level(logging.WARNING, logger="zutil.validate"):
            ZCFDInputModelBase.model_validate(data)
        assert any("'IC_Unused' is not used" in r.message for r in caplog.records)

    def test_the_warning_names_its_block(self):
        data = _minimal_valid_config(
            extra_ic={"IC_Unused": {"temperature": 400, "pressure": 200000}}
        )
        _model, (node,) = validate_zcfd_comprehensive(data)
        assert node["severity"] == "warning"
        assert node["field_path"] == "solver.reference_conditions"
        assert node["nodes"] == ["reference conditions"]


# ---------------------------------------------------------------------------
# dead error-handling code removed; ZCFDValidationError alias
# ---------------------------------------------------------------------------


class TestDeadCodeRemovedAndAlias:
    def test_extract_error_bullets_removed(self):
        assert not hasattr(error_handling, "extract_error_bullets")

    def test_enhance_type_error_context_removed(self):
        assert not hasattr(error_handling, "enhance_type_error_context")

    def test_handle_validation_exceptions_removed(self):
        assert not hasattr(error_handling, "handle_validation_exceptions")

    def test_zcfd_validation_error_alias_is_importable_and_matches(self):
        assert ZCFDValidationError is ValidationErrorHandler


class TestOpenDecisionsFailValidation:
    def test_a_decide_string_in_a_free_text_field_is_an_error(self):
        data = _minimal_valid_config()
        label = next(iter(data["model"]))
        data["model"][label]["mesh"] = (
            "DECIDE: which mesh file? (see conversion report, item 1)"
        )
        model, nodes = validate_zcfd_comprehensive(data)
        assert model is None
        errors = [n for n in nodes if n.get("severity", "error") == "error"]
        assert [n["field_path"] for n in errors] == [f"model.{label}.mesh"]
        assert "item 1" in errors[0]["display_message"]

    def test_a_decide_string_inside_a_list_names_its_index(self):
        data = _minimal_valid_config()
        data["solver"]["output_settings"] = {
            "solution": {"volume_variables": ["p", "DECIDE: keep 'eddy'?"]}
        }
        _, nodes = validate_zcfd_comprehensive(data)
        paths = [n["field_path"] for n in nodes]
        assert "solver.output_settings.solution.volume_variables.1" in paths

    def test_the_minimal_config_is_still_valid(self):
        model, nodes = validate_zcfd_comprehensive(_minimal_valid_config())
        assert model is not None
        assert not [n for n in nodes if n.get("severity", "error") == "error"]
