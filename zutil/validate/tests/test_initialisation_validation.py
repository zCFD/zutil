"""
Tests for InitialisationModel validation rules.

Verifies that initial_conditions is required unless restart=True.
"""

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.initialisation_schemas import InitialisationModel


def dummy_init_func(*, location, wall_distance, pressure, temperature, velocity):
    """Dummy callable for testing function-based initialisation."""
    return {"pressure": 101325.0, "temperature": 300.0}


class TestInitialConditionsRequired:
    """initial_conditions is required unless restart=True."""

    def test_no_initial_conditions_without_restart_raises(self):
        """Should raise when neither initial_conditions nor restart is set."""
        with pytest.raises(ValidationError, match="initial_conditions.*required"):
            InitialisationModel()

    def test_no_initial_conditions_with_restart_passes(self):
        """Restart-only (no IC) should pass validation."""
        model = InitialisationModel(restart=True)
        assert model.initial_conditions is None
        assert model.reference is None

    def test_string_initial_conditions_passes(self):
        """Named IC string should work fine."""
        model = InitialisationModel(initial_conditions="IC_1")
        assert model.initial_conditions == "IC_1"

    def test_callable_initial_conditions_with_reference_passes(self):
        """Callable IC with reference should pass."""
        model = InitialisationModel(
            initial_conditions=dummy_init_func, reference="IC_1"
        )
        assert callable(model.initial_conditions)
        assert model.reference == "IC_1"

    def test_string_with_reference_passes(self):
        """Named IC with separate reference should both be accepted."""
        model = InitialisationModel(
            initial_conditions="IC_2",
            reference="IC_1",
        )
        assert model.initial_conditions == "IC_2"
        assert model.reference == "IC_1"


class TestReferenceRequiredForCallable:
    """reference is required when initial_conditions is a callable."""

    def test_callable_without_reference_raises(self):
        """Callable IC without reference should fail validation."""
        with pytest.raises(ValidationError, match="reference.*required"):
            InitialisationModel(initial_conditions=dummy_init_func)

    def test_callable_with_reference_passes(self):
        """Callable IC with reference should pass."""
        model = InitialisationModel(
            initial_conditions=dummy_init_func, reference="IC_1"
        )
        assert callable(model.initial_conditions)
        assert model.reference == "IC_1"

    def test_string_without_reference_passes(self):
        """String IC without reference should still pass (reference optional)."""
        model = InitialisationModel(initial_conditions="IC_1")
        assert model.initial_conditions == "IC_1"
        assert model.reference is None
