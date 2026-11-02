"""
Direct accept/reject coverage for the hand-written plain-validator-function
types in base_types.py: Vector3d, NormalisedVector3d, and IntVector3d.

test_schemas_json_schema.py covers the *JSON Schema shape* these types
render as; this file covers the runtime validate()/normalize() behaviour.
"""

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from zutil.validate.schemas.base_types import (
    AnyOf,
    IntVector3d,
    LinearVelocity,
    RotationalVelocity,
    When,
    zCFD_Field,
    zcfd_metadata,
)


class TestVector3dValidate:
    """Vector3d.validate: accepts a 3-element list/tuple, coerces to float."""

    def test_list_of_floats_accepted(self):
        v = LinearVelocity(type="linear", vector=[1.0, 2.0, 3.0])
        assert list(v.vector) == [1.0, 2.0, 3.0]

    def test_list_of_ints_coerced_to_float(self):
        v = LinearVelocity(type="linear", vector=[1, 2, 3])
        assert list(v.vector) == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in v.vector)

    def test_tuple_accepted(self):
        v = LinearVelocity(type="linear", vector=(1.0, 0.0, 0.0))
        assert list(v.vector) == [1.0, 0.0, 0.0]

    def test_non_list_or_tuple_rejected(self):
        with pytest.raises(ValidationError, match=r"must be 3 numbers"):
            LinearVelocity(type="linear", vector="1,0,0")

    def test_too_few_elements_rejected(self):
        with pytest.raises(ValidationError, match="exactly 3 numbers, not 2"):
            LinearVelocity(type="linear", vector=[1.0, 0.0])

    def test_too_many_elements_rejected(self):
        with pytest.raises(ValidationError, match="exactly 3 numbers, not 4"):
            LinearVelocity(type="linear", vector=[1.0, 0.0, 0.0, 0.0])

    def test_the_message_names_the_field_and_shows_the_shape(self):
        """These are read in the UI's error panel, so they say what to
        type rather than naming the Python type that rejected it."""
        with pytest.raises(ValidationError) as excinfo:
            LinearVelocity(type="linear", vector="1,0,0")
        message = str(excinfo.value)
        assert "'vector' must be 3 numbers, e.g. [1.0, 0.0, 0.0]" in message
        assert "Vector3d" not in message


class TestNormalisedVector3dNormalize:
    """NormalisedVector3d.normalize (chained after Vector3d.validate):
    divides by magnitude so the stored vector is always unit length."""

    def test_already_unit_vector_unchanged(self):
        rv = RotationalVelocity(
            type="rotating",
            axis=[1.0, 0.0, 0.0],
            origin=[0.0, 0.0, 0.0],
            omega=1.0,
        )
        assert list(rv.axis) == [1.0, 0.0, 0.0]

    def test_non_unit_vector_is_normalised(self):
        rv = RotationalVelocity(
            type="rotating",
            axis=[2.0, 0.0, 0.0],
            origin=[0.0, 0.0, 0.0],
            omega=1.0,
        )
        assert list(rv.axis) == pytest.approx([1.0, 0.0, 0.0])

    def test_non_axis_aligned_vector_normalised_to_unit_length(self):
        rv = RotationalVelocity(
            type="rotating",
            axis=[3.0, 4.0, 0.0],
            origin=[0.0, 0.0, 0.0],
            omega=1.0,
        )
        mag = sum(x * x for x in rv.axis) ** 0.5
        assert mag == pytest.approx(1.0)
        assert list(rv.axis) == pytest.approx([0.6, 0.8, 0.0])

    def test_zero_vector_rejected(self):
        """A [0,0,0] axis is physically meaningless (no direction to
        normalise), so normalise_vector_with_warning raises ValueError
        instead of silently passing the zero vector through unnormalised -
        matching this module's convention for invalid geometric input
        (e.g. Vector3d.validate's shape/length checks).
        """
        with pytest.raises(ValidationError, match="[Cc]annot normalise"):
            RotationalVelocity(
                type="rotating",
                axis=[0.0, 0.0, 0.0],
                origin=[0.0, 0.0, 0.0],
                omega=1.0,
            )

    def test_wrong_length_still_rejected(self):
        """The inherited Vector3d.validate length check still applies before
        normalisation runs."""
        with pytest.raises(ValidationError, match="exactly 3 numbers"):
            RotationalVelocity(
                type="rotating",
                axis=[1.0, 0.0],
                origin=[0.0, 0.0, 0.0],
                omega=1.0,
            )


class TestWallRotation:
    def test_the_centre_of_rotation_is_origin(self):
        rv = RotationalVelocity(
            type="rotating",
            axis=[0.0, 0.0, 1.0],
            origin=[1.0, 2.0, 3.0],
            omega=1.0,
        )
        assert list(rv.origin) == [1.0, 2.0, 3.0]

    def test_center_is_refused(self):
        with pytest.raises(ValidationError, match="center"):
            RotationalVelocity(
                type="rotating",
                axis=[0.0, 0.0, 1.0],
                center=[0.0, 0.0, 0.0],
                origin=[0.0, 0.0, 0.0],
                omega=1.0,
            )


class TestWhenWording:
    """``text`` and ``hint`` change a clause's wording, never what it holds for."""

    CONDITIONS = {"solver.time_settings.type": ("unsteady",)}

    def test_default_wording_is_generated_from_the_conditions(self):
        when = When(self.CONDITIONS)
        assert when.condition_text() == "solver.time settings.type is 'unsteady'"
        assert when.describe("motion") == (
            "'motion' only applies when solver.time settings.type is 'unsteady'"
        )
        assert when.describe_required("motion") == (
            "'motion' is required when solver.time settings.type is 'unsteady'"
        )
        assert when.to_schema() == {"solver.time_settings.type": ["unsteady"]}

    def test_text_replaces_the_generated_condition(self):
        when = When(self.CONDITIONS, text="the run is unsteady")
        assert when.condition_text() == "the run is unsteady"
        assert (
            when.describe("motion") == "'motion' only applies when the run is unsteady"
        )
        assert when.to_schema() == {"solver.time_settings.type": ["unsteady"]}

    def test_hint_follows_the_sentence(self):
        when = When(
            self.CONDITIONS, text="the run is unsteady", hint="Use a frame instead."
        )
        assert when.describe("motion") == (
            "'motion' only applies when the run is unsteady. Use a frame instead."
        )
        assert when.describe_required("motion") == (
            "'motion' is required when the run is unsteady. Use a frame instead."
        )
        assert when.describe("type", value="rotating frame") == (
            "'type': 'rotating frame' only applies when the run is unsteady. "
            "Use a frame instead."
        )
        assert when.to_schema() == {"solver.time_settings.type": ["unsteady"]}

    def test_wording_does_not_change_what_the_clause_holds_for(self):
        root = {"solver": {"time_settings": {"type": "steady"}}}
        plain = When(self.CONDITIONS)
        worded = When(self.CONDITIONS, text="the run is unsteady", hint="Hint.")
        assert plain.evaluate(root) is worded.evaluate(root) is False

    def test_any_of_wording_is_unchanged(self):
        clause = AnyOf(When({"a": (1,)}), When({"b": (2,)}))
        assert clause.describe("x") == "'x' only applies when either a is 1 or b is 2"


class _IntVecHolder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    v: IntVector3d


class TestIntVector3dValidate:
    """IntVector3d.validate: accepts a 3-element list/tuple, coerces to int."""

    def test_list_of_ints_accepted(self):
        holder = _IntVecHolder(v=[1, 2, 3])
        assert list(holder.v) == [1, 2, 3]

    def test_list_of_floats_coerced_to_int(self):
        holder = _IntVecHolder(v=[1.9, 2.1, 3.0])
        assert list(holder.v) == [1, 2, 3]
        assert all(isinstance(x, int) for x in holder.v)

    def test_non_list_or_tuple_rejected(self):
        with pytest.raises(ValidationError, match="must be 3 whole numbers"):
            _IntVecHolder(v="1,2,3")

    def test_wrong_length_rejected(self):
        with pytest.raises(ValidationError, match="exactly 3 whole numbers, not 2"):
            _IntVecHolder(v=[1, 2])


class TestZcfdMetadata:
    """zcfd_metadata reads a field's zCFD_Field extras without building a schema."""

    def test_reads_what_the_field_declares(self):
        from zutil.validate.schemas.numerical_schemes_schemas import (
            FVCompressibleViscousNumericalScheme,
        )

        field = FVCompressibleViscousNumericalScheme.model_fields[
            "viscous_time_step_factor"
        ]
        assert zcfd_metadata(field)["advanced"] is True

    def test_merges_a_json_schema_extra_zcfd_entry(self):
        class Holder(BaseModel):
            x: int = zCFD_Field(
                default=1, units="m", json_schema_extra={"zcfd": {"note": "n"}}
            )

        assert zcfd_metadata(Holder.model_fields["x"]) == {
            "advanced": False,
            "updatable": False,
            "units": "m",
            "note": "n",
        }

    def test_a_plain_field_has_none(self):
        class Holder(BaseModel):
            x: int = 1

        assert zcfd_metadata(Holder.model_fields["x"]) == {}
