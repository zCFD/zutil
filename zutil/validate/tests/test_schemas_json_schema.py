"""
JSON Schema generation must succeed for every BaseModel in schemas/.

Two things can break it:
  (i) Vector3d/NormalisedVector3d/IntVector3d use a plain-validator-function
      core schema, so without __get_pydantic_json_schema__ pydantic cannot
      introspect them for JSON Schema generation.
  (ii) Bare Callable-typed fields (not inside a Union) raise
      PydanticInvalidForJsonSchema.

Fix: base_types.py now provides __get_pydantic_json_schema__ for the vector
types (array of exactly 3 numbers/integers); Callable-only fields are wrapped
in pydantic.json_schema.SkipJsonSchema so they're omitted from the generated
schema without touching the runtime core schema.

This module is the smoke test: import every module in schemas/, collect
every BaseModel subclass defined there, and assert model_json_schema()
succeeds for all of them (was 36/108 failing before the fix).
"""

import importlib
import inspect
import pkgutil

import pytest
from pydantic import BaseModel

import zutil.validate.schemas as schemas_pkg


def _collect_schema_models():
    """Import every module under schemas/ and collect BaseModel subclasses defined there."""
    models = {}
    for modinfo in pkgutil.iter_modules(
        schemas_pkg.__path__, schemas_pkg.__name__ + "."
    ):
        module = importlib.import_module(modinfo.name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseModel)
                and obj is not BaseModel
                and obj.__module__ == module.__name__
            ):
                models[f"{module.__name__}.{name}"] = obj
    return models


SCHEMA_MODELS = _collect_schema_models()


def test_at_least_one_hundred_models_discovered():
    """Sanity check that model discovery is actually walking the package."""
    assert len(SCHEMA_MODELS) >= 100


@pytest.mark.parametrize(
    "qualname", sorted(SCHEMA_MODELS.keys()), ids=lambda s: s.split(".")[-1]
)
def test_model_json_schema_succeeds(qualname):
    """Every BaseModel subclass in schemas/ must generate a JSON Schema cleanly."""
    model_cls = SCHEMA_MODELS[qualname]
    schema = model_cls.model_json_schema()
    assert isinstance(schema, dict)


class TestVectorJsonSchema:
    """Vector3d/NormalisedVector3d/IntVector3d render as fixed-length arrays."""

    def test_vector3d_schema_is_3_element_number_array(self):
        from zutil.validate.schemas.base_types import LinearVelocity

        schema = LinearVelocity.model_json_schema()
        vector_schema = schema["properties"]["vector"]
        assert vector_schema["type"] == "array"
        assert vector_schema["items"] == {"type": "number"}
        assert vector_schema["minItems"] == 3
        assert vector_schema["maxItems"] == 3

    def test_normalised_vector3d_inherits_array_schema(self):
        from zutil.validate.schemas.base_types import RotationalVelocity

        schema = RotationalVelocity.model_json_schema()
        axis_schema = schema["properties"]["axis"]
        assert axis_schema["type"] == "array"
        assert axis_schema["items"] == {"type": "number"}
        assert axis_schema["minItems"] == 3
        assert axis_schema["maxItems"] == 3

    def test_int_vector3d_schema_uses_integer_items(self):
        from zutil.validate.schemas.output_variables import (  # noqa: F401 - ensures package import works
            get_valid_variables_for_equation_type,
        )
        from zutil.validate.schemas.base_types import IntVector3d
        from pydantic import BaseModel, ConfigDict

        class _IntVecHolder(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            v: IntVector3d

        schema = _IntVecHolder.model_json_schema()
        v_schema = schema["properties"]["v"]
        assert v_schema["type"] == "array"
        assert v_schema["items"] == {"type": "integer"}
        assert v_schema["minItems"] == 3
        assert v_schema["maxItems"] == 3


class TestCallableFieldsStillValidateAtRuntime:
    """SkipJsonSchema must not change runtime validation behaviour for callables."""

    @staticmethod
    def _dummy_func(*, location):
        return location

    def test_momentum_source_zone_func_still_accepts_callable(self):
        from zutil.validate.schemas.fluid_zones_schemas import MomentumSourceZone

        def drive(*, locations):
            return [(0.0, 0.0, 1.0) for _ in locations]

        model = MomentumSourceZone(type="momentum", func=drive)
        assert callable(model.func)

    def test_transform_zone_base_requires_callable_func(self):
        from zutil.validate.schemas.fluid_zones_schemas import TransformZoneBase
        from pydantic import ValidationError

        model = TransformZoneBase(zones=[0], func=self._dummy_func)
        assert callable(model.func)

        with pytest.raises(ValidationError):
            TransformZoneBase(zones=[0])

    def test_cfl_model_ramp_still_accepts_callable(self):
        from zutil.validate.schemas.convergence_schemas import CFLModel

        model = CFLModel(cfl=1.0, cfl_ramp=lambda *, cycle, current_cfl: current_cfl)
        assert callable(model.cfl_ramp)

    def test_force_model_transform_still_accepts_callable(self):
        from zutil.validate.schemas.output_schemas import ForceModel

        model = ForceModel(zones=[0], transform=lambda *, vector: vector)
        assert callable(model.transform)

    def test_callable_fields_are_absent_from_generated_schema(self):
        """SkipJsonSchema should make the field disappear from the schema entirely."""
        from zutil.validate.schemas.fluid_zones_schemas import TransformZoneBase

        schema = TransformZoneBase.model_json_schema()
        assert "function" not in schema["properties"]
