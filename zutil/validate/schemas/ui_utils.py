"""Utilities for UI introspection of discriminated union schemas.

Provides functions to automatically discover the structure of Pydantic
discriminated union types, so the UI can render the correct form fields
at each stage of type selection without manual variant mappings.

Supports two patterns used in the codebase:
  1. Tag-based: ``Annotated[Union[Annotated[Model, Tag("x")], ...], Field(discriminator="type")]``
  2. Discriminator-based: ``Annotated[Union[Model, ...], Discriminator("type")]``
     where each model has a ``Literal`` default on the discriminator field.
"""

from typing import Dict, Literal, Type, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic.types import Discriminator
from pydantic import Tag


def get_discriminator_field(union_type) -> str:
    """Extract the discriminator field name from a discriminated Union type.

    Args:
        union_type: An ``Annotated[Union[...], Field(discriminator=...) | Discriminator(...)]`` type.

    Returns:
        The field name used as discriminator (e.g. ``"type"``, ``"kind"``).

    Raises:
        ValueError: If the type does not contain a recognisable discriminator.
    """
    if get_origin(union_type) is not type(union_type):
        # Annotated type — metadata contains the discriminator
        args = get_args(union_type)
        for arg in args[1:]:
            # Pattern 1: Field(discriminator="...")
            if isinstance(arg, FieldInfo) and arg.discriminator:
                return arg.discriminator
            # Pattern 2: Discriminator("...")
            if isinstance(arg, Discriminator) and isinstance(arg.discriminator, str):
                return arg.discriminator

    raise ValueError(f"Cannot extract discriminator field from type: {union_type}")


def get_union_variants(union_type) -> Dict[str, Union[Type[BaseModel], type]]:
    """Extract discriminator-value → concrete-model mapping from a discriminated Union.

    For **single-stage** discriminators (e.g. ``WallBoundary``), the dict value
    is the concrete Pydantic model class.

    For **two-stage** discriminators (e.g. ``InflowBoundary`` which is itself a
    discriminated Union on ``kind``), the dict value is the sub-Union type.
    Call :func:`is_union_type` on the result to check, and call
    :func:`get_union_variants` again to drill into the second level.

    Works with both Tag-based and Discriminator-based unions.

    Args:
        union_type: An ``Annotated[Union[...], ...]`` discriminated union type.

    Returns:
        Dict mapping discriminator string values to model classes or sub-union types.

    Example::

        >>> from zutil.validate.schemas.boundary_conditions_schemas import BoundaryCondition
        >>> variants = get_union_variants(BoundaryCondition)
        >>> variants["wall"]
        <class '...WallBoundary'>
        >>> is_union_type(variants["inflow"])
        True
    """
    args = get_args(union_type)
    if not args:
        raise ValueError(f"Not a Union or Annotated type: {union_type}")

    # The first arg is the Union[...] itself
    inner = args[0]
    if get_origin(inner) is Union:
        members = get_args(inner)
    elif get_origin(union_type) is Union:
        # union_type is directly a Union (no Annotated wrapper)
        members = get_args(union_type)
    else:
        # Annotated wrapping a non-Union — single concrete model
        members = (inner,)

    disc_field = get_discriminator_field(union_type)
    variants: Dict[str, Union[Type[BaseModel], type]] = {}

    for member in members:
        member_args = get_args(member)

        if member_args:
            # Annotated[SomeType, Tag("value"), ...] or Annotated[SubUnion, FieldInfo, Tag("value")]
            tag_value = None
            inner_type = member_args[0]

            for ma in member_args[1:]:
                if isinstance(ma, Tag):
                    tag_value = ma.tag
                    break

            if tag_value is not None:
                # Check if inner_type is itself a Union (two-stage)
                if get_origin(inner_type) is Union:
                    # Wrap it back as Annotated with its metadata for recursive calls
                    # Find the FieldInfo with sub-discriminator
                    sub_metadata = [
                        ma
                        for ma in member_args[1:]
                        if isinstance(ma, (FieldInfo, Discriminator))
                    ]
                    if sub_metadata:
                        from typing import Annotated

                        variants[tag_value] = Annotated[
                            tuple([inner_type] + sub_metadata)
                        ]
                    else:
                        variants[tag_value] = inner_type
                else:
                    variants[tag_value] = inner_type
        else:
            # Plain model class (Discriminator-based pattern, no Tag wrapping).
            # Extract the discriminator value from the model's Literal field default.
            if isinstance(member, type) and issubclass(member, BaseModel):
                field_info = member.model_fields.get(disc_field)
                if field_info is None:
                    continue
                if get_origin(field_info.annotation) is Literal:
                    for value in get_args(field_info.annotation):
                        variants[value] = member
                elif field_info.default is not None:
                    variants[field_info.default] = member

    return variants


def is_union_type(type_hint) -> bool:
    """Check whether a type hint is a discriminated Union requiring further selection.

    Returns ``True`` if the hint is an ``Annotated[Union[...], discriminator]``
    or a bare ``Union`` — i.e. the UI needs to present another set of choices
    rather than rendering a concrete model's fields.

    Returns ``False`` for concrete Pydantic model classes.
    """
    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        return False

    origin = get_origin(type_hint)
    if origin is Union:
        return True

    # Annotated wrapper — check if the first arg is a Union
    args = get_args(type_hint)
    if args and get_origin(args[0]) is Union:
        return True

    return False
