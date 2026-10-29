"""The tags of a discriminated union, read from the union itself.

Each tag is written once, as the ``Literal`` on the discriminator field of
one member model. Every other list of the same tags (the choices on a UI
base model, a set of scheme names, an error message) is derived from the
union with :func:`tags_of` or :func:`literal_of`, so it cannot drift.

A member may itself be a union, as ``InflowBoundary`` is inside
``BoundaryCondition``; its tag is the value its own members share on the
outer discriminator.
"""

from typing import Annotated, Literal, Optional, Tuple, Union, get_args, get_origin

from pydantic import BaseModel, Discriminator
from pydantic.fields import FieldInfo


def _strip(hint):
    while get_origin(hint) is Annotated:
        hint = get_args(hint)[0]
    return hint


def discriminator_of(union) -> Optional[str]:
    """The field name a discriminated union is keyed on, or ``None``."""
    if get_origin(union) is not Annotated:
        return None
    for meta in get_args(union)[1:]:
        if isinstance(meta, FieldInfo) and isinstance(meta.discriminator, str):
            return meta.discriminator
        if isinstance(meta, Discriminator) and isinstance(meta.discriminator, str):
            return meta.discriminator
    return None


def members_of(union) -> tuple:
    """The direct members of a union, each still carrying its annotations."""
    inner = _strip(union)
    return get_args(inner) if get_origin(inner) is Union else (inner,)


def _literal_values(hint) -> tuple:
    hint = _strip(hint)
    if get_origin(hint) is Literal:
        return get_args(hint)
    if get_origin(hint) is Union:
        return tuple(v for arg in get_args(hint) for v in _literal_values(arg))
    return ()


def _member_tags(member, field: str) -> tuple:
    model = _strip(member)
    if isinstance(model, type) and issubclass(model, BaseModel):
        values = _literal_values(model.model_fields[field].annotation)
        if not values:
            raise TypeError(f"{model.__name__}.{field} is not a Literal")
        return values
    return tags_of(member, field)


def tags_of(union, field: Optional[str] = None) -> Tuple[str, ...]:
    """The tags of ``union`` in declaration order, without repeats.

    ``field`` names the discriminator; it is needed only for a bare
    ``Union`` that carries no discriminator of its own.
    """
    field = field or discriminator_of(union)
    if field is None:
        raise TypeError(f"{union!r} has no string discriminator; pass field=")
    tags = (tag for member in members_of(union) for tag in _member_tags(member, field))
    return tuple(dict.fromkeys(tags))


def literal_of(union, field: Optional[str] = None):
    """``Literal[...]`` of the tags of ``union``, for a field that names one of them."""
    return Literal[tags_of(union, field)]
