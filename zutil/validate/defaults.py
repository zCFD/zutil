"""Drop the settings a deck writes at their default."""

import typing
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .schemas.base_types import zcfd_metadata
from .schemas.union_tags import discriminator_of
from .schemas.variants import Variants
from .updatable import _nested_models
from .validate_zcfd_schema import validate_zcfd_comprehensive
from .validation_utils import replace_spaces_in_keys
from .zcfd_schema import (
    MESH_OVERRIDES,
    ZCFDInputModelBase,
    inherited_block,
    mesh_difference,
    resolve_mesh_settings,
)

STRIP_MODES = ("strict", "moderate", "none")


def strip_defaults(deck: dict, mode: str = "strict") -> dict:
    """*deck* without the values it sets to their schema default.

    ``"strict"`` drops every default, ``"moderate"`` only those of
    ``advanced`` fields, ``"none"`` nothing. Each block is stripped against
    the class it validated as. A key is kept at its
    default when the deck needs it to choose a version of a block: a union's
    discriminator, or a setting a version table reads whose removal would
    choose another version. A mesh
    block is never stripped against the schema, since a mesh value at the
    schema default still overrides the solver's; it drops only what equals
    the value it inherits. ``config_version`` is always kept, since
    :func:`check_deck` routes on it before the schema sees the deck. The
    result is the deck with its keys normalised.
    A deck that does not validate comes back unchanged.
    """
    if mode not in STRIP_MODES:
        raise ValueError(f"Unknown strip mode '{mode}': use one of {STRIP_MODES}.")
    if mode == "none":
        return deck
    model, _ = validate_zcfd_comprehensive(deck)
    if model is None:
        return deck
    normalised = replace_spaces_in_keys(deck)
    stripped = _strip(normalised, model, mode, "", normalised)
    stripped["model"] = _mesh_differences(normalised)
    if "config_version" in normalised:
        stripped = {"config_version": normalised["config_version"], **stripped}
    return stripped


def _mesh_differences(deck: dict) -> dict:
    resolved = resolve_mesh_settings(deck)["model"]
    meshes = {}
    for name, mesh in deck["model"].items():
        meshes[name] = {
            key: mesh_difference(inherited_block(deck, key), resolved[name][key])
            if key in MESH_OVERRIDES and isinstance(value, dict)
            else value
            for key, value in mesh.items()
        }
    return meshes


def _strip(raw: dict, validated: BaseModel, mode: str, path: str, deck: dict) -> dict:
    fields = type(validated).model_fields
    result = {}
    for key, value in raw.items():
        field = fields.get(key)
        if field is None:
            result[key] = value
            continue
        where = f"{path}.{key}" if path else key
        attr = getattr(validated, key)
        strippable = mode == "strict" or zcfd_metadata(field).get("advanced", False)
        if (
            strippable
            and _equal(attr, field.get_default(call_default_factory=True))
            and not _decides_a_version(deck, where)
        ):
            continue
        if isinstance(value, dict) and isinstance(attr, BaseModel):
            value = _strip_block(value, attr, field, mode, where, deck)
        elif isinstance(value, dict) and isinstance(attr, dict):
            value = {
                name: _strip_block(entry, attr[name], field, mode, where, deck)
                if isinstance(entry, dict) and isinstance(attr.get(name), BaseModel)
                else entry
                for name, entry in value.items()
            }
        result[key] = value
    return result


def _strip_block(
    raw: dict, validated: BaseModel, field: FieldInfo, mode: str, path: str, deck: dict
) -> dict:
    stripped = _strip(raw, validated, mode, path, deck)
    for tag in _tags_choosing(field, type(validated)):
        if tag in raw:
            stripped[tag] = raw[tag]
    return {key: stripped[key] for key in raw if key in stripped}


def _equal(value: object, default: object) -> bool:
    try:
        return bool(value == default)
    except ValueError:
        return False


def _tags_choosing(field: FieldInfo, model: type) -> set:
    """The discriminators of the unions in *field*'s type that lead to *model*."""
    tags = _tags_on_way_to(field.annotation, model)
    if tags is None:
        return set()
    return tags | (
        {field.discriminator} if isinstance(field.discriminator, str) else set()
    )


def _tags_on_way_to(hint: object, model: type) -> Optional[set]:
    if hint is model:
        return set()
    for arg in typing.get_args(hint):
        tags = _tags_on_way_to(arg, model)
        if tags is not None:
            return tags | ({discriminator_of(hint)} - {None})
    return None


def _decides_a_version(deck: dict, path: str) -> bool:
    without = _without(deck, path.split("."))
    return any(
        table.version_for(deck) is not table.version_for(without)
        for table in _variant_tables(ZCFDInputModelBase)
        if path in table.deciding
    )


def _without(node: dict, keys: list) -> dict:
    head, *rest = keys
    if head not in node:
        return node
    if not rest:
        return {k: v for k, v in node.items() if k != head}
    child = node[head]
    return {**node, head: _without(child, rest)} if isinstance(child, dict) else node


@lru_cache(maxsize=None)
def _variant_tables(root: type) -> tuple:
    tables, seen, todo = [], set(), [root]
    while todo:
        model = todo.pop()
        if model in seen:
            continue
        seen.add(model)
        for field in model.model_fields.values():
            tables += [m for m in field.metadata if isinstance(m, Variants)]
            todo += _nested_models(field.annotation)
    return tuple(tables)


__all__ = ["STRIP_MODES", "strip_defaults"]
