"""Dynamic-update whitelist derived from schema ``updatable=True`` metadata.

``zCFD_Field`` (see ``zutil.validate.schemas.base_types``) lets schema
authors mark a field as dynamically updatable at solver runtime::

    cfl: float = zCFD_Field(default=1.0, updatable=True)

That metadata lands in ``FieldInfo.json_schema_extra["zcfd"]["updatable"]``.
This module walks the schema model tree (starting from
``ZCFDInputModelBase``) and turns that per-field metadata into a flat set
of dotted parameter paths — the whitelist consumed by
``Parameters.partition_dynamic_update`` to enforce that a
control-file re-read at runtime only changes parameters the solver is
actually able to pick up live (restart/mesh/BC topology changes are not
in this set and must trigger a full restart instead).

This lives in its own module (rather than in ``zcfd_schema.py`` or
``Parameters.py``) specifically so the runtime bridge and the schema
package can evolve independently without file conflicts.
"""

from __future__ import annotations

import typing
from functools import lru_cache

from pydantic import BaseModel
from pydantic.fields import FieldInfo

__all__ = [
    "collect_updatable_paths",
    "updatable_paths",
    "schema_path_for_diff_key",
    "is_updatable",
    "display_name",
]


def _nested_models(annotation: typing.Any) -> typing.Iterator[type[BaseModel]]:
    """Yield BaseModel subclasses reachable from a field's type annotation.

    Handles the annotation shapes that show up across the schema package:
    plain model references, ``Optional[Model]`` / ``Union[...]``,
    ``List[Model]`` / ``Tuple[Model, ...]``, and ``Dict[str, Model]``
    (discriminated unions and containers-of-unions are covered because we
    recurse into every type argument, not just the first).
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return

    origin = typing.get_origin(annotation)
    if origin is None:
        return

    for arg in typing.get_args(annotation):
        if arg is type(None):
            continue
        yield from _nested_models(arg)


def _is_updatable(field: FieldInfo) -> bool:
    """True if a field's ``zCFD_Field(updatable=True)`` metadata is set."""
    extra = field.json_schema_extra
    if callable(extra):
        # zCFD_Field emits a merging callable (so annotation-level
        # contributions like applies_when survive) — materialise it.
        materialised: dict = {}
        extra(materialised)
        extra = materialised
    if not isinstance(extra, dict):
        return False
    zcfd_meta = extra.get("zcfd")
    if not isinstance(zcfd_meta, dict):
        return False
    return zcfd_meta.get("updatable") is True


def collect_updatable_paths(
    model: type[BaseModel],
    *,
    _prefix: str = "",
    _seen: frozenset[type[BaseModel]] = frozenset(),
) -> set[str]:
    """Return dotted paths of every ``updatable=True`` field under ``model``.

    Recurses into nested ``BaseModel`` fields at any depth (including
    those buried in ``Optional``/``Union``/``List``/``Dict`` annotations).
    A dotted path mirrors Python attribute access, e.g.
    ``"solver.time_marching.steady.cfl"``.

    ``_seen`` guards against infinite recursion if models ever reference
    each other cyclically; callers should not pass it explicitly.
    """
    if model in _seen:
        return set()
    seen = _seen | {model}

    paths: set[str] = set()
    for name, field in model.model_fields.items():
        path = f"{_prefix}.{name}" if _prefix else name
        if _is_updatable(field):
            paths.add(path)
        for nested in _nested_models(field.annotation):
            paths |= collect_updatable_paths(nested, _prefix=path, _seen=seen)
    return paths


@lru_cache(maxsize=1)
def updatable_paths() -> frozenset[str]:
    """Return the whitelist of dynamically-updatable parameter paths.

    Derived from ``ZCFDInputModelBase`` and every model reachable from it.
    The import of ``ZCFDInputModelBase`` is local (rather than at module
    scope) so that importing this module — and, transitively, importing
    ``zutil.Parameters`` — does not pull in the full validation/schema
    package unless dynamic-update enforcement is actually exercised.

    Cached: the schema model tree is static for the lifetime of the
    process, so this is only ever walked once.
    """
    from zutil.validate.zcfd_schema import ZCFDInputModelBase

    return frozenset(collect_updatable_paths(ZCFDInputModelBase))


def schema_path_for_diff_key(key: str) -> str:
    """Map a parameter-diff key onto the schema attribute path it refers to.

    A diff key from ``_compare_nested_structures`` differs from a schema
    path in two ways:

    - It joins segments with ``:`` rather than ``.``.
    - Anything under ``model`` carries the mesh name as its second segment
      (``model:naca0012:convergence_control:cycles``), which the schema
      walk never emits: the key of a ``Dict[str, Model]`` field is data the
      user chose, not part of the schema. That segment is dropped, giving
      ``model.convergence_control.cycles``.

    The second rule is unconditional because it is a property of the diff
    engine, not a guess: ``_compare_nested_structures`` walks the real
    dict, and ``model`` is a mapping of mesh name to settings, so every
    diff key below ``model`` names a mesh. (``model`` alone — the whole
    block appearing for the first time — has no second segment and is left
    as is; it is not whitelisted, so it is rejected either way.)

    Takes a **diff key**, not a schema path, and is deliberately not
    idempotent: feeding it its own output would strip a second real
    segment. Sniffing which form was passed would mean accepting a path
    whenever *either* reading is whitelisted, and a mesh named after a
    schema block (``model:convergence_control:...``) would then slip a
    non-updatable field past the gate. A safety gate should not rest on
    mesh names being sensible.
    """
    segments = key.replace(":", ".").split(".")
    if len(segments) > 1 and segments[0] == "model":
        del segments[1]
    return ".".join(segments)


def is_updatable(key: str) -> bool:
    """True if a parameter-diff key names a dynamically-updatable field."""
    return schema_path_for_diff_key(key) in updatable_paths()


def _field_titles(path: str) -> typing.List[str]:
    """Schema ``title`` of each segment of a dotted path, root first.

    Walks the model tree the same way :func:`collect_updatable_paths` does.
    Where a segment's annotation offers several models (a variant union),
    the first one carrying the next segment is followed. A segment with no
    ``title`` contributes its own name; an unresolvable path is cut short.
    """
    from zutil.validate.zcfd_schema import ZCFDInputModelBase

    titles: typing.List[str] = []
    candidates: typing.List[type[BaseModel]] = [ZCFDInputModelBase]
    for segment in path.split("."):
        field = None
        for model in candidates:
            field = model.model_fields.get(segment)
            if field is not None:
                break
        if field is None:
            break
        titles.append(field.title or segment.replace("_", " "))
        candidates = list(_nested_models(field.annotation))
    return titles


@lru_cache(maxsize=1)
def _ambiguous_leaf_titles() -> frozenset[str]:
    """Leaf titles shared by more than one updatable path."""
    counts: typing.Dict[str, int] = {}
    for path in updatable_paths():
        titles = _field_titles(path)
        if titles:
            counts[titles[-1]] = counts.get(titles[-1], 0) + 1
    return frozenset(title for title, n in counts.items() if n > 1)


def display_name(key: str) -> str:
    """Human name for a parameter-diff key, from the schema ``title`` metadata.

    ``solver:convergence_control:cfl:cfl`` -> ``CFL``;
    ``solver:output_settings:report:frequency`` -> ``Report Frequency``
    (the parent title is prefixed only when the leaf title alone is
    ambiguous); ``model:wing:convergence_control:cycles`` -> ``Cycles [wing]``.
    Falls back to the key itself when the schema has no field for it.
    """
    segments = key.replace(":", ".").split(".")
    mesh = segments[1] if len(segments) > 2 and segments[0] == "model" else None
    titles = _field_titles(schema_path_for_diff_key(key))
    if not titles or len(titles) != len(segments) - (1 if mesh else 0):
        return key
    name = titles[-1]
    if len(titles) > 1 and titles[-2] != name and name in _ambiguous_leaf_titles():
        name = f"{titles[-2]} {name}"
    return f"{name} [{mesh}]" if mesh else name
