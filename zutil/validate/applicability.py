"""Generic enforcement of applies_when clauses and exclusive fields.

A setting annotated ``ConditionalOptional[X, When({...})]`` (or one of its
siblings in ``schemas.base_types``) carries its applicability clause as
data.  This module is the validation-side consumer: it walks a validated
root model instance, finds every annotated field that is set while its
clause evaluates false, or unset while a required clause holds, and every
group of :class:`Exclusive` alternatives set more than once, and reports
each with the rule's own wording.  The UI consumes the same rules from the
JSON schema (``zcfd.applies_when``, ``zcfd.exclusive``) — neither side
restates them.
"""

import functools
from typing import Any, Callable, Dict, List, Tuple, Type

from pydantic import BaseModel

from .schemas.base_types import (
    Advises,
    AppliesWhen,
    Connects,
    Disjoint,
    Exclusive,
    ModelWide,
    RequiredWhen,
    deck_spelling,
)


class ApplicabilityError(ValueError):
    """One or more settings are set while their applies_when clause is false.

    Carries the structured ``violations`` — ``(field_path, message)`` pairs —
    so error reporting can route each one to its real tree node / form field
    instead of a generic configuration-level bucket.
    """

    def __init__(self, violations: List[Tuple[str, str]]):
        self.violations = violations
        super().__init__(
            "Configuration contains settings that are inconsistent with the "
            "rest of the case:\n  • "
            + "\n  • ".join(f"{path}: {message}" for path, message in violations)
        )


@functools.cache
def _fields_with(model_cls: Type[BaseModel], marker_type: type) -> tuple:
    """``(field_name, marker, field_info)`` for each field marked ``marker_type``."""
    return tuple(
        (name, marker, field_info)
        for name, field_info in model_cls.model_fields.items()
        for marker in field_info.metadata
        if isinstance(marker, marker_type)
    )


def _is_engaged(value: Any, field_info: Any) -> bool:
    """True if the field is actively set to something other than its default.

    A value equal to the field's default is not a violation even when the
    clause is false — e.g. a flag left at ``False`` (always present in the
    dump) is inert, and rejecting it would flag every deck that never
    touched the setting.
    """
    if value is None:
        return False
    default = field_info.get_default(call_default_factory=True)
    try:
        return not bool(value == default)
    except Exception:
        return value is not default


def _selects_gated_value(value: Any, gated: tuple) -> bool:
    try:
        return any(bool(value == v) for v in gated)
    except Exception:
        return False


def _dump(instance: "BaseModel | None") -> "dict | None":
    if instance is None:
        return None
    return instance.model_dump(exclude_none=True, mode="python")


def _owner_key(path: str) -> str:
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _clause_violations(
    instance: BaseModel,
    path: str,
    root_dict: dict,
    mesh_anchor: "str | None",
    parent: "BaseModel | None",
) -> List[Tuple[str, str]]:
    violations = []
    nodes: dict = {}

    def anchored(marker):
        if marker.when.uses_self_anchor and "self" not in nodes:
            nodes["self"] = _dump(instance)
        if marker.when.uses_parent_anchor and "parent" not in nodes:
            nodes["parent"] = _dump(parent)
        return nodes.get("self"), nodes.get("parent")

    for field_name, marker, field_info in _fields_with(type(instance), AppliesWhen):
        value = getattr(instance, field_name, None)
        gated = marker.values
        if gated is not None:
            engaged = _selects_gated_value(value, gated)
        elif marker.written:
            engaged = value is not None and field_name in instance.model_fields_set
        else:
            engaged = _is_engaged(value, field_info)
        if not engaged and not (marker.required and value is None):
            continue
        self_node, parent_node = anchored(marker)
        holds = marker.when.evaluate(
            root_dict,
            mesh_anchor=mesh_anchor,
            self_node=self_node,
            parent_node=parent_node,
        )
        if engaged == holds:
            continue
        field_path = f"{path}.{field_name}" if path else field_name
        subject = field_name
        if marker.when.uses_parent_anchor and path:
            subject = f"{_owner_key(path)}.{field_name}"
        if not engaged:
            message = marker.when.describe_required(subject)
        elif gated is None:
            message = marker.when.describe(subject)
        else:
            message = marker.when.describe(subject, value=value)
            if marker.hint:
                message += f"; {marker.hint}"
        violations.append((field_path, message))

    for field_name, marker, _ in _fields_with(type(instance), RequiredWhen):
        if getattr(instance, field_name, None) is not None:
            continue
        self_node, parent_node = anchored(marker)
        if not marker.when.evaluate(
            root_dict,
            mesh_anchor=mesh_anchor,
            self_node=self_node,
            parent_node=parent_node,
        ):
            continue
        field_path = f"{path}.{field_name}" if path else field_name
        message = marker.when.describe_missing(field_name)
        applies = next(
            (
                m
                for name, m, _ in _fields_with(type(instance), AppliesWhen)
                if name == field_name and m.values is None
            ),
            None,
        )
        if applies is not None:
            self_node, parent_node = anchored(applies)
            if not applies.when.evaluate(
                root_dict,
                mesh_anchor=mesh_anchor,
                self_node=self_node,
                parent_node=parent_node,
            ):
                message += f", and only applies when {applies.when.condition_text()}"
        violations.append((field_path, message))

    for field_name, marker, _ in _fields_with(type(instance), Exclusive):
        given = [
            name
            for name in marker.members(field_name)
            if getattr(instance, name, None) is not None
        ]
        field_path = f"{path}.{field_name}" if path else field_name
        if len(given) > 1:
            violations.append((field_path, marker.describe(field_name)))
        elif marker.required and not given:
            violations.append((field_path, marker.describe_missing(field_name)))
    return violations


def _walk(
    instance: Any,
    path: str,
    visit: Callable[[BaseModel, str, "str | None", "BaseModel | None"], None],
    mesh_anchor: "str | None" = None,
    parent: "BaseModel | None" = None,
) -> None:
    if isinstance(instance, BaseModel):
        visit(instance, path, mesh_anchor, parent)
        for field_name in type(instance).model_fields:
            child = getattr(instance, field_name, None)
            if child is not None:
                child_path = f"{path}.{field_name}" if path else field_name
                _walk(child, child_path, visit, mesh_anchor, instance)
    elif isinstance(instance, dict):
        for key, child in instance.items():
            child_path = f"{path}.{key}" if path else str(key)
            # Entering model.<name> establishes the mesh anchor for
            # <mesh>-relative When conditions in that subtree.
            child_anchor = child_path if path == "model" else mesh_anchor
            _walk(child, child_path, visit, child_anchor, parent)
    elif isinstance(instance, (list, tuple)):
        for i, child in enumerate(instance):
            _walk(child, f"{path}[{i}]", visit, mesh_anchor, parent)


def find_applicability_violations(root: BaseModel) -> List[Tuple[str, str]]:
    """All applies_when violations in a validated root model instance.

    Returns ``(field_path, message)`` pairs. Clause paths are root-anchored
    dotted paths into the v2 parameters dict, so clauses are evaluated
    against the root model's dump.
    """
    root_dict = root.model_dump(exclude_none=True, mode="python")
    violations: List[Tuple[str, str]] = []
    connections: list = []
    model_wide: Dict[Tuple[str, str], list] = {}
    disjoint: Dict[Tuple[str, str, Disjoint], list] = {}

    def visit(instance, path, mesh_anchor, parent):
        if mesh_anchor is not None:
            for field_name, _, _ in _fields_with(type(instance), ModelWide):
                model_wide.setdefault((mesh_anchor, field_name), []).append(
                    (path, instance, field_name)
                )
            for field_name, marker, _ in _fields_with(type(instance), Disjoint):
                value = getattr(instance, field_name, None)
                if value is not None:
                    disjoint.setdefault((mesh_anchor, field_name, marker), []).append(
                        (path, value)
                    )
        for field_name, marker, _ in _fields_with(type(instance), Connects):
            value = getattr(instance, field_name, None)
            if value is not None:
                connections.append(
                    (path, mesh_anchor, instance, field_name, marker, value)
                )
        violations.extend(
            _clause_violations(instance, path, root_dict, mesh_anchor, parent)
        )

    _walk(root, "", visit)
    for connection in connections:
        violations.extend(_connection_violations(root, *connection))
    for (mesh_anchor, field_name), holders in model_wide.items():
        violations.extend(_model_wide_violations(mesh_anchor, field_name, holders))
    for (_, field_name, marker), holders in disjoint.items():
        violations.extend(_disjoint_violations(field_name, marker, holders))
    return violations


def _disjoint_violations(
    field_name: str, marker: Disjoint, holders: list
) -> List[Tuple[str, str]]:
    violations = []
    for index, (path, value) in enumerate(holders):
        for other_path, other_value in holders[:index]:
            shared = [v for v in value if v in other_value]
            if shared:
                violations.append(
                    (
                        f"{path}.{field_name}",
                        marker.describe(field_name, shared, _owner_key(other_path)),
                    )
                )
    return violations


def _model_wide_violations(
    mesh_anchor: str, field_name: str, holders: list
) -> List[Tuple[str, str]]:
    def shown(instance) -> str:
        value = getattr(instance, field_name)
        default = "" if field_name in instance.model_fields_set else " (the default)"
        return f"{value}{default}"

    first_path, first, _ = next(
        (h for h in holders if field_name in h[1].model_fields_set), holders[0]
    )
    model = mesh_anchor.split(".", 1)[1]
    key = deck_spelling(field_name)
    return [
        (
            f"{path}.{field_name}",
            f"'{key}' is {shown(instance)} here and {shown(first)} on "
            f"'{first_path.rsplit('.', 1)[-1]}'. The solver applies one value to "
            f"the whole of model '{model}', so every boundary on it gives the same "
            "one.",
        )
        for path, instance, _ in holders
        if getattr(instance, field_name) != getattr(first, field_name)
    ]


def find_applicability_warnings(root: BaseModel) -> List[Tuple[str, str]]:
    """``(field_path, message)`` for every :class:`Advises` field that is
    set while its clause holds, in a validated root model instance."""
    root_dict = root.model_dump(exclude_none=True, mode="python")
    warnings: List[Tuple[str, str]] = []

    def visit(instance, path, mesh_anchor, parent):
        markers = _fields_with(type(instance), Advises)
        if not markers:
            return
        self_node = _dump(instance)
        parent_node = _dump(parent)
        for field_name, marker, field_info in markers:
            if not _is_engaged(getattr(instance, field_name, None), field_info):
                continue
            if marker.when.evaluate(
                root_dict,
                mesh_anchor=mesh_anchor,
                self_node=self_node,
                parent_node=parent_node,
            ):
                field_path = f"{path}.{field_name}" if path else field_name
                warnings.append((field_path, marker.describe(_owner_key(path))))

    _walk(root, "", visit)
    return warnings


def _names(value) -> List[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _connection_violations(
    root, path, mesh_anchor, instance, field_name, marker, value
):
    field_path = f"{path}.{field_name}"
    models = getattr(root, "model", None) or {}
    if mesh_anchor is None:
        return []
    own_model = mesh_anchor.split(".", 1)[1]
    own_boundary = path.rsplit(".", 1)[-1]
    targets = _names(getattr(value, "model", None)) or [own_model]

    found = []
    for target in targets:
        if target not in models:
            found.append(
                (
                    field_path,
                    f"'{field_name}' names model '{target}', which is not a model "
                    f"in this case (models: {', '.join(models)})",
                )
            )
        elif marker.other_model and target == own_model:
            found.append(
                (
                    field_path,
                    f"'{field_name}' must name another model, not '{own_model}' itself",
                )
            )
    if found or not marker.boundary:
        return found

    target_model = targets[0]
    boundary = value.boundary
    boundaries = models[target_model].boundary_conditions or {}
    if target_model == own_model and boundary == own_boundary:
        return [
            (
                field_path,
                f"'{field_name}' names its own boundary '{boundary}'; it must name "
                "the boundary on the other side",
            )
        ]
    if boundary not in boundaries:
        return [
            (
                field_path,
                f"'{field_name}' names boundary '{boundary}' on '{target_model}', "
                f"which does not exist (boundaries on {target_model}: "
                f"{', '.join(boundaries)})",
            )
        ]
    if not marker.reciprocal:
        return []

    other = boundaries[boundary]
    kind = getattr(instance, "kind", None)
    if (getattr(other, "type", None), getattr(other, "kind", None)) != (
        getattr(instance, "type", None),
        kind,
    ):
        return [
            (
                field_path,
                f"'{field_name}' names '{boundary}' on '{target_model}', which is "
                f"not a {kind} {getattr(instance, 'type', '')}",
            )
        ]
    back = getattr(other, field_name, None)
    back_model = (_names(getattr(back, "model", None)) or [target_model])[0]
    back_boundary = getattr(back, "boundary", None)
    if (back_model, back_boundary) != (own_model, own_boundary):
        return [
            (
                field_path,
                f"'{field_name}' names '{boundary}' on '{target_model}', whose "
                f"{field_name} is to '{back_boundary}' on '{back_model}', not back "
                f"to '{own_boundary}' on '{own_model}'",
            )
        ]
    mismatch = getattr(instance, "connection_mismatch", None)
    if mismatch is not None and (own_model, own_boundary) < (target_model, boundary):
        message = mismatch(other, own_boundary, boundary)
        if message:
            return [(field_path, message)]
    return []
