"""
Schema introspection for the reference generator: reader-facing facts,
union resolution and fingerprints, read from the canonical pydantic models.
"""

import collections.abc
import hashlib
import json
import sys
import typing
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import annotated_types as at
import pydantic_core
from pydantic import BaseModel, Tag
from pydantic.fields import FieldInfo
from pydantic.types import Discriminator

from zutil.validate.schemas.base_types import (
    AppliesWhen,
    Exclusive,
    IntVector3d,
    NormalisedVector3d,
    RequiredWhen,
    Vector3d,
)


class UnresolvedUnionError(ValueError):
    """A union member the generator cannot place under a tag or routing condition."""

    def __init__(self, location: str, members: List[str], reason: str) -> None:
        self.location = location
        self.members = members
        self.reason = reason
        super().__init__(
            f"{location}: cannot resolve union member(s) {', '.join(members)} ({reason})"
        )


@dataclass
class Variant:
    """One leaf model of a union, with the condition that selects it."""

    model: type
    tag: Optional[str] = None
    when: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Shape:
    """What a field's value is: a scalar, one block, or a union of blocks, possibly in a container."""

    kind: str
    container: Optional[str] = None
    model: Optional[type] = None
    discriminator: Optional[str] = None
    variants: List[Variant] = field(default_factory=list)
    unresolved: Optional[UnresolvedUnionError] = None


def is_model(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def unwrap(annotation: Any) -> Tuple[Any, list]:
    """Peel Annotated wrappers and Optional, returning the core type and all metadata."""
    meta: list = []
    core = annotation
    while True:
        if get_origin(core) is typing.Annotated or hasattr(core, "__metadata__"):
            meta.extend(getattr(core, "__metadata__", ()))
            core = get_args(core)[0]
            continue
        if get_origin(core) is Union:
            args = [a for a in get_args(core) if a is not type(None)]
            if len(args) == 1 and len(args) != len(get_args(core)):
                core = args[0]
                continue
            if len(args) != len(get_args(core)):
                core = Union[tuple(args)]
        return core, meta


def literal_values(annotation: Any) -> Optional[list]:
    core, _ = unwrap(annotation)
    if get_origin(core) is Literal:
        return list(get_args(core))
    if get_origin(core) is Union:
        values = [v for m in get_args(core) for v in literal_values(m) or []]
        return values or None
    return None


def zcfd_extra(field_info: FieldInfo) -> dict:
    jse = field_info.json_schema_extra
    js: dict = {}
    if callable(jse):
        jse(js)
    elif isinstance(jse, dict):
        js = jse
    return js.get("zcfd", {}) or {}


def option_definitions(field_info: FieldInfo) -> Dict[str, str]:
    """The schema's one-sentence definition of each value of a choice field, keyed by the value as text."""
    return dict(zcfd_extra(field_info).get("options") or {})


def unknown_option_definitions(model: type) -> List[dict]:
    """Definitions a model's own fields give for values their type does not allow."""
    out = []
    for key in own_fields(model):
        fi = model.model_fields[key]
        allowed = {str(v) for v in literal_values(fi.annotation) or []}
        for value in option_definitions(fi):
            if value not in allowed:
                out.append({"model": model.__name__, "key": key, "option": value})
    return out


def applies_when(field_info: FieldInfo, key: str) -> Optional[dict]:
    metadata = list(field_info.metadata) + unwrap(field_info.annotation)[1]
    clause = next((m for m in metadata if isinstance(m, AppliesWhen)), None)
    exclusive = next((m for m in metadata if isinstance(m, Exclusive)), None)
    required_when = next((m for m in metadata if isinstance(m, RequiredWhen)), None)
    if clause is None and exclusive is None and required_when is None:
        return None
    facts: dict = {"conditions": clause.when.to_schema() if clause else None}
    sentences = []
    if clause is not None and clause.required:
        facts["required"] = True
        sentences.append(f"{clause.when.describe_required(key)}, and only applies then")
    elif required_when is not None:
        facts["required_when"] = required_when.when.to_schema()
        sentence = required_when.when.describe_required(key)
        if clause is not None:
            sentence += f", and only applies when {clause.when.condition_text()}"
        sentences.append(sentence)
    elif clause is not None:
        sentences.append(clause.when.describe(key))
    if exclusive is not None:
        facts["exclusive"] = exclusive.to_schema()
        sentences.append(
            exclusive.describe_missing(key)
            if exclusive.required
            else exclusive.describe(key)
        )
    facts["text"] = "; ".join(sentences)
    return facts


def _is_callable(tp: Any) -> bool:
    return (
        tp is collections.abc.Callable
        or get_origin(tp) is collections.abc.Callable
        or tp is typing.Callable
    )


def _scalar_type(core: Any, key: str) -> Optional[str]:
    if isinstance(core, type) and issubclass(core, NormalisedVector3d):
        return "unit vector (3 floats)"
    if isinstance(core, type) and issubclass(core, Vector3d):
        return "vector (3 floats)"
    if isinstance(core, type) and issubclass(core, IntVector3d):
        return "3 ints"
    if get_origin(core) is Literal:
        return "choice"
    if core is bool:
        return "bool"
    if core is int:
        return "int"
    if core is float:
        return "float"
    if core is str:
        return "str"
    if core is Any:
        return "value"
    if _is_callable(core):
        return "Python function"
    if core is dict or get_origin(core) is dict:
        args = get_args(core)
        if args and any(is_model(unwrap(a)[0]) for a in args[1:]):
            return None
        return "settings block"
    if get_origin(core) in (list, tuple) or core in (list, tuple):
        args = [a for a in get_args(core) if a is not Ellipsis]
        inner = unwrap(args[0])[0] if args else None
        if inner is int:
            return "list of zone ids" if key in ("zone", "zones") else "list of int"
        if inner is float:
            return "list of float"
        if inner is str:
            return "list of str"
        if inner is not None and (is_model(inner) or _union_of_models(inner)):
            return None
        return "list"
    return None


def _union_of_models(core: Any) -> bool:
    if get_origin(core) is not Union:
        return False
    return any(
        is_model(unwrap(a)[0]) or _union_of_models(unwrap(a)[0]) for a in get_args(core)
    )


def type_plain(annotation: Any, key: str) -> str:
    """The reader-facing type of a field."""
    core, _ = unwrap(annotation)
    scalar = _scalar_type(core, key)
    if scalar is not None:
        return scalar
    if is_model(core):
        return "settings block"
    origin = get_origin(core)
    if origin is dict:
        return "named settings blocks"
    if origin in (list, tuple):
        return "list of settings blocks"
    if origin is Union:
        members = [unwrap(a)[0] for a in get_args(core)]
        if all(is_model(m) or _union_of_models(m) for m in members):
            return "choice of settings blocks"
        parts: List[str] = []
        for m in get_args(core):
            p = type_plain(m, key)
            if p not in parts:
                parts.append(p)
        return " or ".join(parts)
    return "value"


def accepts_python_function(annotation: Any) -> bool:
    core, _ = unwrap(annotation)
    if _is_callable(core):
        return True
    return any(
        accepts_python_function(a)
        for a in get_args(core)
        if a is not Ellipsis and not isinstance(a, (list, str))
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return json_safe(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if callable(value):
        return {"python": getattr(value, "__name__", repr(value))}
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def default_of(field_info: FieldInfo) -> Tuple[bool, Any]:
    """(has_genuine_default, JSON-safe default value)."""
    if field_info.default_factory is not None:
        try:
            value = field_info.default_factory()
        except TypeError:
            return False, None
        return True, json_safe(value)
    if field_info.default is pydantic_core.PydanticUndefined:
        return False, None
    return field_info.default is not None, json_safe(field_info.default)


def selected_tag(field_info: FieldInfo, key: str) -> Any:
    """The ``key`` value of a union field's genuine default block, else None."""
    if field_info.default_factory is not None:
        try:
            value = field_info.default_factory()
        except TypeError:
            return None
    else:
        value = field_info.default
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None) if is_model(type(value)) else None


def _bounds_in(metadata: list) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for meta in metadata:
        if isinstance(meta, FieldInfo):
            out.update(_bounds_in(list(meta.metadata)))
            continue
        for name in ("gt", "ge", "lt", "le"):
            cls = {"gt": at.Gt, "ge": at.Ge, "lt": at.Lt, "le": at.Le}[name]
            if isinstance(meta, cls):
                out[name] = json_safe(getattr(meta, name))
            elif (
                isinstance(meta, at.Interval) and getattr(meta, name, None) is not None
            ):
                out[name] = json_safe(getattr(meta, name))
    return out


def bounds(field_info: FieldInfo) -> Dict[str, Any]:
    """Declared bounds, including those on the one bounded number of a number-or-block union."""
    core, meta = unwrap(field_info.annotation)
    out = _bounds_in(list(field_info.metadata) + meta)
    if out or get_origin(core) is not Union:
        return out
    members = [unwrap(m) for m in get_args(core)]
    scalars = [_bounds_in(m_meta) for m_core, m_meta in members if not is_model(m_core)]
    bounded = [b for b in scalars if b]
    return bounded[0] if len(bounded) == 1 else {}


def bounds_text(b: Dict[str, Any], units: Optional[str]) -> Optional[str]:
    lo = ("ge", b["ge"]) if "ge" in b else ("gt", b["gt"]) if "gt" in b else None
    hi = ("le", b["le"]) if "le" in b else ("lt", b["lt"]) if "lt" in b else None
    suffix = f" {units}" if units else ""
    if lo and hi and lo[0] == "ge" and hi[0] == "le":
        return f"{lo[1]} to {hi[1]}{suffix}"
    parts = []
    if lo:
        parts.append(f"{'>=' if lo[0] == 'ge' else '>'} {lo[1]}{suffix}")
    if hi:
        parts.append(f"{'<=' if hi[0] == 'le' else '<'} {hi[1]}{suffix}")
    return " and ".join(parts) if parts else None


def own_annotations(cls: type) -> set:
    """Names annotated in the class body itself, excluding inherited ones."""
    if sys.version_info >= (3, 14):
        import annotationlib

        return set(
            annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF)
        )
    return set(vars(cls).get("__annotations__", {}))


def declares(cls: type, key: str) -> bool:
    return is_model(cls) and key in own_annotations(cls)


def declaring_class(model: type, key: str) -> type:
    """The class in the MRO whose own annotations declare the field."""
    for cls in model.__mro__:
        if declares(cls, key):
            return cls
    return model


def own_fields(model: type) -> List[str]:
    own = own_annotations(model)
    return [
        k
        for k, fi in model.model_fields.items()
        if k in own and not zcfd_extra(fi).get("hidden")
    ]


def _member_tag(member: Any) -> Optional[str]:
    for m in getattr(member, "__metadata__", ()) or ():
        if isinstance(m, Tag):
            return m.tag
    return None


def _declared_discriminator(meta: list) -> Optional[str]:
    for m in meta:
        if isinstance(m, FieldInfo) and isinstance(m.discriminator, str):
            return m.discriminator
        if (
            isinstance(m, FieldInfo)
            and isinstance(m.discriminator, Discriminator)
            and isinstance(m.discriminator.discriminator, str)
        ):
            return m.discriminator.discriminator
        if isinstance(m, Discriminator) and isinstance(m.discriminator, str):
            return m.discriminator
    return None


def _literal_tag(model: type, disc: str) -> Optional[str]:
    fi = model.model_fields.get(disc)
    if fi is None:
        return None
    values = literal_values(fi.annotation)
    if values and len(values) == 1:
        return values[0]
    return None


def resolve_union(
    annotation: Any, location: str, routed: Optional[list] = None
) -> Tuple[Optional[str], List[Variant]]:
    """Leaf variants of a union of blocks.

    A union resolves through a declared discriminator (``Tag`` members or a
    ``Literal`` tag on each member), through ``zcfd.variants`` routing, or,
    lacking both, through a single-valued ``Literal`` field shared by every
    member.  A member that none of these places raises UnresolvedUnionError.
    """
    core, meta = unwrap(annotation)
    members = (
        [m for m in get_args(core) if m is not type(None)]
        if get_origin(core) is Union
        else [annotation]
    )
    disc = _declared_discriminator(meta)
    if disc is not None:
        leaves: List[Variant] = []
        bad = []
        for member in members:
            inner, inner_meta = unwrap(member)
            tag = _member_tag(member)
            if is_model(inner):
                tag = tag if tag is not None else _literal_tag(inner, disc)
                if tag is None:
                    bad.append(inner.__name__)
                    continue
                leaves.append(Variant(inner, tag, {disc: tag}))
            elif get_origin(inner) is Union and tag is not None:
                sub_disc, sub = resolve_union(member, f"{location}[{tag}]")
                for leaf in sub:
                    leaves.append(Variant(leaf.model, tag, {disc: tag, **leaf.when}))
            else:
                bad.append(_member_name(member))
        if bad:
            raise UnresolvedUnionError(location, bad, f"no '{disc}' tag")
        return disc, leaves
    model_members = [
        unwrap(m)[0]
        for m in members
        if is_model(unwrap(m)[0]) or _union_of_models(unwrap(m)[0])
    ]
    if routed:
        by_model: Dict[type, dict] = {}
        for entry_when, cls in routed:
            if cls not in by_model:
                by_model[cls] = {k: v for k, v in entry_when.items() if v is not None}
        missing = [
            _member_name(m) for m in model_members if is_model(m) and m not in by_model
        ]
        missing += [_member_name(m) for m in model_members if not is_model(m)]
        if missing:
            raise UnresolvedUnionError(
                location, missing, "not named by zcfd.variants routing"
            )
        return None, [
            Variant(cls, None, when)
            for cls, when in by_model.items()
            if cls in model_members
        ]
    shared = _shared_literal(model_members)
    bad = [_member_name(m) for m in model_members if not is_model(m)]
    if shared is None or bad:
        names = bad or [_member_name(m) for m in model_members]
        raise UnresolvedUnionError(
            location,
            names,
            "undiscriminated union: no declared discriminator, no routing and no shared single-valued Literal tag",
        )
    return shared, [
        Variant(m, _literal_tag(m, shared), {shared: _literal_tag(m, shared)})
        for m in model_members
    ]


def _shared_literal(models: list) -> Optional[str]:
    candidates = None
    for m in models:
        if not is_model(m):
            continue
        keys = {k for k in m.model_fields if _literal_tag(m, k) is not None}
        candidates = keys if candidates is None else candidates & keys
    if not candidates:
        return None
    for preferred in ("type", "kind", "name"):
        if preferred in candidates:
            tags = [_literal_tag(m, preferred) for m in models if is_model(m)]
            if len(set(tags)) == len(tags):
                return preferred
    return None


def _member_name(member: Any) -> str:
    inner, _ = unwrap(member)
    if is_model(inner):
        return inner.__name__
    names = [_member_name(a) for a in get_args(inner) if a is not type(None)]
    return "Union[" + ", ".join(names) + "]" if names else repr(member)


def routed_variants(
    field_info: FieldInfo, owner: Optional[type] = None
) -> Optional[list]:
    """The (when, model) pairs of a context-routed field, resolved from its zcfd.variants refs."""
    raw = zcfd_extra(field_info).get("variants")
    if not raw:
        return None
    core, _ = unwrap(field_info.annotation)
    if get_origin(core) is dict:
        core = unwrap(get_args(core)[1])[0]
    pool = {}
    for m in get_args(core) if get_origin(core) is Union else (core,):
        inner = unwrap(m)[0]
        if is_model(inner):
            pool[inner.__name__] = inner
    pairs = []
    for entry in raw:
        name = entry["ref"].rsplit("/", 1)[-1]
        if name in pool:
            pairs.append((entry["when"], pool[name]))
    return pairs


def shape_of(field_info: FieldInfo, location: str) -> Shape:
    """Classify a field's value, resolving any union of blocks."""
    annotation = field_info.annotation
    core, meta = unwrap(annotation)
    container = None
    if get_origin(core) is dict and len(get_args(core)) == 2:
        value = get_args(core)[1]
        inner = unwrap(value)[0]
        if is_model(inner) or _union_of_models(inner):
            container, annotation = "named", value
            core, meta = unwrap(value)
    elif get_origin(core) in (list, tuple) and get_args(core):
        value = get_args(core)[0]
        inner = unwrap(value)[0]
        if is_model(inner) or _union_of_models(inner):
            container, annotation = "list", value
            core, meta = unwrap(value)
    if is_model(core):
        return Shape("block", container, core)
    if _union_of_models(core):
        members = [unwrap(m)[0] for m in get_args(core)]
        if (
            not all(is_model(m) or _union_of_models(m) for m in members)
            and _declared_discriminator(meta) is None
        ):
            models = [m for m in members if is_model(m)]
            if len(models) == 1:
                return Shape("block", container, models[0])
        try:
            disc, leaves = resolve_union(
                annotation, location, routed_variants(field_info)
            )
        except UnresolvedUnionError as err:
            return Shape("union", container, None, None, [], err)
        return Shape("union", container, None, disc, leaves)
    return Shape("scalar", container)


def facts_of(field_info: FieldInfo, key: str, discriminator: bool = False) -> dict:
    extra = zcfd_extra(field_info)
    required = field_info.is_required()
    genuine, default = default_of(field_info)
    fixed = discriminator and len(literal_values(field_info.annotation) or []) == 1
    b = bounds(field_info)
    units = extra.get("units")
    shape = shape_of(field_info, key)
    return {
        "type": type_plain(field_info.annotation, key),
        "required": required,
        "default": {"value": default, "show": genuine and not required and not fixed},
        "units": units,
        "bounds": b or None,
        "bounds_text": bounds_text(b, units),
        "advanced": bool(extra.get("advanced", False)),
        "updatable": bool(extra.get("updatable", False)),
        "group": extra.get("group"),
        "applies_when": applies_when(field_info, key),
        "discriminator": discriminator,
        "accepts_python_function": accepts_python_function(field_info.annotation),
        "container": shape.container,
        "overridable_per_model": False,
    }


def _fingerprinted_rules(aw: Optional[dict]) -> Any:
    if not aw:
        return None
    rules = {k: v for k, v in aw.items() if k != "text"}
    return aw["conditions"] if set(rules) == {"conditions"} else rules


def fingerprint(field_info: FieldInfo, key: str) -> str:
    """Hash of the schema facts a side-file entry describes: type, options, default, units, bounds, When."""
    _, default = default_of(field_info)
    aw = applies_when(field_info, key)
    facts = {
        "type": type_plain(field_info.annotation, key),
        "options": json_safe(literal_values(field_info.annotation)),
        "default": default,
        "units": zcfd_extra(field_info).get("units"),
        "bounds": bounds(field_info),
        "when": _fingerprinted_rules(aw),
    }
    blob = json.dumps(facts, sort_keys=True, default=repr).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
