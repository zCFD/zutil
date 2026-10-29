from contextlib import contextmanager
from contextvars import ContextVar
from typing import Annotated, Any, Dict, Iterable, Mapping, Optional, Tuple, Literal
import math
import os
import re

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo


def zCFD_Field(
    *args,
    advanced: bool = False,
    hidden: bool = False,
    updatable: bool = False,
    units: Optional[str] = None,
    group: Optional[str] = None,
    placeholder: Optional[str] = None,
    variants: Optional[list] = None,
    options: Optional[Mapping[Any, str]] = None,
    **kwargs,
) -> FieldInfo:
    """Extend pydantic Field() with zCFD-specific metadata.

    Adds a ``zcfd`` key to ``json_schema_extra`` containing:
      - advanced:    True if this field should be hidden behind an "Advanced" toggle in the UI.
      - hidden:      True if the reference leaves this field out; pair it with
        ``SkipJsonSchema`` so the UI leaves it out too.
      - updatable:   True if this parameter can be dynamically updated at solver runtime.
      - units:       Physical units of the value (e.g. "rad/s", "m", "Pa").
      - group:       Named group the field renders under within its panel.
      - placeholder: Placeholder text for empty inputs/selects.
      - variants:    For a block whose version is decided by other settings
        in the deck: an ordered list of ``(when, model_cls)`` rows. The
        validator checks the block against the first row whose conditions
        hold, and the rows are serialised as ``[{"when": {path: value,
        ...}, "ref": "#/$defs/<ClassName>"}, ...]`` for the UI. See
        :mod:`zutil.validate.schemas.variants`. On a ``Dict[str, Model]``
        field each entry is checked against the chosen version.
      - options:     For a choice field, a one-sentence definition of each
        value, keyed by the value. The reference, zui dropdown help and
        validation messages read it; every key must be one of the field's
        values.

    Display names use pydantic's native ``title=`` kwarg (pass it through
    ``kwargs``); only set it where the auto-generated title reads badly.

    The metadata is merged into any existing ``zcfd`` entry on the JSON
    schema node rather than replacing it, so it composes with
    annotation-level contributions such as ``ConditionalOptional``'s
    ``applies_when`` (which pydantic applies before field-level extras).
    """
    existing_extra = kwargs.pop("json_schema_extra", None)
    zcfd_extra: Dict[str, Any] = {"advanced": advanced, "updatable": updatable}
    if hidden:
        zcfd_extra["hidden"] = True
    if units is not None:
        zcfd_extra["units"] = units
    if group is not None:
        zcfd_extra["group"] = group
    if placeholder is not None:
        zcfd_extra["placeholder"] = placeholder
    if options is not None:
        zcfd_extra["options"] = {str(k): v for k, v in options.items()}
    table = None
    if variants is not None:
        from .variants import Variants

        table = Variants(variants)
        zcfd_extra["variants"] = table.json_rows()

    def _merge_extra(js: Dict[str, Any]) -> None:
        if isinstance(existing_extra, dict):
            for key, value in existing_extra.items():
                if key == "zcfd" and isinstance(value, dict):
                    js.setdefault("zcfd", {}).update(value)
                else:
                    js[key] = value
        elif callable(existing_extra):
            existing_extra(js)
        js.setdefault("zcfd", {}).update(zcfd_extra)

    kwargs["json_schema_extra"] = _merge_extra
    field = Field(*args, **kwargs)
    if table is not None:
        field.metadata.append(table)
    return field


def zcfd_metadata(field: FieldInfo) -> Dict[str, Any]:
    """The ``zcfd`` entry ``field`` writes into its JSON-schema node.

    Covers what :func:`zCFD_Field` and ``json_schema_extra`` declare
    (``advanced``, ``updatable``, ``units``, ...). Annotation markers such as
    :class:`AppliesWhen` live on ``field.metadata`` and are not included.
    """
    extra = field.json_schema_extra
    if callable(extra):
        schema: Dict[str, Any] = {}
        extra(schema)
        extra = schema
    return dict((extra or {}).get("zcfd", {}))


class _Unset:
    """Sentinel allowed-value for :class:`When`: the clause also holds when
    the referenced setting is absent from the configuration (i.e. the
    solver's default applies).  Serialises to ``null`` in the JSON schema —
    which is exactly what a UI sees when looking up an unset key."""

    _instance = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


class _Set:
    """Sentinel allowed-value for :class:`When`: the clause holds when the
    referenced setting is present with any non-null value — "applies when
    <setting> is configured at all". Serialises to the string
    ``"__present__"`` in the JSON schema."""

    _instance = None

    def __new__(cls) -> "_Set":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SET"


SET = _Set()


class NotIn:
    """Allowed-values form for :class:`When`: the condition holds when the
    referenced setting is anything but these values, including unset.
    Serialises to ``{"not": [...]}`` in the JSON schema."""

    __slots__ = ("values",)

    def __init__(self, *values):
        if not values:
            raise ValueError("NotIn() requires at least one value")
        self.values = tuple(values)

    def __eq__(self, other) -> bool:
        return isinstance(other, NotIn) and self.values == other.values

    def __hash__(self) -> int:
        return hash(("NotIn", repr(self.values)))

    def __repr__(self) -> str:
        return f"NotIn{self.values!r}"


NONZERO = NotIn(UNSET, 0.0, [0.0, 0.0, 0.0])


def deck_spelling(name: str) -> str:
    return name.replace("_", " ")


def _is_zero(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_is_zero(v) for v in value)
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0
    )


def _not_in_phrase(values: tuple, show) -> str:
    unset = any(v is UNSET for v in values)
    zero = any(_is_zero(v) for v in values)
    others = [show(v) for v in values if v is not UNSET and not _is_zero(v)]
    if not unset and not zero:
        verb = "is not" if len(others) == 1 else "is none of"
        return f"{verb} {', '.join(others)}"
    parts = ["set"] if unset else []
    if zero:
        parts.append("non-zero")
    if others:
        parts.append(("not " if len(others) == 1 else "none of ") + ", ".join(others))
    return "is " + " and ".join(parts)


def _condition_phrases(conditions: Mapping[str, Any]) -> list:
    def show(v):
        return "unset" if v is UNSET else "set" if v is SET else repr(v)

    phrases = []
    for path, allowed in conditions.items():
        for anchor in ("<self>.", "<parent>."):
            if path.startswith(anchor):
                path = path[len(anchor) :]
        path = deck_spelling(path)
        if isinstance(allowed, NotIn):
            phrases.append(f"{path} {_not_in_phrase(allowed.values, show)}")
            continue
        shown = [show(v) for v in allowed]
        verb = "is" if len(shown) == 1 else "is one of"
        phrases.append(f"{path} {verb} {', '.join(shown)}")
    return phrases


class _Clause:
    """Shared wording for :class:`When` and :class:`AnyOf`."""

    __slots__ = ()

    @property
    def uses_self_anchor(self) -> bool:
        return "<self>" in self._anchors()

    @property
    def uses_parent_anchor(self) -> bool:
        return "<parent>" in self._anchors()

    hint: Optional[str] = None

    def _with_hint(self, sentence: str) -> str:
        return sentence if self.hint is None else f"{sentence}. {self.hint}"

    def describe(self, field_name: str, value: Any = UNSET) -> str:
        name = deck_spelling(field_name)
        subject = f"'{name}'" if value is UNSET else f"'{name}': {value!r}"
        return self._with_hint(f"{subject} only applies when {self.condition_text()}")

    def describe_required(self, field_name: str) -> str:
        return self._with_hint(
            f"'{deck_spelling(field_name)}' is required when {self.condition_text()}"
        )

    def describe_missing(self, field_name: str) -> str:
        """The error for a required setting left unset, offering any sibling that would do instead."""
        conditions = getattr(self, "conditions", {})
        sibling = {
            path: path.split(".", 1)[1]
            for path, allowed in conditions.items()
            if path.startswith("<self>.") and allowed == (UNSET,)
        }
        if not sibling:
            return self.describe_required(field_name)
        siblings = list(sibling.values())
        rest = {
            path: allowed for path, allowed in conditions.items() if path not in sibling
        }
        names = " or ".join(f"'{deck_spelling(n)}'" for n in [field_name, *siblings])
        if not rest:
            return f"Set {names}."
        return f"Set {names}; one is required when {' and '.join(_condition_phrases(rest))}."


class When(_Clause):
    """A setting's applicability clause, as data: the machine-readable form
    of the manual sentence "only used when <setting> is <one of these>".

    ``conditions`` maps root-anchored dotted paths into the v2 parameters
    dict (e.g. ``"solver.convergence_control.scheme.name"``) to the values
    for which the annotated setting applies.  All conditions must hold.

    Allowed-value sentinels:
      - :data:`UNSET` — the clause also holds while the referenced setting
        is left unset (defaults apply).
      - :data:`SET` — the clause holds for any present, non-null value
        ("applies when <setting> is configured at all").
      - :class:`NotIn` in place of the allowed values — the clause holds
        for any value but those, unset included.
      - :data:`NONZERO` — the clause holds when the referenced setting is
        present and is neither zero nor a zero vector ("applies when
        gravity is on").

    Path features:
      - a ``*`` segment matches any key at that level; the condition holds
        if ANY match yields an allowed value (existential), and under
        :class:`NotIn` only if no match is one of the excluded values.
      - a leading ``<mesh>`` anchor resolves to the model block containing
        the annotated field (``model.<name>``), for same-mesh conditions
        like "this mesh has an overset boundary". A clause with a
        ``<mesh>`` anchor on a field outside any model block never holds.
      - a leading ``<self>`` anchor resolves to the model the annotated
        field belongs to, for sibling conditions that must follow the
        block wherever it sits (solver level or a per-mesh override).
      - a leading ``<parent>`` anchor resolves to the nearest model that
        contains the annotated field's model, for a sub-block setting
        that depends on the block it sits in (a wall's roughness on the
        wall's kind). A clause with a ``<parent>`` anchor on a field of
        the root model never holds.

    Because it is plain data it serialises into the JSON schema
    (``zcfd.applies_when``), so the UI hides the setting client-side, and
    the same clause drives the validation error when a config sets the
    field anyway — the rule is stated exactly once.  :class:`AnyOf` joins
    several clauses where one conjunction cannot state the rule.
    """

    __slots__ = ("conditions", "text", "hint")

    def __init__(
        self,
        conditions: Mapping[str, Iterable],
        text: Optional[str] = None,
        hint: Optional[str] = None,
    ):
        self.text = text
        self.hint = hint
        if not conditions:
            raise ValueError("When() requires at least one condition")
        normalised = {}
        for path, allowed in conditions.items():
            if not isinstance(path, str) or not path:
                raise ValueError(
                    f"When() condition path must be a non-empty string, got {path!r}"
                )
            if isinstance(allowed, Mapping) and set(allowed) == {"not"}:
                allowed = NotIn(*allowed["not"])
            if isinstance(allowed, NotIn):
                normalised[path] = allowed
                continue
            values = (
                tuple(allowed)
                if isinstance(allowed, (list, tuple, set, frozenset))
                else (allowed,)
            )
            if not values:
                raise ValueError(f"When() condition for '{path}' has no allowed values")
            normalised[path] = values
        self.conditions = normalised

    def _anchors(self) -> set:
        return {path.split(".")[0] for path in self.conditions}

    def to_schema(self) -> Dict[str, Any]:
        def encode(v):
            if v is UNSET:
                return None
            if v is SET:
                return "__present__"
            return v

        return {
            path: (
                {"not": [encode(v) for v in values.values]}
                if isinstance(values, NotIn)
                else [encode(v) for v in values]
            )
            for path, values in self.conditions.items()
        }

    @staticmethod
    def _value_matches(node: Any, allowed: Any) -> bool:
        if isinstance(allowed, NotIn):
            return node not in allowed.values
        if node is UNSET:
            return UNSET in allowed
        if SET in allowed and node is not None:
            return True
        return node in allowed

    @classmethod
    def _matches(cls, node: Any, segments: list, allowed: tuple) -> bool:
        if not segments:
            return cls._value_matches(node, allowed)
        key, rest = segments[0], segments[1:]
        if key == "*":
            if not isinstance(node, Mapping):
                return cls._value_matches(UNSET, allowed)
            matches = (cls._matches(child, rest, allowed) for child in node.values())
            return all(matches) if isinstance(allowed, NotIn) else any(matches)
        if isinstance(node, Mapping) and key in node:
            return cls._matches(node[key], rest, allowed)
        return cls._value_matches(UNSET, allowed)

    def evaluate(
        self,
        root: Mapping,
        mesh_anchor: "Optional[str]" = None,
        self_node: "Optional[Mapping]" = None,
        parent_node: "Optional[Mapping]" = None,
    ) -> bool:
        """True if the annotated setting applies for the given root dict.

        A missing path means the referenced setting is not present in this
        configuration — either unset (defaults apply; holds iff the clause
        allows :data:`UNSET`) or nonexistent in the active union variant
        (the annotated setting does not apply).

        ``mesh_anchor`` is the root-anchored path of the model block the
        annotated field sits under (e.g. ``"model.plate"``), used to
        resolve ``<mesh>``-anchored conditions. ``self_node`` is the dump
        of the model the annotated field belongs to, used to resolve
        ``<self>``-anchored conditions, and ``parent_node`` the dump of
        the model containing that one, for ``<parent>``-anchored ones.
        """
        for path, allowed in self.conditions.items():
            segments = path.split(".")
            node = root
            if segments[0] == "<mesh>":
                if mesh_anchor is None:
                    return False
                segments = mesh_anchor.split(".") + segments[1:]
            elif segments[0] in ("<self>", "<parent>"):
                anchored = self_node if segments[0] == "<self>" else parent_node
                if anchored is None:
                    return False
                node, segments = anchored, segments[1:]
            if not self._matches(node, segments, allowed):
                return False
        return True

    def condition_text(self) -> str:
        if self.text is not None:
            return self.text
        return " and ".join(_condition_phrases(self.conditions))

    def __repr__(self) -> str:
        return f"When({self.conditions!r})"


class AnyOf(_Clause):
    """``AnyOf(When({...}), When({...}))`` — a clause that holds when any
    one of its :class:`When` branches holds, for a rule that is not a
    single conjunction ("on a plain wall, or on any wall that is not a
    wall-function wall").

    Usable wherever a :class:`When` is. Serialises to
    ``{"any_of": [<branch>, ...]}`` in the JSON schema; each branch keeps
    the :class:`When` encoding.
    """

    __slots__ = ("branches",)

    def __init__(self, *branches):
        if len(branches) < 2:
            raise ValueError("AnyOf() requires at least two When clauses")
        clauses = []
        for branch in branches:
            if isinstance(branch, Mapping):
                branch = When(branch)
            if not isinstance(branch, When):
                raise TypeError(
                    f"AnyOf() branches must be When clauses, got {branch!r}"
                )
            clauses.append(branch)
        self.branches = tuple(clauses)

    def _anchors(self) -> set:
        return set().union(*(branch._anchors() for branch in self.branches))

    def to_schema(self) -> Dict[str, Any]:
        return {"any_of": [branch.to_schema() for branch in self.branches]}

    def evaluate(
        self,
        root: Mapping,
        mesh_anchor: "Optional[str]" = None,
        self_node: "Optional[Mapping]" = None,
        parent_node: "Optional[Mapping]" = None,
    ) -> bool:
        return any(
            branch.evaluate(root, mesh_anchor, self_node, parent_node)
            for branch in self.branches
        )

    def condition_text(self) -> str:
        first, rest = self.branches[0], self.branches[1:]
        common = {
            path: allowed
            for path, allowed in first.conditions.items()
            if all(branch.conditions.get(path) == allowed for branch in rest)
        }
        remainders = [
            {p: a for p, a in branch.conditions.items() if p not in common}
            for branch in self.branches
        ]
        if not all(remainders):
            common, remainders = {}, [b.conditions for b in self.branches]
        options = " or ".join(
            " and ".join(_condition_phrases(remainder)) for remainder in remainders
        )
        if not common:
            return f"either {options}"
        return f"{' and '.join(_condition_phrases(common))} and either {options}"

    def __repr__(self) -> str:
        return f"AnyOf{self.branches!r}"


def _as_clause(when: Any, owner: str) -> "_Clause":
    if isinstance(when, Mapping):
        when = When(when)
    if not isinstance(when, _Clause):
        raise TypeError(f"{owner} must be a When or AnyOf clause, got {when!r}")
    return when


class AppliesWhen:
    """Annotation marker carrying a :class:`When` or :class:`AnyOf` clause.

    Emits ``zcfd.applies_when`` into the field's JSON schema node and is
    discovered on ``FieldInfo.metadata`` by the generic applicability
    validator (see ``zutil.validate.applicability``).  Authors normally use
    :class:`ConditionalOptional`, :class:`ConditionalRequired`,
    :class:`ConditionalDefault` or :class:`ConditionalChoice` rather than
    this directly.

    With ``values`` the clause gates only those values of the field, not the
    field itself, and is emitted under ``zcfd.applies_when_value`` keyed by
    value.  With ``required`` the field must also be set whenever the clause
    holds, emitted as ``zcfd.required_when_applies``.  With ``written`` the
    field counts as set whenever the config writes it, even at its default.
    """

    __slots__ = ("when", "values", "required", "written", "hint")

    def __init__(
        self,
        when: "_Clause",
        values: "Optional[Tuple[Any, ...]]" = None,
        required: bool = False,
        written: bool = False,
        hint: "Optional[str]" = None,
    ):
        if values is not None and (required or written):
            raise ValueError("a value-gated clause cannot be required or written")
        self.when = when
        self.values = values
        self.required = required
        self.written = written
        self.hint = hint

    def __get_pydantic_json_schema__(self, core_schema, handler):
        js = handler(core_schema)
        zcfd = js.setdefault("zcfd", {})
        if self.values is None:
            zcfd["applies_when"] = self.when.to_schema()
            if self.required:
                zcfd["required_when_applies"] = True
        else:
            gated = zcfd.setdefault("applies_when_value", {})
            for value in self.values:
                gated[value] = self.when.to_schema()
        return js


def _two_parameters(cls, params, shape: str):
    if not isinstance(params, tuple) or len(params) != 2:
        raise TypeError(
            f"{cls.__name__}[...] takes exactly two parameters: {cls.__name__}[{shape}]"
        )
    return params


class ConditionalOptional:
    """``ConditionalOptional[X, When({...})]`` — a setting that only exists
    in some configurations.

    Expands to ``Annotated[Optional[X], AppliesWhen(...)]``: optional by
    construction, hidden by the UI when the clause is false, and rejected by
    validation — with the clause's own wording — if a config sets it while
    inapplicable.
    """

    def __class_getitem__(cls, params):
        typ, when = _two_parameters(cls, params, "Type, When({...})")
        when = _as_clause(when, "ConditionalOptional second parameter")
        return Annotated[Optional[typ], AppliesWhen(when)]


class ConditionalRequired:
    """``ConditionalRequired[X, When({...})]`` — a setting that exists in
    exactly the configurations its clause describes.

    Expands to ``Annotated[Optional[X], AppliesWhen(..., required=True)]``:
    like :class:`ConditionalOptional`, and validation also rejects a config
    that leaves it unset while the clause holds.
    """

    def __class_getitem__(cls, params):
        typ, when = _two_parameters(cls, params, "Type, When({...})")
        when = _as_clause(when, "ConditionalRequired second parameter")
        return Annotated[Optional[typ], AppliesWhen(when, required=True)]


class ConditionalDefault:
    """``ConditionalDefault[X, When({...})]`` — a setting with a default
    that a config may only write in some configurations.

    Expands to ``Annotated[X, AppliesWhen(..., written=True)]``: the type
    and default are kept, and validation rejects a config that writes the
    key while the clause is false, even when it writes the default.
    """

    def __class_getitem__(cls, params):
        typ, when = _two_parameters(cls, params, "Type, When({...})")
        when = _as_clause(when, "ConditionalDefault second parameter")
        return Annotated[typ, AppliesWhen(when, written=True)]


class RequiredWhen:
    """Annotation marker: the field must be set whenever its clause holds.

    Where the field applies is left to its own :class:`AppliesWhen` clause,
    so ``Annotated[ConditionalOptional[X, When(a)], RequiredWhen(When(b))]``
    is optional wherever ``a`` holds and required wherever ``b`` holds.
    Emits ``zcfd.required_when`` into the field's JSON schema node, and is
    enforced by the generic applicability validator.
    """

    __slots__ = ("when",)

    def __init__(self, when: "_Clause"):
        self.when = _as_clause(when, "RequiredWhen parameter")

    def __get_pydantic_json_schema__(self, core_schema, handler):
        js = handler(core_schema)
        js.setdefault("zcfd", {})["required_when"] = self.when.to_schema()
        return js


class Exclusive:
    """Annotation marker: the annotated field and the named sibling fields
    are alternatives, so a config sets at most one of them — exactly one
    with ``required``.

    State it on one member only. Emits ``zcfd.exclusive`` as
    ``{"with": [...], "required": bool}`` into that member's JSON schema
    node, and is enforced by the generic applicability validator.
    """

    __slots__ = ("others", "required")

    def __init__(self, *others: str, required: bool = False):
        if not others:
            raise ValueError("Exclusive() requires at least one other field")
        self.others = tuple(others)
        self.required = required

    def members(self, field_name: str) -> Tuple[str, ...]:
        return (field_name, *self.others)

    def _choices(self, field_name: str) -> str:
        names = [f"'{deck_spelling(n)}'" for n in self.members(field_name)]
        return f"{', '.join(names[:-1])} or {names[-1]}"

    def describe(self, field_name: str) -> str:
        if len(self.others) == 1:
            return f"give either {self._choices(field_name)}, not both"
        return f"give only one of {self._choices(field_name)}"

    def describe_missing(self, field_name: str) -> str:
        if len(self.others) == 1:
            return f"give either {self._choices(field_name)}"
        return f"give one of {self._choices(field_name)}"

    def to_schema(self) -> Dict[str, Any]:
        return {"with": list(self.others), "required": self.required}

    def __get_pydantic_json_schema__(self, core_schema, handler):
        js = handler(core_schema)
        js.setdefault("zcfd", {})["exclusive"] = self.to_schema()
        return js


class Connects:
    __slots__ = ("boundary", "other_model", "reciprocal")

    def __init__(self, *, boundary: bool, other_model: bool, reciprocal: bool = False):
        self.boundary = boundary
        self.other_model = other_model
        self.reciprocal = reciprocal

    def to_schema(self) -> Dict[str, Any]:
        return {
            "model": "model",
            "boundary": "boundary_conditions" if self.boundary else None,
            "other_model": self.other_model,
            "reciprocal": self.reciprocal,
        }

    def __get_pydantic_json_schema__(self, core_schema, handler):
        js = handler(core_schema)
        js.setdefault("zcfd", {})["references"] = self.to_schema()
        return js


class ModelWide:
    """A boundary setting the solver applies to its whole model.

    Every boundary condition on one model that carries the field must give it
    the same value. ``zutil.validate.applicability`` enforces it and the UI
    reads ``zcfd.model_wide``.
    """

    __slots__ = ()

    def __get_pydantic_json_schema__(self, core_schema, handler):
        js = handler(core_schema)
        js.setdefault("zcfd", {})["model_wide"] = True
        return js


class Disjoint:
    """A list setting whose entries no two holders on one model share.

    ``item`` names one entry and ``holder`` the block that carries the list,
    for the message: "Cell zone 17 is in 'cell zones' here and on 'fan'. Each
    cell zone belongs to one rotating fluid zone at most". Every block on one
    model whose field carries the same marker is compared.
    ``zutil.validate.applicability`` enforces it and the UI reads
    ``zcfd.disjoint``.
    """

    __slots__ = ("item", "holder")

    def __init__(self, item: str, holder: str):
        self.item = item
        self.holder = holder

    def describe(self, field_name: str, shared: list, other: str) -> str:
        values = ", ".join(str(v) for v in shared)
        noun, verb = (self.item, "is") if len(shared) == 1 else (f"{self.item}s", "are")
        return (
            f"{noun[0].upper()}{noun[1:]} {values} {verb} in "
            f"'{deck_spelling(field_name)}' here and on '{other}'. Each {self.item} "
            f"belongs to one {self.holder} at most: remove it from one of them."
        )

    def __get_pydantic_json_schema__(self, core_schema, handler):
        js = handler(core_schema)
        js.setdefault("zcfd", {})["disjoint"] = True
        return js


class ConditionalChoice:
    """``ConditionalChoice[X, {value: When({...}), ...}]`` — a setting whose
    listed values only exist in some configurations.

    Expands to ``Annotated[X, AppliesWhen(when, values=(value,)), ...]``: the
    type is kept as given, the UI can withhold each gated value while its
    clause is false, and validation rejects a config that selects it anyway,
    with the clause's own wording. A gated value is rejected even when it is
    the field's default. A gate may be ``(When, hint)``: the hint is appended
    to the rejection to say what to use instead.
    """

    def __class_getitem__(cls, params):
        typ, gates = _two_parameters(cls, params, "Type, {value: When({...})}")
        if not isinstance(gates, Mapping) or not gates:
            raise TypeError(
                f"ConditionalChoice second parameter must be a non-empty mapping of value to When clause, got {gates!r}"
            )
        markers = [
            AppliesWhen(
                _as_clause(
                    when[0] if isinstance(when, tuple) else when,
                    f"ConditionalChoice clause for {value!r}",
                ),
                values=(value,),
                hint=when[1] if isinstance(when, tuple) else None,
            )
            for value, when in gates.items()
        ]
        return Annotated[(typ, *markers)]


class Advises:
    """Annotation marker: a setting that is valid but has no effect while
    its clause holds, such as ``motion`` at zero rate on a model.

    The generic applicability walker reports it as a warning, never an
    error, when the field is set away from its default and the clause
    holds. ``advice`` is the warning text; ``{owner}`` in it is replaced
    by the name of the entry the field sits in. Emits ``zcfd.advises`` as
    ``{"when": clause, "advice": text}`` into the field's JSON schema node,
    with ``{owner}`` left for the form to fill.
    """

    __slots__ = ("when", "advice")

    def __init__(self, when: "_Clause", advice: str):
        self.when = _as_clause(when, "Advises clause")
        self.advice = advice

    def describe(self, owner: str) -> str:
        return self.advice.format(owner=owner)

    def to_schema(self) -> Dict[str, Any]:
        return {"when": self.when.to_schema(), "advice": self.advice}

    def __get_pydantic_json_schema__(self, core_schema, handler):
        js = handler(core_schema)
        js.setdefault("zcfd", {})["advises"] = self.to_schema()
        return js


_UNCHOSEN = object()
_case_directory: ContextVar = ContextVar("case_directory", default=_UNCHOSEN)


@contextmanager
def in_case_directory(directory):
    """Resolve every :func:`case_file` path against ``directory`` while the
    block runs.

    ``None`` turns the file checks off, for callers that validate a deck
    away from its files. Outside any such block the checks are off.
    """
    token = _case_directory.set(None if directory is None else str(directory))
    try:
        yield
    finally:
        _case_directory.reset(token)


@contextmanager
def in_solver_directory():
    """Resolve :func:`case_file` paths against the working directory, which
    is where the solver reads them, unless an enclosing
    :func:`in_case_directory` has already chosen."""
    if _case_directory.get() is _UNCHOSEN:
        with in_case_directory("."):
            yield
    else:
        yield


_VTK_POLYDATA = re.compile(rb'<VTKFile\b[^>]*\btype="PolyData"')
_POINT_DATA = re.compile(rb"<PointData\b[^>]*?(/>|>(.*?)</PointData>)", re.S)
_ARRAY_NAME = re.compile(rb'<DataArray\b[^>]*?\bName="([^"]*)"')


def vtp_point_arrays(path: str) -> "Optional[list]":
    """Names of the point arrays in the first piece of a VTK XML PolyData
    file, read from its XML header; ``None`` if the file is not PolyData."""
    header = b""
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1 << 16)
            header += chunk
            if not chunk or b"</PointData>" in header or b"<AppendedData" in header:
                break
    header = header.split(b"<AppendedData", 1)[0]
    if not _VTK_POLYDATA.search(header):
        return None
    block = _POINT_DATA.search(header)
    if block is None or block.group(2) is None:
        return []
    return [
        name.decode("utf-8", "replace") for name in _ARRAY_NAME.findall(block.group(2))
    ]


def _check_case_file(label: str, point_array: "Optional[str]"):
    def check(path: str) -> str:
        directory = _case_directory.get()
        if directory is None or directory is _UNCHOSEN:
            return path
        full = os.path.join(directory, path)
        if not os.path.isfile(full):
            where = "" if os.path.isabs(path) else f" in {os.path.abspath(directory)}"
            raise ValueError(f'{label} field "{path}" not found{where}')
        if not os.access(full, os.R_OK):
            raise ValueError(f'{label} field "{path}" cannot be read')
        if point_array is None:
            return path
        arrays = vtp_point_arrays(full)
        if arrays is None:
            raise ValueError(
                f'{label} field "{path}" is not a VTK PolyData (.vtp) file'
            )
        if point_array not in arrays:
            found = (
                f"its point arrays are {', '.join(arrays)}"
                if arrays
                else "it has no point arrays"
            )
            raise ValueError(
                f'{label} field "{path}" has no point array named "{point_array}"; {found}'
            )
        return path

    return check


def case_file(label: str, point_array: "Optional[str]" = None):
    """The type of a path to a file the solver reads, relative to the
    directory it runs in (see :func:`in_solver_directory`).

    Validation rejects a path that does not name a readable file, calling
    it the ``<label> field`` in the message. With ``point_array`` it also
    requires a VTK PolyData file whose points carry an array of that name.
    """
    return Annotated[str, AfterValidator(_check_case_file(label, point_array))]


def normalise_vector_with_warning(
    v: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    mag = math.sqrt(sum(x * x for x in v))
    if mag == 0:
        # A zero vector has no direction, so it cannot be normalised.
        # Raising ValueError matches the convention used throughout this
        # module for invalid geometric input (e.g. Vector3d.validate).
        raise ValueError(
            f"Cannot normalise a zero-magnitude vector {tuple(v)}: "
            "a direction vector must be non-zero"
        )
    return (v[0] / mag, v[1] / mag, v[2] / mag)


class Vector3d(list):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.with_info_plain_validator_function(cls.validate)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # The core schema is a plain validator function pydantic can't
        # introspect, so hand it an explicit JSON Schema: a 3-element
        # array of numbers.
        return {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        }

    @classmethod
    def validate(cls, v, info):
        if isinstance(v, cls):
            v = list(v)

        # Extract field path from validation context if available
        field_name = (
            getattr(info, "field_name", "unknown_field") if info else "unknown_field"
        )

        if not isinstance(v, (list, tuple)):
            raise ValueError(f"'{field_name}' must be 3 numbers, e.g. [1.0, 0.0, 0.0]")
        if len(v) != 3:
            raise ValueError(f"'{field_name}' must be exactly 3 numbers, not {len(v)}")
        return cls([float(x) for x in v])

    @property
    def as_tuple(self) -> Tuple[float, float, float]:
        return tuple(self)


class NormalisedVector3d(Vector3d):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        # Chain the parent validator (Vector3d.validate) with normalize
        return core_schema.with_info_after_validator_function(
            cls.normalize, super().__get_pydantic_core_schema__(source_type, handler)
        )

    @classmethod
    def normalize(cls, v, info):
        nv = normalise_vector_with_warning(tuple(v))
        return cls(nv)


class IntVector3d(list):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.with_info_plain_validator_function(cls.validate)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # See Vector3d.__get_pydantic_json_schema__ — same rationale, but
        # the elements are integers rather than floats.
        return {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3,
        }

    @classmethod
    def validate(cls, v, info):
        if isinstance(v, cls):
            v = list(v)

        # Extract field path from validation context if available
        field_name = (
            getattr(info, "field_name", "unknown_field") if info else "unknown_field"
        )

        if not isinstance(v, (list, tuple)):
            raise ValueError(f"'{field_name}' must be 3 whole numbers, e.g. [1, 0, 0]")
        if len(v) != 3:
            raise ValueError(
                f"'{field_name}' must be exactly 3 whole numbers, not {len(v)}"
            )
        return cls([int(x) for x in v])


_VELOCITY_TYPE = (
    "How the wall moves: `linear` translates it along a vector, `rotating` "
    "turns it about an axis."
)


class LinearVelocity(BaseModel):
    """Linear velocity specification."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["linear"] = Field(..., description=_VELOCITY_TYPE)
    vector: Vector3d = Field(..., description="Direction of linear velocity")
    mach: Optional[float] = Field(
        None,
        description="Mach number for linear velocity - overrides magnitude of the translation velocity",
    )


class Rotation(BaseModel):
    """A rigid rotation about an axis."""

    model_config = ConfigDict(extra="forbid")

    axis: NormalisedVector3d = zCFD_Field(
        ..., title="Axis", description="Axis of rotation."
    )
    origin: Vector3d = zCFD_Field(
        ...,
        title="Origin",
        units="m",
        description="A point on the axis of rotation.",
    )
    omega: float = zCFD_Field(
        ...,
        title="Omega",
        units="rad/s",
        description="Angular velocity, positive right-handed about 'axis'.",
    )


class RotationalVelocity(Rotation):
    """Rotational velocity specification."""

    type: Literal["rotating"] = Field(..., description=_VELOCITY_TYPE)
