"""Choose which version of a settings block a deck uses.

Many blocks come in several versions, and the version is decided by a
setting elsewhere in the deck. The numerical scheme has one version per
solver and equation type, a reference condition one per equation type,
and time settings one per time-stepping kind. Pydantic only chooses a
version from a key inside the block itself, so these blocks list their
versions as a table instead::

    NUMERICAL_SCHEME_VARIANTS = [
        ({"solver.equations.type": "euler",
          "solver.solver_settings.type": "compressible"},
         FVCompressibleEulerNumericalScheme),
        ...
    ]

and pass it to the field with ``zCFD_Field(variants=...)``. Each row is
a set of conditions on the deck and the version that applies when all of
them hold. Paths start at the top of the deck, ``None`` means the
setting is left unset, and an empty set of conditions always holds. The
first row that holds wins. A row may name :class:`NotAvailable` instead
of a version, for a combination the solver cannot run.

The same table is written into the JSON schema as ``zcfd.variants``, so
zui shows the form for the version the validator checks against.

While a deck validates, :class:`VariantRoot` keeps it available to every
block inside it. A block validated on its own, outside any deck, has no
deck to read and is refused.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError, model_validator
from pydantic_core import (
    ErrorDetails,
    InitErrorDetails,
    PydanticCustomError,
    core_schema,
)

from .base_types import UNSET, When, deck_spelling

_DECK: ContextVar = ContextVar("zcfd_deck_being_validated", default=None)


class NotAvailable:
    """A table row for a combination of settings the solver cannot run."""

    def __init__(self, reason: str):
        self.reason = reason


Row = Tuple[Mapping[str, Any], Union[type, NotAvailable]]


@contextmanager
def deck_being_validated(deck: Mapping):
    """Make ``deck`` the deck that every block validated inside reads from."""
    token = _DECK.set(_plain(deck))
    try:
        yield
    finally:
        _DECK.reset(token)


class VariantRoot(BaseModel):
    """Base for the top of a deck: keeps the deck available while it validates."""

    @model_validator(mode="wrap")
    @classmethod
    def _keep_deck_available(cls, data, handler):
        if _DECK.get() is not None or not isinstance(data, Mapping):
            return handler(data)
        with deck_being_validated(data):
            return handler(data)


def _plain(node: Any) -> Any:
    if isinstance(node, BaseModel):
        return _plain(node.model_dump(exclude_unset=True))
    if isinstance(node, Mapping):
        return {k: _plain(v) for k, v in node.items()}
    return node


def _value_at(deck: Mapping, path: str) -> Any:
    node: Any = deck
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def _where(path: str) -> str:
    return " › ".join(deck_spelling(p) for p in path.split("."))


def _when_text(conditions: Mapping[str, Any]) -> str:
    return " and ".join(
        f"{_where(path)} is {'unset' if value is None else repr(value)}"
        for path, value in conditions.items()
    )


def _either_text(rows: Sequence[Mapping[str, Any]]) -> str:
    paths = [p for p in rows[0] if all(p in r for r in rows)]
    parts = []
    for path in paths:
        values = list(dict.fromkeys(r[path] for r in rows))
        shown = " or ".join("unset" if v is None else repr(v) for v in values)
        parts.append(f"{_where(path)} is {shown}")
    return " and ".join(parts)


class Variants:
    """The table of versions for one block, and the check that uses it."""

    def __init__(self, rows: Sequence[Row]):
        self.rows = [(dict(conditions), target) for conditions, target in rows]
        self.deciding = list(dict.fromkeys(p for c, _ in self.rows for p in c))

    def row_for(self, deck: Mapping) -> Optional[Row]:
        return next((row for row in self.rows if _holds(row[0], deck)), None)

    def version_for(self, deck: Mapping) -> Optional[type]:
        row = self.row_for(deck)
        if row is None or isinstance(row[1], NotAvailable):
            return None
        return row[1]

    def _deciding_setting_is_wrong(self, deck: Mapping, value: Any) -> bool:
        for path in self.deciding:
            if _inside([path], deck, value):
                continue
            if any(path not in c for c, _ in self.rows):
                continue
            known = {c[path] for c, _ in self.rows}
            if _value_at(deck, path) not in known and None not in known:
                return True
        return False

    def _check_own_key(self, deck: Mapping, value: Any) -> None:
        """Report an unknown value of a key inside the block that picks its version."""
        inside = _inside(self.deciding, deck, value)
        for path in inside:
            key = path.rsplit(".", 1)[1]
            offered = [
                c[path]
                for c, t in self.rows
                if path in c
                and c[path] is not None
                and not isinstance(t, NotAvailable)
                and _holds({p: v for p, v in c.items() if p not in inside}, deck)
            ]
            given = _value_at(deck, path)
            if offered and given not in offered:
                expected = ", ".join(repr(v) for v in dict.fromkeys(offered))
                raise ValidationError.from_exception_data(
                    "Variants",
                    [
                        InitErrorDetails(
                            type="literal_error",
                            loc=(key,),
                            input=given,
                            ctx={"expected": expected},
                        )
                    ],
                )

    def __get_pydantic_core_schema__(self, source, handler):
        per_entry = get_origin(source) in (dict, Dict)
        check = self._check_entries if per_entry else self._check_one
        return core_schema.no_info_wrap_validator_function(check, handler(source))

    def _choose(self, value) -> Optional[type]:
        deck = _DECK.get()
        if deck is None:
            raise PydanticCustomError(
                "variant_needs_deck",
                "This block's version depends on other settings, so it can only be "
                "checked as part of a whole deck.",
            )
        row = self.row_for(deck)
        if row is None:
            self._check_own_key(deck, value)
            if self._deciding_setting_is_wrong(deck, value):
                return None
            raise PydanticCustomError(
                "variant_unavailable",
                "No version of this block applies when {settings}. A version applies "
                "when {choices}.",
                {
                    "settings": _when_text(
                        {p: _value_at(deck, p) for p in self.deciding}
                    ),
                    "choices": "; or when ".join(
                        _when_text(c)
                        for c, t in self.rows
                        if not isinstance(t, NotAvailable)
                    ),
                },
            )
        if isinstance(row[1], NotAvailable):
            reason = PydanticCustomError("variant_unavailable", row[1].reason)
            inside = _inside(row[0], deck, value)
            if not inside:
                raise reason
            raise ValidationError.from_exception_data(
                "Variants",
                [
                    InitErrorDetails(
                        type=reason,
                        loc=(inside[-1].rsplit(".", 1)[1],),
                        input=_value_at(deck, inside[-1]),
                    )
                ],
            )
        return row[1]

    def _check_one(self, value, default_check):
        if value is None:
            return default_check(value)
        version = self._choose(value)
        if version is None or isinstance(value, version):
            return value
        try:
            return version.model_validate(_plain(value))
        except ValidationError as e:
            raise self._with_hints(e, version) from None

    def _check_entries(self, value, default_check):
        """Check each entry; a function is refused, since an entry is a fixed state."""
        if not isinstance(value, Mapping) or not value:
            return default_check(value)
        version = self._choose(value)
        if version is None:
            return value
        functions = [
            InitErrorDetails(
                type=PydanticCustomError(
                    "function_condition", "{message}", {"message": _function(name)}
                ),
                loc=(name,),
                input=entry,
            )
            for name, entry in value.items()
            if callable(entry)
        ]
        entries = {k: _plain(v) for k, v in value.items() if not callable(v)}
        try:
            checked = _entries_of(version).validate_python(entries)
        except ValidationError as e:
            error = self._with_hints(e, version, per_entry=True)
            if not functions:
                raise error from None
            functions += [_with_message(d, d["type"], d["msg"]) for d in error.errors()]
        if functions:
            raise ValidationError.from_exception_data("Variants", functions)
        return checked

    def _hint(
        self, error: ErrorDetails, chosen: type, per_entry: bool
    ) -> Optional[str]:
        loc = error["loc"][1:] if per_entry else error["loc"]
        if len(loc) != 1 or error["type"] not in ("extra_forbidden", "literal_error"):
            return None
        key = loc[0]
        allowing = [
            conditions
            for conditions, other in self.rows
            if other is not chosen
            and not isinstance(other, NotAvailable)
            and key in other.model_fields
            and (
                error["type"] == "extra_forbidden"
                or _accepts(other, key, error["input"])
            )
        ]
        if not allowing:
            return None
        where = _either_text(allowing)
        if error["type"] == "extra_forbidden":
            return f"'{key}' is only used when {where}."
        return f"{error['input']!r} is only available when {where}. {error['msg']}."

    def _with_hints(
        self, error: ValidationError, chosen: type, per_entry: bool = False
    ):
        details = error.errors()
        hints = [self._hint(e, chosen, per_entry) for e in details]
        if not any(hints):
            return error
        rebuilt = [
            _with_message(e, "variant_only", hint) if hint else _unchanged(e)
            for e, hint in zip(details, hints)
        ]
        return ValidationError.from_exception_data(error.title, rebuilt)

    def json_rows(self):
        return [
            {"when": dict(c), "ref": f"#/$defs/{t.__name__}"}
            for c, t in self.rows
            if not isinstance(t, NotAvailable)
        ]


def _inside(paths: Sequence[str], deck: Mapping, value: Any) -> list:
    """The paths in ``paths`` that name a key of the block ``value``."""
    return [p for p in paths if _value_at(deck, p.rsplit(".", 1)[0]) == _plain(value)]


def _holds(conditions: Mapping[str, Any], deck: Mapping) -> bool:
    if not conditions:
        return True
    return When(
        {p: (UNSET if v is None else v) for p, v in conditions.items()}
    ).evaluate(deck)


def _accepts(version: type, key: str, value: Any) -> bool:
    try:
        version.__pydantic_validator__.validate_assignment(
            version.model_construct(), key, value
        )
    except ValidationError:
        return False
    return True


def _function(name: str) -> str:
    return (
        f"Reference condition '{name}' is a function. A reference condition is "
        "a fixed state. To trim a farfield's angle of attack to a lift or drag "
        "coefficient, give the farfield a 'target', as "
        '"target": {"alpha": {"cl": 0.5}}.'
    )


def _with_message(error: ErrorDetails, kind: str, message: str) -> InitErrorDetails:
    text = message.replace("{", "{{").replace("}", "}}")
    return InitErrorDetails(
        type=PydanticCustomError(kind, text), loc=error["loc"], input=error["input"]
    )


def _unchanged(error: ErrorDetails) -> InitErrorDetails:
    if error["type"] not in BUILTIN_ERROR_TYPES:
        return _with_message(error, error["type"], error["msg"])
    return InitErrorDetails(
        type=error["type"],
        loc=error["loc"],
        input=error["input"],
        ctx=error.get("ctx", {}),
    )


@lru_cache(maxsize=None)
def _entries_of(version: type) -> TypeAdapter:
    return TypeAdapter(Dict[str, version])


BUILTIN_ERROR_TYPES = frozenset(core_schema.ErrorType.__args__)
