"""
Turn validation errors into the nodes zui highlights and the CLI prints.

One adapter, :func:`map_errors_to_nodes`, reads every error at its path in
the deck and gives it a node::

    {"nodes": ["boundary conditions", "bc_1"],
     "field_name": "temperature",
     "field_path": "model.m1.boundary_conditions.bc_1.temperature",
     "path": ["model", "m1", "boundary_conditions", "bc_1", "temperature"],
     "title": "Temperature",
     "display_message": "Wall temperature must be greater than 0 K.",
     "message": "model.m1.boundary_conditions.bc_1.temperature: Wall ...",
     "anchor": "reference/model/boundary-conditions/wall#temperature",
     "severity": "error"}

A path is shown in the deck's own spelling, a tree node is named by the deck
spelling of its key, and a field is named by its schema title.

``nodes`` is deprecated: it names zui's tree, which zui now derives from
``field_path`` itself. It is still emitted for older zui builds, and nothing
here keys on it.
:func:`format_errors_as_text` renders the same nodes for a terminal.
"""

import difflib
import functools
import importlib
import inspect
import os
import pkgutil
import re
from collections.abc import Iterator

from pydantic import BaseModel

from zutil.docs.anchors import (
    anchor,
    breadcrumb,
    deck_key,
    is_named,
    locate,
    normalise_key,
    reference_url,
    TYPED_NODES,
)

from .schemas.variants import BUILTIN_ERROR_TYPES

_TREE_NAMES = {"output_settings": "output"}


@functools.lru_cache(maxsize=1)
def _known_model_class_names() -> frozenset:
    """Return the names of every Pydantic model class defined under schemas/.

    Pydantic v2 inserts the class name of a union member directly into an
    error's ``loc`` tuple for "smart" (non-discriminated) unions — e.g.
    validating ``FluidModel = Union[FluidModelBase, ViscousFluidModel]``
    against a bad dict produces loc segments ``'FluidModelBase'`` and
    ``'ViscousFluidModel'``.  Those are internal routing artifacts, not
    part of the user's config path, and must be dropped.

    User-supplied identifiers (a BC named ``Inlet``, an IC named ``IC1``)
    can look superficially similar (capitalised, no underscore) but are
    not model class names, so they must be kept.  Identify the artifacts
    structurally — by checking the segment against the actual set of
    model classes the schemas package defines — rather than guessing from
    the string's shape.
    """
    from . import schemas as schemas_pkg

    names = set()
    for module_info in pkgutil.iter_modules(schemas_pkg.__path__):
        try:
            module = importlib.import_module(
                f"{schemas_pkg.__name__}.{module_info.name}"
            )
        except ImportError:
            continue
        for _, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
            ):
                names.add(obj.__name__)
    return frozenset(names)


_MISSING = object()


def _descend(node, segment):
    """The value `segment` addresses in `node`, or `_MISSING`."""
    if isinstance(node, dict):
        return node.get(segment, node.get(str(segment), _MISSING))
    if isinstance(node, (list, tuple)) and isinstance(segment, int):
        return node[segment] if -len(node) <= segment < len(node) else _MISSING
    if isinstance(node, BaseModel):
        return getattr(node, str(segment), _MISSING)
    return _MISSING


def _clean_loc_parts(loc: tuple, data: dict | None = None) -> list[str]:
    """Convert a Pydantic error loc tuple to clean path parts.

    Strips Pydantic v2 internal segments:
      - function-wrap[...], function-after[...] (validator wrappers)
      - bare model class names (union discriminator artifacts — see
        ``_known_model_class_names``)
      - discriminated-union tags, which pydantic inserts as a level of
        their own (``bc_1.wall.velocity`` for a wall's velocity)

    A tag can be spelled the same as a real field — ``pressure`` is both a
    reference-condition field and an inflow kind — so tags are identified
    by walking `data` rather than by name: an intermediate segment the
    user's own config doesn't contain is routing, not path. The final
    segment is kept unless the value it would index is a plain number or
    text, where it can only be the tag of a scalar union member, or a list
    it cannot index by name, where it is the tag of the member; for a
    `missing` error it is precisely the key that isn't there. Without
    `data` only the two name-based strips apply.
    """
    known_models = _known_model_class_names()

    kept = [
        segment
        for segment in loc
        if "[" not in str(segment) and str(segment) not in known_models
    ]

    parts = []
    node = data
    for index, segment in enumerate(kept):
        if data is not None and index < len(kept) - 1:
            stepped = _descend(node, segment)
            if stepped is _MISSING:
                continue
            node = stepped
        elif (
            data is not None
            and parts
            and (
                isinstance(node, (str, int, float))
                or (isinstance(node, (list, tuple)) and not isinstance(segment, int))
            )
        ):
            continue
        parts.append(str(segment))
    return parts


def _tag_after(raw: list, segment: str) -> str:
    if segment not in raw:
        return ""
    after = raw.index(segment) + 1
    return raw[after] if after < len(raw) else ""


def _retired_contexts(path: list, raw: list) -> list:
    contexts = []
    if len(path) == 2 and path[0] == "solver":
        contexts.append("solver")
    if len(path) == 3 and path[0] == "model":
        contexts.append("model")
    if len(path) > 1 and path[-2] == "output_settings":
        contexts.append("output settings")
    transforms = len(path) > 2 and path[0] == "model" and path[2] == "transforms"
    if transforms and len(path) == 4:
        contexts.append("transforms")
    motion = transforms and len(path) > 4 and path[3] == "motion"
    if motion:
        contexts.append("motion")
        if _tag_after(raw, "motion") == "translating":
            contexts.append("translating")
    rotating_zone = (
        len(path) == 5
        and path[2] == "fluid_zones"
        and _tag_after(raw, path[3]) == "rotating"
    )
    if rotating_zone:
        contexts.append("rotating zone")
    if (
        rotating_zone
        or (motion and _tag_after(raw, "motion") == "rotating")
        or _tag_after(raw, "velocity") == "rotating"
    ):
        contexts.append("rotation")
    boundary = len(path) > 3 and path[2] == "boundary_conditions"
    if boundary and _tag_after(raw, path[3]) in ("outflow", "farfield"):
        contexts.append(_tag_after(raw, path[3]))
    if boundary and len(path) == 6 and path[4] == "target":
        contexts.append("target")
    if boundary and len(path) == 7 and path[4:6] == ["target", "alpha"]:
        contexts.append("target alpha")
    contexts.append("*")
    return contexts


def retired_error(error: dict, data: dict | None = None) -> "tuple[list, str] | None":
    """The cleaned path and the text for an error on a retired type or setting.

    None unless the error is an unknown fluid-zone type or a forbidden key
    that ``schemas.retired`` names.
    """
    from .schemas.retired import RETIRED_SETTINGS, RETIRED_TYPES

    raw = [str(s) for s in error.get("loc", ())]
    error_type = error.get("type", "")
    path = _clean_loc_parts(error.get("loc", ()), data)
    if error_type == "union_tag_invalid":
        value = error.get("input")
        tag = value.get("type") if isinstance(value, dict) else None
        if len(path) < 2 or path[-2] != "fluid_zones" or not isinstance(tag, str):
            return None
        text = RETIRED_TYPES.get(("fluid zone", tag.lower()))
        if text is None:
            return None
        return path + ["type"], text.replace("{owner}", path[-1])
    if error_type != "extra_forbidden" or not path:
        return None
    key = normalise_key(path[-1])
    for context in _retired_contexts(path, raw):
        text = RETIRED_SETTINGS.get((context, key))
        if text is not None:
            owner = path[-2] if len(path) > 1 else ""
            return path, text.replace("{key}", key.replace("_", " ")).replace(
                "{owner}", owner
            )
    return None


def _humanise_model_name(name: str) -> str:
    """`InflowVectorModel` -> `inflow vector`. Class names leak into
    pydantic's messages; they are our vocabulary, not the user's."""
    trimmed = re.sub(r"(Model|Boundary|Settings|Control)$", "", name) or name
    words = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", trimmed)
    return " ".join(w.lower() for w in words) if words else name.lower()


_BOUND_ERRORS = (
    "greater_than",
    "greater_than_equal",
    "less_than",
    "less_than_equal",
)

_TYPE_MESSAGES = {
    "int_type": "Must be a whole number.",
    "int_parsing": "Must be a whole number.",
    "float_type": "Must be a number.",
    "float_parsing": "Must be a number.",
    "decimal_parsing": "Must be a number.",
    "string_type": "Must be text.",
    "bool_type": "Must be true or false.",
    "bool_parsing": "Must be true or false.",
    "list_type": "Must be a list of values.",
    "dict_type": "Must be a set of settings, not a single value.",
}


def _values(n) -> str:
    return f"{n} value" if n == 1 else f"{n} values"


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _bound_sentence(
    relation: str, bound: float, subject: str | None, units: str | None
) -> str:
    if subject is None:
        return f"Must be {relation} {bound}."
    if isinstance(bound, float) and bound.is_integer():
        bound = int(bound)
    suffix = f" {units}" if units else ""
    return f"{subject} must be {relation} {bound}{suffix}."


def _sentence_case(title: str) -> str:
    words = title.split()
    return " ".join(
        w if i == 0 or (w.isupper() and len(w) > 1) else w.lower()
        for i, w in enumerate(words)
    )


def _declaring_field(parts: list, data: dict | None):
    owners = locate(parts, deck_type(parts, data)).owners
    return owners[0].model_fields.get(normalise_key(parts[-1])) if owners else None


def _title(parts: list, data: dict | None) -> str:
    """The schema title of the field at `parts`, or its key in the deck's spelling."""
    field_info = _declaring_field(parts, data)
    if field_info is not None and field_info.title:
        return field_info.title
    return deck_key(normalise_key(parts[-1])).title()


def _field_subject(parts: list, data: dict | None) -> tuple:
    """The declaring field's title in sentence case and its units, or ``(None, None)``."""
    from zutil.docs.introspect import zcfd_extra

    field_info = _declaring_field(parts, data)
    if field_info is None or not field_info.title:
        return None, None
    return _sentence_case(field_info.title), zcfd_extra(field_info).get("units")


def _humanise_message(
    error_type: str,
    msg: str,
    ctx: dict,
    subject: str | None = None,
    units: str | None = None,
) -> str:
    """Turn one pydantic error into a sentence aimed at whoever is filling
    in the form, not at whoever wrote the model.

    A bound error names ``subject`` and states the bound in ``units`` when
    they are given. An error raised by a zCFD rule already reads as a
    sentence and is kept as written.
    """
    if error_type not in BUILTIN_ERROR_TYPES:
        return _sentence(msg)

    if error_type in _TYPE_MESSAGES:
        return _TYPE_MESSAGES[error_type]

    if error_type == "model_type":
        match = re.search(r"instance of (\w+)", msg)
        if match:
            return f"Must be a set of {_humanise_model_name(match.group(1))} settings."
        return "Must be a set of settings, not a single value."

    if error_type in ("greater_than", "greater_than_equal"):
        bound = ctx.get("gt", ctx.get("ge"))
        at_least = "" if error_type == "greater_than" else " or equal to"
        return _bound_sentence(f"greater than{at_least}", bound, subject, units)

    if error_type in ("less_than", "less_than_equal"):
        bound = ctx.get("lt", ctx.get("le"))
        at_most = "" if error_type == "less_than" else " or equal to"
        return _bound_sentence(f"less than{at_most}", bound, subject, units)

    if error_type in ("too_short", "too_long"):
        low, high = ctx.get("min_length"), ctx.get("max_length")
        if low is not None and low == high:
            return f"Must have exactly {_values(low)}."
        if error_type == "too_short" and low == 1:
            return "Must have at least one entry."
        if error_type == "too_short" and low is not None:
            return f"Must have at least {_values(low)}."
        if high is not None:
            return f"Must have at most {_values(high)}."

    if error_type in ("literal_error", "enum"):
        allowed = ctx.get("expected")
        if allowed:
            return f"Must be one of {allowed}."

    if error_type == "value_error":
        text = re.sub(r"^Value error,\s*", "", msg).strip()
        return _sentence(text) if text else msg

    if error_type in ("union_tag_invalid", "union_tag_not_found"):
        key = str(ctx.get("discriminator", "")).strip("'")
        allowed = ctx.get("expected_tags")
        if error_type == "union_tag_not_found":
            return f'Missing required field "{deck_key(key).title()}"' if key else msg
        if allowed:
            return f"{deck_key(key).title()} must be one of {allowed}."

    return msg


@functools.lru_cache(maxsize=None)
def _section_shape(top: str, key: str) -> str:
    """``"collection"``, ``"block"`` or ``""`` for the section ``key`` of the solver or a mesh."""
    from zutil.docs.introspect import shape_of

    from .schemas import model_schemas as mod
    from .schemas import solver_schemas as ss

    owner = ss.SolverBlockModel if top == "solver" else mod.ModelSettings
    field_info = owner.model_fields.get(key)
    if field_info is None:
        return ""
    shape = shape_of(field_info, f"{top}.{key}")
    if shape.container == "named":
        return "collection"
    return "" if shape.kind == "scalar" else "block"


def _tree_name(key: str) -> str:
    return _TREE_NAMES.get(key, deck_key(key))


def _placement(parts: list) -> tuple:
    """The tree nodes an error at `parts` highlights, and where its open panel ends."""
    top = parts[0]
    if top == "model" and len(parts) == 2:
        return [parts[1]], 2
    if top not in ("solver", "model"):
        return [], 1
    if len(parts) < (2 if top == "solver" else 3):
        return [top], 1
    start = 1 if top == "solver" else 2
    key = parts[start]
    shape = _section_shape(top, key)
    if not shape:
        return ([top] if top == "solver" else [parts[1]]), start
    if shape == "collection" and len(parts) > start + 1:
        return [_tree_name(key), parts[start + 1]], start + 2
    return [_tree_name(key)], start + 1


def _node(error: dict, data: dict | None, severity: str) -> dict:
    retired = retired_error(error, data)
    parts = retired[0] if retired else _clean_loc_parts(error.get("loc", ()), data)
    error_type = error.get("type", "")
    msg = error.get("msg", "")
    ctx = error.get("ctx") or {}

    if not parts:
        text = _humanise_message(error_type, msg, ctx)
        return {
            "nodes": [],
            "field_name": "",
            "field_path": "",
            "path": [],
            "title": "",
            "display_message": text,
            "message": text,
            "anchor": None,
            "severity": severity,
        }

    nodes, depth = _placement(parts)
    instances = locate(parts, deck_type(parts, data)).instances
    tail = list(range(depth, len(parts))) or [len(parts) - 1]
    names = [parts[i] if i in instances else _title(parts[: i + 1], data) for i in tail]
    if error_type == "union_tag_not_found":
        key = str(ctx.get("discriminator", "")).strip("'")
        names.append(_title(parts + [key], data))
    field_display = " → ".join(n for n in names if n)

    if retired:
        display_message = retired[1]
    elif error_type in ("missing", "union_tag_not_found"):
        display_message = f'Missing required field "{field_display}"'
    elif error_type == "extra_forbidden":
        display_message = f'Unknown key "{deck_key(normalise_key(parts[-1]))}".'
    elif error_type in _BOUND_ERRORS:
        subject, units = _field_subject(parts, data)
        display_message = _humanise_message(error_type, msg, ctx, subject, units)
    else:
        display_message = _humanise_message(error_type, msg, ctx)

    field_path = ".".join(parts)
    return {
        "nodes": nodes,
        "field_name": parts[-1] if len(parts) > depth else "",
        "field_path": field_path,
        "path": parts,
        "title": names[-1] if names else "",
        "display_message": display_message,
        "message": f"{field_path}: {display_message}",
        "anchor": node_anchor(parts, data),
        "severity": severity,
    }


def _suggestions(
    unknown: list, missing: list, data: dict | None
) -> Iterator[tuple[dict, str]]:
    """Pair each unknown-key node with the schema key it most likely stands for.

    The closest accepted key wins; failing that, a block with one unknown
    key and one missing key pairs the two.
    """

    def parent(node: dict) -> tuple:
        return tuple(node["path"][:-1])

    for node in unknown:
        parts = node["path"]
        accepted = locate(parts, deck_type(parts, data)).accepted
        match = difflib.get_close_matches(normalise_key(parts[-1]), accepted, n=1)
        if not match:
            siblings = [n for n in unknown if parent(n) == parent(node)]
            absent = [n for n in missing if parent(n) == parent(node)]
            if len(siblings) == 1 and len(absent) == 1:
                match = [absent[0]["path"][-1]]
        if match:
            yield node, match[0]


def deck_type(key_path, data: dict | None) -> str | None:
    """The boundary-condition or fluid-zone type a key path runs through, read from the deck."""
    parts = key_path.split(".") if isinstance(key_path, str) else list(key_path)
    if len(parts) < 3 or parts[0] != "model" or parts[2] not in TYPED_NODES:
        return None
    depth = 4 if is_named(normalise_key(parts[2])) else 3
    if len(parts) < depth:
        return None
    node = data
    for segment in parts[:depth]:
        if not isinstance(node, dict):
            return None
        wanted = normalise_key(segment)
        node = next((v for k, v in node.items() if normalise_key(k) == wanted), None)
    value = node.get("type") if isinstance(node, dict) else None
    return value if isinstance(value, str) else None


def node_anchor(key_path, data: dict | None = None) -> str | None:
    """The reference anchor for an error at `key_path` in the deck `data`."""
    return anchor(key_path, deck_type(key_path, data))


def map_errors_to_nodes(
    errors: list[dict],
    data: dict | None = None,
    severity: str = "error",
) -> list[dict]:
    """Turn pydantic error dicts into nodes, one per problem.

    ``data`` is the deck the errors came from: it tells a real path segment
    from pydantic's union routing and names each boundary condition's type.
    A field whose type is a union fails once per branch; those failures are
    one problem with several acceptable answers, so they merge into one node.
    A misspelt key is one problem too: its node suggests the key it stands
    for, and the missing-key node for that key is dropped. A retired setting
    has its own text and gets no suggestion.
    """
    result = []
    seen = set()
    unknown = []
    missing = []
    for error in errors:
        node = _node(error, data, severity)
        key = (node["field_path"], node["display_message"])
        if key in seen:
            continue
        seen.add(key)
        result.append(node)
        if (
            node["path"]
            and error.get("type") == "extra_forbidden"
            and not retired_error(error, data)
        ):
            unknown.append(node)
        elif node["path"] and error.get("type") == "missing":
            missing.append(node)
    for node, match in _suggestions(unknown, missing, data):
        node["display_message"] += f' Did you mean "{deck_key(match)}"?'
        node["message"] = f"{node['field_path']}: {node['display_message']}"
        replaced = tuple(node["path"][:-1]) + (match,)
        result = [
            n for n in result if not (n in missing and tuple(n["path"]) == replaced)
        ]
    return _merge_union_alternatives(result)


def _merge_union_alternatives(nodes: list[dict]) -> list[dict]:
    """Collapse the several branch failures of one union-typed field into
    one node, keeping declaration order (the simple form comes first)."""
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []

    for node in nodes:
        if not node["field_path"]:
            key = (len(order),)
        else:
            key = (node["field_path"], node["severity"])
        if key not in merged:
            merged[key] = dict(node)
            order.append(key)
            continue

        first = merged[key]
        alternative = node["display_message"].rstrip(".")
        alternative = re.sub(r"^Must be\s+", "", alternative)
        alternative = alternative[:1].lower() + alternative[1:]
        first["display_message"] = (
            f"{first['display_message'].rstrip('.')} — or {alternative}."
        )
        first["message"] = f"{first['field_path']}: {first['display_message']}"

    return [merged[key] for key in order]


def format_errors_as_text(
    error_nodes: list[dict], data: dict | None = None, origin: str | None = None
) -> str:
    """Format structured error_nodes as terminal text.

    Each error is a breadcrumb in the deck's own spelling, the message and a
    link to its reference entry::

        Configuration validation failed with 1 error:

          solver › equations › turbulence › model
            Must be one of 'sst', 'sas', 'sa-neg', 'sst-transition' or 'spalart-allmaras'.
            Reference: https://docs.zenotech.com/v2026.08.14860/reference/solver/equations/turbulence#model

    `data` is the deck, which names each boundary condition's and fluid
    zone's type. `origin` is the deck file, which then heads the text in
    place of the generic line.
    """
    if not error_nodes:
        return "Configuration is valid."

    n = len(error_nodes)
    count = f"{n} error{'' if n == 1 else 's'}"
    if origin:
        header = f"{os.path.basename(origin)}: {count}"
    else:
        header = f"Configuration validation failed with {count}:"
    return "\n\n".join([header] + [_error_block(e, data) for e in error_nodes])


def _error_block(node: dict, data: dict | None) -> str:
    path = node.get("path") or (node.get("field_path") or "").split(".")
    message = node.get("display_message") or node.get("message", "")
    if not any(path):
        return "  " + message.replace("\n", "\n  ")
    message = message.replace("\n", "\n    ")
    crumbs = " › ".join(breadcrumb(path, deck_type(path, data)))
    lines = [f"  {crumbs}", f"    {message}"]
    link = node["anchor"] if "anchor" in node else node_anchor(path, data)
    url = reference_url(link)
    if url:
        lines.append(f"    Reference: {url}")
    return "\n".join(lines)
