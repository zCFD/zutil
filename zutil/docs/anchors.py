"""
Routes and anchors for reference entries.

Every settings block with keys of its own is a page, at any depth.  The
route names the page and mirrors the schema: each block key adds a segment,
instance names and list indices are dropped and the boundary-condition or
fluid-zone type is inserted.  The fragment is the entry's key, lowercased,
spaces and underscores turned into hyphens.  Union tags add no segment.

``anchor(key_path, type)`` gives the ``route#fragment`` for any deck key path,
so validation errors, the form schema and the generator agree on one link.
"""

import functools
import re
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Tuple, Union

_SEPARATORS = re.compile(r"[\s_]+")
_RELEASE_TAG = re.compile(r"v?(\d{4}\.\d{2}\.\d+)")

DOCS_ROOT = "https://docs.zenotech.com"

TYPED_NODES = {
    "boundary_conditions": "boundary-conditions",
    "fluid_zones": "fluid-zones",
}


@functools.lru_cache(maxsize=None)
def is_named(key: str) -> bool:
    """Whether the model block's typed node ``key`` holds entries under names of the deck's choosing."""
    from zutil.validate.schemas import model_schemas as mod

    from .introspect import shape_of

    fi = mod.ModelSettings.model_fields.get(key)
    return fi is not None and shape_of(fi, key).container == "named"


def kebab(key: str) -> str:
    """A dict key or node name as one anchor or route segment."""
    return _SEPARATORS.sub("-", str(key).strip().lower()).strip("-")


def fragment(key_path: Iterable[str]) -> str:
    """The fragment for an entry at ``key_path`` below its page root."""
    return "-".join(kebab(k) for k in key_path)


def deck_key(key: str) -> str:
    """A schema key as decks spell it."""
    return key.replace("_", " ")


def normalise_key(key: str) -> str:
    """A deck or side-file key in schema spelling (case- and space-folded)."""
    return _SEPARATORS.sub("_", str(key).strip().lower())


@dataclass(frozen=True)
class Location:
    """Where a deck key path lands in the reference.

    ``keys`` are the schema keys that make the fragment; ``instances`` are the
    positions in the key path that hold an instance name or a list index;
    ``owners`` are the models that declare the path's last key, empty when the
    path ends on an instance name or a key the schema does not have.
    ``accepted`` are the keys the schema takes in place of the last key when
    the schema does not have it, and empty otherwise.
    """

    route: Optional[str]
    keys: Tuple[str, ...] = ()
    instances: FrozenSet[int] = frozenset()
    owners: Tuple[type, ...] = ()
    accepted: Tuple[str, ...] = ()

    @property
    def anchor(self) -> Optional[str]:
        if self.route is None:
            return None
        return f"{self.route}#{fragment(self.keys)}" if self.keys else self.route


def anchor(
    key_path: Union[str, Iterable[str]], type: Optional[str] = None
) -> Optional[str]:
    """The reference ``route#fragment`` for a deck key path.

    ``key_path`` is root-anchored, dotted or as a sequence, in deck or schema
    spelling, with instance names in place
    (``model.m1.boundary_conditions.bc_1.velocity.omega``).  ``type`` is the
    boundary-condition or fluid-zone type the path runs through; without it
    such a path resolves to the type index.  A key the schema does not have
    ends the fragment at its parent, so the link always names a real entry.
    """
    return locate(key_path, type).anchor


def locate(key_path: Union[str, Iterable[str]], type: Optional[str] = None) -> Location:
    """The route, fragment keys, instance positions and owners of a deck key path."""
    keys = _split(key_path)
    if not keys:
        return Location(None)
    from zutil.validate.schemas import model_schemas as mod
    from zutil.validate.zcfd_schema import ZCFDInputModelBase

    if keys[0] == "solver":
        return _solver(keys)
    if keys[0] == "model":
        return _model(keys, type, mod.ModelSettings)
    return _walk("reference/input-deck", [ZCFDInputModelBase], keys, 0, set())


def breadcrumb(
    key_path: Union[str, Iterable[str]], type: Optional[str] = None
) -> List[str]:
    """A deck key path in the deck's own spelling: spaced keys, instance names as written."""
    parts = [str(p) for p in _raw(key_path)]
    instances = locate(parts, type).instances
    return [
        p if i in instances else deck_key(normalise_key(p)) for i, p in enumerate(parts)
    ]


def reference_url(
    anchor_text: Optional[str], version: Optional[str] = None
) -> Optional[str]:
    """The docs URL for an anchor: frozen at a release tag, else the root."""
    if not anchor_text:
        return None
    release = _release(version if version is not None else zcfd_version())
    base = f"{DOCS_ROOT}/v{release}" if release else DOCS_ROOT
    return f"{base}/{anchor_text}"


@functools.lru_cache(maxsize=1)
def zcfd_version() -> Optional[str]:
    """The installed solver's version, or None when no solver is importable."""
    try:
        import libzCFDVersion  # type: ignore
    except ImportError:
        return None
    try:
        return str(libzCFDVersion.get_project_version())
    except Exception:
        return None


def _release(version: Optional[str]) -> Optional[str]:
    match = _RELEASE_TAG.fullmatch((version or "").strip())
    return match.group(1) if match else None


def _raw(key_path: Union[str, Iterable[str]]) -> list:
    parts = key_path.split(".") if isinstance(key_path, str) else list(key_path)
    return [p for p in parts if str(p) != ""]


def _split(key_path: Union[str, Iterable[str]]) -> List[str]:
    return [normalise_key(p) for p in _raw(key_path)]


@functools.lru_cache(maxsize=1)
def _solver_blocks() -> tuple:
    from zutil.validate.zcfd_schema import ZCFDInputModelBase

    from .introspect import routed_variants

    fi = ZCFDInputModelBase.model_fields["solver"]
    blocks: list = []
    for _, block in routed_variants(fi) or [({}, fi.annotation)]:
        if block not in blocks:
            blocks.append(block)
    return tuple(blocks)


def _solver(keys: List[str]) -> Location:
    from zutil.validate.zcfd_schema import ZCFDInputModelBase

    blocks = list(_solver_blocks())
    if len(keys) == 1:
        return Location("reference/solver", owners=(ZCFDInputModelBase,))
    section = keys[1]
    owners = tuple(b for b in blocks if section in b.model_fields)
    if not owners:
        accepted = _fields_of(blocks) if len(keys) == 2 else ()
        return Location("reference/solver", accepted=accepted)
    route = f"reference/solver/{kebab(section)}"
    fis = [b.model_fields[section] for b in owners]
    models, named = _leaves(fis, f"solver.{section}")
    if named and len(keys) > 2:
        return _walk(route, models, keys, 3, {2})
    return _walk(route, models, keys, 2, set(), owners)


def _model(keys: List[str], type: Optional[str], settings) -> Location:
    from zutil.validate.schemas.solver_schemas import MESH_OVERRIDES
    from zutil.validate.zcfd_schema import ZCFDInputModelBase

    if len(keys) == 1:
        return Location("reference/model", owners=(ZCFDInputModelBase,))
    if len(keys) == 2:
        return Location("reference/model", (), frozenset({1}))
    key = keys[2]
    fi = settings.model_fields.get(key)
    if fi is None:
        accepted = _fields_of([settings]) if len(keys) == 3 else ()
        return Location("reference/model", (), frozenset({1}), accepted=accepted)
    owners = (settings,)
    if key in MESH_OVERRIDES:
        route = f"reference/solver/{kebab(key)}"
        solver_fis = [
            b.model_fields[key] for b in _solver_blocks() if key in b.model_fields
        ]
        models, _ = _leaves(solver_fis + [fi], f"model.<mesh>.{key}")
        return _walk(route, models, keys, 3, {1}, owners)
    if key in TYPED_NODES:
        return _typed(keys, type, fi, f"reference/model/{TYPED_NODES[key]}")
    models, _ = _leaves([fi], f"model.<mesh>.{key}")
    if models:
        return _walk(f"reference/model/{kebab(key)}", models, keys, 3, {1}, owners)
    return _walk("reference/model", [settings], keys, 2, {1})


def _typed(keys: List[str], type: Optional[str], fi, route: str) -> Location:
    from zutil.validate.schemas import model_schemas as mod

    from .introspect import shape_of

    shape = shape_of(fi, route)
    named = shape.container == "named"
    wanted = kebab(type) if type is not None else None
    group = [v.model for v in shape.variants if wanted and kebab(v.tag) == wanted]
    if len(keys) == 3 and (named or not group):
        return Location(route, (), frozenset({1}), (mod.ModelSettings,))
    instances = {1, 3} if named else {1}
    if not group:
        return Location(route, (), frozenset(instances))
    return _walk(f"{route}/{wanted}", group, keys, 4 if named else 3, instances)


def _leaves(fis: list, location: str) -> Tuple[list, bool]:
    from .introspect import shape_of

    models: list = []
    contained = False
    for fi in fis:
        shape = shape_of(fi, location)
        contained = contained or shape.container is not None
        found = (
            [shape.model]
            if shape.kind == "block"
            else [v.model for v in shape.variants]
        )
        for model in found:
            if model not in models:
                models.append(model)
    return models, contained


def _walk(
    route: str,
    models: list,
    keys: List[str],
    start: int,
    instances: set,
    owners: tuple = (),
) -> Location:
    path: List[str] = []
    i = start
    while i < len(keys) and models:
        key = keys[i]
        holders = tuple(m for m in models if key in m.model_fields)
        if not holders:
            break
        owners = holders
        i += 1
        models, contained = _leaves([m.model_fields[key] for m in holders], key)
        if any(m.model_fields for m in models):
            route = f"{route}/{kebab(key)}"
        else:
            path.append(key)
        if contained and i < len(keys):
            instances.add(i)
            owners = ()
            i += 1
    accepted = _fields_of(models) if i == len(keys) - 1 else ()
    if i < len(keys):
        owners = ()
    return Location(route, tuple(path), frozenset(instances), owners, accepted)


def _fields_of(models: Iterable[type]) -> tuple:
    return tuple(dict.fromkeys(k for m in models for k in m.model_fields))
