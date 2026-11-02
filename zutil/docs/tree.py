"""
Walk the canonical deck models from the root into reference pages.

Every schema node and every settings block with keys of its own, at any
depth, becomes a page at a route that mirrors the schema; a named or listed
block's page documents one instance.  Boundary conditions and fluid zones get
one page per type.  Entries keep schema order; union members are merged into
one entry list, each entry tagged with the member models that carry it.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import copy
import re
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

from pydantic.fields import FieldInfo

from zutil.validate.schemas import model_schemas as mod
from zutil.validate.schemas import output_variables as ov
from zutil.validate.schemas.solver_schemas import MESH_OVERRIDES
from zutil.validate.zcfd_schema import ZCFDInputModelBase

from .anchors import TYPED_NODES, deck_key, fragment, kebab
from .sidefiles import KeyEntry, OptionEntry, SideFile
from .introspect import (
    UnresolvedUnionError,
    Variant,
    declares,
    declaring_class,
    facts_of,
    fingerprint,
    is_model,
    literal_values,
    option_definitions,
    resolve_union,
    routed_variants,
    selected_tag,
    shape_of,
    unwrap,
    zcfd_extra,
)


@dataclass
class Page:
    route: str
    title: str
    kind: str
    parent: Optional[str]
    order: int
    deck_path: List[str]
    description: Optional[str] = None
    type: Optional[dict] = None
    facts: dict = field(default_factory=dict)
    variants: List[Variant] = field(default_factory=list)
    types: List[dict] = field(default_factory=list)
    entries: List[dict] = field(default_factory=list)
    extra_deck_paths: List[List[str]] = field(default_factory=list)
    outline: Optional[dict] = None
    walkthrough: Optional[dict] = None
    split_from: Optional[Tuple[str, str]] = None
    type_value: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "format": 1,
            "route": self.route,
            "title": self.title,
            "kind": self.kind,
            "parent": self.parent,
            "deck_path": self.deck_path,
            "description": self.description,
            "type": self.type,
            "facts": self.facts,
            "variants": [
                {"model": v.model.__name__, "tag": v.tag, "when": v.when}
                for v in self.variants
            ],
            "types": self.types,
            "entries": self.entries,
            "outline": self.outline,
            "walkthrough": self.walkthrough,
        }


class Walker:
    """Builds every reference page and records what the build must report."""

    def __init__(self, side_files: Optional[Dict[str, SideFile]] = None) -> None:
        self.side_files = side_files or {}
        self.unresolved: List[UnresolvedUnionError] = []
        self.discriminators: Dict[type, set] = defaultdict(set)
        self.reached: set = set()
        self.pages: List[Page] = []
        self.block_variants: Dict[int, List[Variant]] = {}

    def leaves(self, field_info: FieldInfo, location: str, when: dict) -> List[Variant]:
        shape = shape_of(field_info, location)
        if shape.unresolved is not None:
            self._record(shape.unresolved)
            return []
        if shape.kind == "block":
            return [Variant(shape.model, None, dict(when))]
        if shape.kind == "union":
            return [Variant(v.model, v.tag, {**when, **v.when}) for v in shape.variants]
        return []

    def _record(self, err: UnresolvedUnionError) -> None:
        if not any(e.location == err.location for e in self.unresolved):
            self.unresolved.append(err)

    def entries(
        self,
        variants: List[Variant],
        path: List[str],
        seen: frozenset,
        location: str,
        holders: tuple = (),
    ) -> List[dict]:
        """Merged entries for the fields of every variant, in first-seen schema order.

        ``holders`` are the fields that hold these variants; their defaults decide
        which discriminator value, if any, is the default.
        """
        occurrences: Dict[str, list] = {}
        for v in variants:
            self.reached.update(
                c
                for c in v.model.__mro__
                if is_model(c) and c.__module__.startswith("zutil.")
            )
            for when_key in v.when:
                parent, _, key = when_key.rpartition(".")
                if parent not in ("", location) or key not in v.model.model_fields:
                    continue
                self.discriminators[v.model].add(key)
                self.discriminators[declaring_class(v.model, key)].add(key)
            for key, fi in v.model.model_fields.items():
                if not zcfd_extra(fi).get("hidden"):
                    occurrences.setdefault(key, []).append((v, fi))
        return [
            self.entry(key, occ, variants, path, seen, location, holders)
            for key, occ in occurrences.items()
        ]

    def entry(
        self,
        key: str,
        occ: list,
        variants: List[Variant],
        path: List[str],
        seen: frozenset,
        location: str,
        holders: tuple = (),
    ) -> dict:
        first_v, first_fi = occ[0]
        key_path = path + [key]
        disc = any(key in self.discriminators.get(v.model, ()) for v, _ in occ)
        facts = [facts_of(fi, key, disc) for _, fi in occ]
        selected = _selected(holders, key) if disc else None
        if disc:
            for f in facts:
                f["default"] = {"value": selected, "show": selected is not None}
        facts_by_variant = None
        if any(f != facts[0] for f in facts[1:]):
            facts_by_variant = {v.model.__name__: f for (v, _), f in zip(occ, facts)}
        declared = declaring_class(first_v.model, key)
        side_entry = self._side_entry(declared, key)
        prose = side_entry or self._inherited_entry(first_v.model, key)
        fixed = disc and len(literal_values(first_fi.annotation) or []) == 1
        entry = {
            "key": key,
            "deck_key": deck_key(key),
            "path": key_path,
            "anchor": fragment(key_path),
            "title": first_fi.title or deck_key(key).capitalize(),
            "facts": facts[0],
            "description": _description(occ, key),
            "gap": False,
            "options": self.options(key, occ, disc, selected),
            "union": None,
            "fields": [],
            "variants": None
            if len(occ) == len(variants)
            else [v.model.__name__ for v, _ in occ],
            "facts_by_variant": facts_by_variant,
            "demo": prose.attrs.get("demo") if prose else None,
            "guide": prose.attrs.get("guide") if prose else None,
            "examples": [],
            "link": None,
            "declared_in": {
                "model": declared.__name__,
                "module": declared.__module__.rsplit(".", 1)[-1],
                "side_file": _side_rel(declared.__module__),
            },
            "fingerprint": self._fingerprint(declared, key, side_entry, fixed),
        }
        entry["gap"] = entry["description"] is None and not fixed
        sub: List[Variant] = []
        held_by: Dict[type, List[Variant]] = defaultdict(list)
        for v, fi in occ:
            for leaf in self.leaves(fi, f"{location}.{key}", {}):
                held_by[leaf.model].append(v)
                if all(leaf.model is not s.model for s in sub):
                    sub.append(leaf)
        self.block_variants[id(entry)] = sub
        errs = [e for e in self.unresolved if e.location == f"{location}.{key}"]
        if errs:
            entry["unresolved"] = {"members": errs[0].members, "reason": errs[0].reason}
        if sub:
            shapes = [shape_of(fi, key) for _, fi in occ]
            union_disc = next(
                (s.discriminator for s in shapes if s.kind == "union"), None
            )
            if len(sub) > 1 or any(s.kind == "union" for s in shapes):
                entry["union"] = {
                    "discriminator": union_disc,
                    "variants": [
                        {"value": s.tag, "model": s.model.__name__} for s in sub
                    ],
                }
            fresh = [s for s in sub if s.model not in seen]
            if fresh:
                entry["fields"] = self.entries(
                    fresh,
                    key_path,
                    seen | {s.model for s in fresh},
                    f"{location}.{key}",
                    tuple(fi for _, fi in occ),
                )
            if union_disc:
                _mark_applicability(entry, sub, held_by, [v for v, _ in occ])
        return entry

    def options(
        self, key: str, occ: list, disc: bool = False, selected: Any = None
    ) -> Optional[list]:
        """Literal options of a key; a discriminator's default is the one its holders select."""
        values: List = []
        holders: Dict = defaultdict(list)
        defaults = {selected} if disc and selected is not None else set()
        for v, fi in occ:
            lits = literal_values(fi.annotation)
            if lits is None:
                continue
            for value in lits:
                if value not in values:
                    values.append(value)
                holders[value].append(v)
            if not disc and fi.default in lits:
                defaults.add(fi.default)
        if not values:
            return None
        out = []
        for value in values:
            model = holders[value][0].model
            option = self._side_option(model, key, str(value))
            out.append(
                {
                    "value": value,
                    "default": value in defaults,
                    "variants": None
                    if disc or len(holders[value]) == len(occ)
                    else [h.model.__name__ for h in holders[value]],
                    "what": self._option_what(model, key, str(value), option),
                    "when": option.attrs.get("when") or None if option else None,
                    "only": None,
                }
            )
        return out

    def _option_what(
        self, model: type, key: str, value: str, option: Optional[OptionEntry]
    ) -> Optional[str]:
        """The schema's definition of an option, else the side file's."""
        for cls in model.__mro__:
            if declares(cls, key):
                defined = option_definitions(cls.model_fields[key]).get(value)
                if defined:
                    return defined
        return option.attrs.get("what") or None if option else None

    def _side_option(self, model: type, key: str, value: str) -> Optional[OptionEntry]:
        for cls in model.__mro__:
            if not declares(cls, key):
                continue
            side_entry = self._side_entry(cls, key)
            if side_entry and value in side_entry.options:
                return side_entry.options[value]
        return None

    def _side_entry(self, declared: type, key: str) -> Optional[KeyEntry]:
        side = self.side_files.get(declared.__module__)
        if side is None:
            return None
        section = side.sections.get(declared.__name__)
        return section.keys.get(key) if section else None

    def _inherited_entry(self, model: type, key: str) -> Optional[KeyEntry]:
        for cls in model.__mro__:
            if declares(cls, key):
                side_entry = self._side_entry(cls, key)
                if side_entry:
                    return side_entry
        return None

    def _fingerprint(
        self,
        declared: type,
        key: str,
        side_entry: Optional[KeyEntry],
        exempt: bool = False,
    ) -> dict:
        current = fingerprint(declared.model_fields[key], key)
        if side_entry is None:
            return {
                "current": current,
                "stored": None,
                "status": "exempt" if exempt else "missing",
            }
        stored = side_entry.attrs.get("fingerprint", "pending")
        status = (
            "pending" if stored == "pending" else "ok" if stored == current else "stale"
        )
        return {"current": current, "stored": stored, "status": status}

    def single(self, model: type, key: str, location: str) -> dict:
        v = Variant(model)
        return self.entry(
            key, [(v, model.model_fields[key])], [v], [], frozenset({model}), location
        )

    def row(
        self,
        owner: type,
        key: str,
        fi: FieldInfo,
        path: List[str],
        link: str,
        overridable: bool = False,
    ) -> dict:
        declared = declaring_class(owner, key)
        side_entry = self._side_entry(declared, key)
        facts = facts_of(fi, key)
        facts["overridable_per_model"] = overridable
        return {
            "key": key,
            "deck_key": deck_key(key),
            "path": path + [key],
            "anchor": fragment(path + [key]),
            "title": fi.title or deck_key(key).capitalize(),
            "facts": facts,
            "description": fi.description,
            "gap": fi.description is None,
            "options": None,
            "union": None,
            "fields": [],
            "variants": None,
            "facts_by_variant": None,
            "demo": side_entry.attrs.get("demo") if side_entry else None,
            "guide": side_entry.attrs.get("guide") if side_entry else None,
            "examples": [],
            "link": link,
            "declared_in": {
                "model": declared.__name__,
                "module": declared.__module__.rsplit(".", 1)[-1],
                "side_file": _side_rel(declared.__module__),
            },
            "fingerprint": self._fingerprint(declared, key, side_entry),
        }

    def build(self) -> List[Page]:
        root = ZCFDInputModelBase
        solver_fi = root.model_fields["solver"]
        solver_variants = self._dedupe(
            routed_variants(solver_fi) or [({}, solver_fi.annotation)]
        )
        self.reached.add(root)

        deck = Page(
            "reference/input-deck",
            "Input deck",
            "index",
            None,
            10,
            [],
            root.__doc__.strip() if root.__doc__ else None,
        )
        for key, fi in root.model_fields.items():
            if key in ("solver", "model"):
                deck.entries.append(self.row(root, key, fi, [], f"reference/{key}"))
            else:
                deck.entries.append(self.single(root, key, ""))
        self.pages.append(deck)

        solver = Page(
            "reference/solver",
            "solver",
            "index",
            "reference/input-deck",
            20,
            ["solver"],
            solver_fi.description,
            variants=[Variant(m, None, w) for w, m in solver_variants],
        )
        self.pages.append(solver)
        node_keys: List[str] = []
        for _, block in solver_variants:
            for key in block.model_fields:
                if key not in node_keys:
                    node_keys.append(key)
        for order, key in enumerate(node_keys, start=1):
            route = f"reference/solver/{kebab(key)}"
            occ = [
                (w, block, block.model_fields[key])
                for w, block in solver_variants
                if key in block.model_fields
            ]
            row = self.row(occ[0][1], key, occ[0][2], [], route, key in MESH_OVERRIDES)
            solver.entries.append(row)
            page = Page(
                route,
                key.replace("_", " "),
                "node",
                "reference/solver",
                order * 10,
                ["solver", key],
                occ[0][2].description,
            )
            leaves: List[Variant] = []
            blocks_per_model: Dict[type, int] = defaultdict(int)
            for w, block, fi in occ:
                for leaf in self.leaves(fi, f"solver.{key}", dict(w)):
                    blocks_per_model[leaf.model] += 1
                    if all(leaf.model is not x.model for x in leaves):
                        leaves.append(leaf)
            routing = {k for w, _, _ in occ for k in w}
            for leaf in leaves:
                if blocks_per_model[leaf.model] == len(solver_variants):
                    leaf.when = {k: v for k, v in leaf.when.items() if k not in routing}
            page.variants = leaves
            shape = shape_of(occ[0][2], key)
            page.facts = {
                "container": shape.container,
                "applies_when": row["facts"]["applies_when"],
                "overridable_per_model": key in MESH_OVERRIDES,
                "required": row["facts"]["required"],
            }
            if shape.container == "named":
                page.deck_path = ["solver", key, "<name>"]
            if key in MESH_OVERRIDES:
                page.extra_deck_paths = [["model", "<mesh>", key]]
                mesh_fi = mod.ModelSettings.model_fields[key]
                for leaf in self.leaves(mesh_fi, f"model.<mesh>.{key}", {}):
                    if all(leaf.model is not x.model for x in leaves):
                        leaves.append(
                            Variant(
                                leaf.model, leaf.tag, {"scope": "model", **leaf.when}
                            )
                        )
            page.entries = self.entries(
                page.variants,
                [],
                frozenset(v.model for v in page.variants),
                f"solver.{key}",
                tuple(fi for _, _, fi in occ),
            )
            self.pages.append(page)

        model = Page(
            "reference/model",
            "model",
            "index",
            "reference/input-deck",
            30,
            ["model", "<mesh>"],
            root.model_fields["model"].description
            or (mod.ModelSettings.__doc__ or "").strip()
            or None,
            variants=[Variant(mod.ModelSettings)],
        )
        self.pages.append(model)
        self.reached.add(mod.ModelSettings)
        for order, (key, fi) in enumerate(
            mod.ModelSettings.model_fields.items(), start=1
        ):
            if key in MESH_OVERRIDES:
                model.entries.append(
                    self.row(
                        mod.ModelSettings,
                        key,
                        fi,
                        [],
                        f"reference/solver/{kebab(key)}",
                        True,
                    )
                )
            elif key in TYPED_NODES:
                route = f"reference/model/{TYPED_NODES[key]}"
                model.entries.append(self.row(mod.ModelSettings, key, fi, [], route))
                self._typed(key, fi, route, order * 10)
            elif shape_of(fi, key).kind in ("block", "union"):
                route = f"reference/model/{kebab(key)}"
                model.entries.append(self.row(mod.ModelSettings, key, fi, [], route))
                page = Page(
                    route,
                    key.replace("_", " "),
                    "node",
                    "reference/model",
                    order * 10,
                    ["model", "<mesh>", key],
                    fi.description,
                )
                page.variants = self.leaves(fi, f"model.<mesh>.{key}", {})
                page.facts = {
                    "container": shape_of(fi, key).container,
                    "applies_when": facts_of(fi, key)["applies_when"],
                    "overridable_per_model": False,
                    "required": fi.is_required(),
                }
                page.entries = self.entries(
                    page.variants,
                    [],
                    frozenset(v.model for v in page.variants),
                    f"model.<mesh>.{key}",
                    (fi,),
                )
                self.pages.append(page)
            else:
                model.entries.append(
                    self.single(mod.ModelSettings, key, "model.<mesh>")
                )
        self.pages = [c for p in self.pages for c in self.split(p)]
        _follow_aliases(self.pages)
        self.pages.append(self._output_variables())
        return self.pages

    def split(self, page: Page) -> List[Page]:
        """The page followed by a child page for each block with keys, recursively."""
        out = [page]
        order = 0
        for entry in page.entries:
            if entry.get("link") or not entry["fields"]:
                continue
            order += 10
            out += self.split(self._child(page, entry, order))
        return out

    def _child(self, page: Page, entry: dict, order: int) -> Page:
        key = entry["key"]
        route = f"{page.route}/{kebab(key)}"
        container = entry["facts"].get("container")
        instance = [INSTANCE[container]] if container else []
        child = Page(
            route,
            deck_key(key),
            "node",
            page.route,
            order,
            page.deck_path + [key] + instance,
            entry["description"],
            split_from=(page.route, key),
            type_value=page.type_value,
        )
        child.extra_deck_paths = [p + [key] + instance for p in page.extra_deck_paths]
        child.variants = list(self.block_variants.get(id(entry), []))
        child.facts = {
            "container": container,
            "applies_when": entry["facts"].get("applies_when"),
            "overridable_per_model": page.facts.get("overridable_per_model", False),
            "required": entry["facts"].get("required", False),
        }
        depth = len(entry["path"])
        aliases = []
        child.entries = copy.deepcopy(entry["fields"])
        for moved, old in zip(_walk(child.entries), _walk(entry["fields"])):
            moved["path"] = moved["path"][depth:]
            moved["anchor"] = fragment(moved["path"])
            self.block_variants[id(moved)] = self.block_variants.get(id(old), [])
            aliases.append(
                {
                    "from": f"{page.route}#{old['anchor']}",
                    "route": route,
                    "field": moved["anchor"],
                }
            )
        child.facts["aliases"] = aliases
        entry["link"] = route
        entry["fields"] = []
        return child

    def _typed(self, key: str, fi: FieldInfo, route: str, order: int) -> None:
        named = shape_of(fi, key).container == "named"
        deck_path = ["model", "<mesh>", key] + (["<name>"] if named else [])
        index = Page(
            route,
            key.replace("_", " "),
            "type-index",
            "reference/model",
            order,
            deck_path,
            fi.description,
        )
        self.pages.append(index)
        location = f"model.<mesh>.{key}"
        try:
            disc, leaves = resolve_union(shape_value_annotation(fi), location)
        except UnresolvedUnionError as err:
            self._record(err)
            return
        groups: Dict[str, List[Variant]] = {}
        for leaf in leaves:
            groups.setdefault(leaf.tag, []).append(leaf)
        for n, (tag, group) in enumerate(groups.items(), start=1):
            type_route = f"{route}/{kebab(tag)}"
            doc = next(
                (g.model.__doc__.strip() for g in group if g.model.__doc__), None
            )
            option = self._side_option(group[0].model, disc, str(tag))
            index.types.append(
                {
                    "value": tag,
                    "route": type_route,
                    "models": [g.model.__name__ for g in group],
                    "description": doc,
                    "what": self._option_what(group[0].model, disc, str(tag), option),
                    "when": option.attrs.get("when") or None if option else None,
                }
            )
            page = Page(
                type_route,
                tag,
                "type",
                route,
                n * 10,
                deck_path,
                doc,
                type={"discriminator": disc, "value": tag},
                type_value=tag,
            )
            page.variants = [
                Variant(g.model, g.tag, {k: v for k, v in g.when.items() if k != disc})
                for g in group
            ]
            for g in group:
                for k in g.when:
                    if k in g.model.model_fields:
                        self.discriminators[g.model].add(k)
                        self.discriminators[declaring_class(g.model, k)].add(k)
            page.facts = {
                "container": "named" if named else None,
                "applies_when": None,
                "overridable_per_model": False,
                "required": False,
            }
            page.entries = self.entries(
                page.variants,
                [],
                frozenset(g.model for g in group),
                f"{location}.<name>[{tag}]" if named else f"{location}[{tag}]",
            )
            self.pages.append(page)

    def _output_variables(self) -> Page:
        page = Page(
            "reference/output-variables",
            "Output variables",
            "output-variables",
            None,
            40,
            [],
            OUTPUT_VARIABLES_LEAD,
        )
        aliases: Dict[str, List[str]] = defaultdict(list)
        for f in ov.OUTPUT_FIELDS:
            if f.alias_of:
                aliases[f.alias_of].append(f.name)
        moved: List[dict] = []
        canonical = [f for f in ov.OUTPUT_FIELDS if not f.alias_of]
        for group in _families(canonical):
            first = group[0]
            if len(group) == 1:
                page.entries.append(
                    _variable(
                        first.name,
                        first.category,
                        first.volume_tier,
                        first.surface_tier,
                        first.description,
                        aliases[first.name],
                    )
                )
                continue
            family = _family(group)
            page.entries.append(family)
            moved += [
                {
                    "from": f"reference/output-variables#{fragment([f.name])}",
                    "field": family["anchor"],
                }
                for f in group
            ]
        for prefix, template in ov._DYNAMIC_FIELDS:
            entry = _variable(
                f"{prefix}<n>",
                "solver",
                "base",
                "base",
                template.format("`<n>`"),
                [],
            )
            entry["path"] = [f"{prefix}n"]
            entry["anchor"] = fragment(entry["path"])
            entry["facts"]["pattern"] = True
            entry["pattern"] = _equation_count_text()
            page.entries.append(entry)
        page.facts = {
            "aliases": [
                {"from": f"reference/output-variables#{kebab(a)}", "field": kebab(k)}
                for k, names in aliases.items()
                for a in dict.fromkeys(n.lower() for n in names)
            ]
            + moved
        }
        return page

    @staticmethod
    def _dedupe(pairs: list) -> list:
        out, seen = [], set()
        for when, model in pairs:
            if model in seen:
                continue
            seen.add(model)
            out.append(({k: v for k, v in when.items() if v is not None}, model))
        return out


def _mark_applicability(
    entry: dict,
    sub: List[Variant],
    held_by: Dict[type, List[Variant]],
    holders: List[Variant],
) -> None:
    """Label each discriminator option offered by only some of the blocks that hold the union."""
    disc = entry["union"]["discriminator"]
    field = next((f for f in entry["fields"] if f["key"] == disc), None)
    if field is None or not field.get("options"):
        return
    for option in field["options"]:
        chosen: List[Variant] = []
        for leaf in sub:
            if leaf.tag == option["value"]:
                chosen += [v for v in held_by[leaf.model] if v not in chosen]
        option["only"] = _only(chosen, holders)


def _only(chosen: List[Variant], holders: List[Variant]) -> Optional[str]:
    """The condition that sets ``chosen`` apart from the other ``holders``, as an 'only' label."""
    if not chosen or len(chosen) == len(holders):
        return None
    for key in chosen[0].when:
        if not all(key in v.when for v in holders):
            continue
        mine = list(dict.fromkeys(str(v.when[key]) for v in chosen))
        others = {str(v.when[key]) for v in holders if v not in chosen}
        if not others & set(mine):
            return " / ".join(mine) + " only"
    return None


INSTANCE = {"named": "<name>", "list": "<n>"}


def _follow_aliases(pages: List[Page]) -> None:
    """Point each alias at where its target finally landed, through every later split."""
    moved = {
        a["from"]: a for p in pages for a in p.facts.get("aliases", []) if "route" in a
    }
    for alias in moved.values():
        nxt = moved.get(f"{alias['route']}#{alias['field']}")
        while nxt is not None:
            alias["route"], alias["field"] = nxt["route"], nxt["field"]
            nxt = moved.get(f"{alias['route']}#{alias['field']}")


def _walk(entries: List[dict]):
    for entry in entries:
        yield entry
        yield from _walk(entry.get("fields") or [])


def _selected(holders: tuple, key: str) -> Any:
    """The discriminator value every holder's default selects, else None."""
    tags = [selected_tag(fi, key) for fi in holders]
    if not tags or None in tags or len(set(tags)) != 1:
        return None
    return tags[0]


def _description(occ: list, key: str) -> Optional[str]:
    """The first description of a key, inherited from a base that declares it when a variant restates it bare."""
    for _, fi in occ:
        if fi.description:
            return fi.description
    for v, _ in occ:
        for cls in v.model.__mro__:
            if declares(cls, key) and cls.model_fields[key].description:
                return cls.model_fields[key].description
    return None


OUTPUT_VARIABLES_LEAD = (
    "Every variable zCFD can write, grouped by family. Name volume variables "
    "in `volume variables` and surface variables in `surface variables`, both "
    "under output settings › solution. Each row says whether the variable is "
    "written to volume output, surface output or both."
)

_NUMBERED = re.compile(r"^(?P<stem>.+_)(?P<token>\d+)$")
_AXIS = re.compile(r"^(?P<stem>.+)(?P<token>[xyz])$")


def _family_key(field: Any, names: set) -> Optional[tuple]:
    """(stem, placeholder) for a field that is one member of a numbered or per-axis family."""
    numbered = _NUMBERED.match(field.name)
    if numbered:
        return numbered["stem"], "n"
    axis = _AXIS.match(field.name)
    if axis and axis["stem"] in names:
        return axis["stem"], "axis"
    return None


def _families(fields: List[Any]) -> List[List[Any]]:
    """The fields in registry order, each numbered or per-axis family gathered at its first member."""
    names = {f.name for f in fields}
    groups: Dict[tuple, List[Any]] = {}
    out: List[List[Any]] = []
    for f in fields:
        key = _family_key(f, names)
        if key is None:
            out.append([f])
            continue
        key = key + (f.category, f.volume_tier, f.surface_tier)
        if key not in groups:
            groups[key] = []
            out.append(groups[key])
        groups[key].append(f)
    singles: List[List[Any]] = []
    for group in out:
        if len(group) > 1 and _template(group) is not None:
            singles.append(group)
        else:
            singles += [[f] for f in group]
    return singles


def _token(field: Any) -> str:
    match = _NUMBERED.match(field.name) or _AXIS.match(field.name)
    return match["token"]


def _template(group: List[Any]) -> Optional[str]:
    """The members' shared description with the member's own number or axis as ``{}``, else None."""
    templates = {
        re.sub(rf"\b{re.escape(_token(f))}\b", "{}", f.description, count=1)
        for f in group
    }
    if len(templates) != 1:
        return None
    template = templates.pop()
    return template if "{}" in template else None


def _family(group: List[Any]) -> dict:
    first = group[0]
    stem, placeholder = _family_key(first, {first.name[:-1]})
    tokens = [_token(f) for f in group]
    entry = _variable(
        f"{stem}<{placeholder}>",
        first.category,
        first.volume_tier,
        first.surface_tier,
        _template(group).format(f"`<{placeholder}>`"),
        [],
    )
    entry["path"] = [f"{stem.rstrip('_')}_{placeholder}"]
    entry["anchor"] = fragment(entry["path"])
    entry["facts"]["pattern"] = True
    if placeholder == "n":
        entry["pattern"] = (
            f"Replace `<n>` with a number from {tokens[0]} to {tokens[-1]}."
        )
    else:
        entry["pattern"] = f"Replace `<axis>` with {_names(tokens)}."
    return entry


def _names(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _equation_count_text() -> str:
    """Which equation numbers a case solves, read from the solver's equation counts."""
    by_count: Dict[int, List[str]] = defaultdict(list)
    for equation, count in ov.EQUATION_TO_NUM_EQ.items():
        by_count[count].append(equation)
    counts = [
        f"{count} with {_names(equations)}"
        for count, equations in sorted(by_count.items())
    ]
    counts += [
        f"{count} with the `{model}` turbulence model"
        for model, count in ov.TURBULENCE_MODEL_NUM_EQ.items()
    ]
    counts[0] = counts[0].replace(" with ", " equations with ", 1)
    return (
        "Replace `<n>` with the number of a solved equation, counting from 1. "
        f"A case solves {_and(counts)}."
    )


def _and(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _variable(
    name: str,
    category: str,
    volume: Optional[str],
    surface: Optional[str],
    description: str,
    aliases: List[str],
) -> dict:
    label = ov.CATEGORIES_BY_KEY.get(category)
    return {
        "key": name,
        "path": [name],
        "anchor": fragment([name]),
        "title": name,
        "facts": {
            "category": category,
            "category_label": label.label if label else None,
            "category_hint": label.hint if label else None,
            "volume": volume,
            "surface": surface,
            "context": "both"
            if volume and surface
            else "volume"
            if volume
            else "surface",
            "pattern": False,
        },
        "pattern": None,
        "aliases": aliases,
        "description": description or None,
        "gap": not description,
    }


def shape_value_annotation(fi: FieldInfo) -> Any:
    core, _ = unwrap(fi.annotation)
    if get_origin(core) is dict:
        return get_args(core)[1]
    return fi.annotation


def _side_rel(module: str) -> str:
    return module.replace(".", "/") + ".md"
