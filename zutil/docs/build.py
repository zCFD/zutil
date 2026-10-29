"""
Build the reference payload: pages, manifest, gap list and reports.
"""

import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pydantic

import zutil

from . import sidefiles
from .examples import derive, items_of, load_corpus, lookup, page_blocks
from .introspect import unknown_option_definitions
from .outline import block_routes, page_outline, walkthrough
from .tree import Page, Walker

FORMAT = 1
PLACEHOLDER = "— no description —"
DEFAULT_TESTCASES = (
    Path(zutil.__file__).resolve().parents[2] / "testcases" / "testcases"
)


class BuildResult:
    def __init__(
        self,
        pages: List[Page],
        walker: Walker,
        gaps: dict,
        mismatches: dict,
        collisions: List[dict],
        corpus: List,
        aliases: Optional[List[dict]] = None,
        alias_errors: Optional[List[str]] = None,
    ) -> None:
        self.pages = pages
        self.walker = walker
        self.gaps = gaps
        self.mismatches = mismatches
        self.collisions = collisions
        self.corpus = corpus
        self.aliases = aliases or []
        self.alias_errors = alias_errors or []

    @property
    def failed(self) -> bool:
        return bool(
            self.walker.unresolved
            or self.collisions
            or self.alias_errors
            or any(self.mismatches[k] for k in DRIFT)
        )


def schema_modules(walker: Walker) -> Dict[str, List[type]]:
    """The reached models grouped by the module that defines them."""
    by_module: Dict[str, List[type]] = defaultdict(list)
    for model in sorted(walker.reached, key=lambda m: (m.__module__, _source_line(m))):
        by_module[model.__module__].append(model)
    return dict(by_module)


def _source_line(model: type) -> int:
    import inspect

    try:
        return inspect.getsourcelines(model)[1]
    except (OSError, TypeError):
        return 0


def load_side_files(modules: List[str]) -> Dict[str, sidefiles.SideFile]:
    out = {}
    for module in modules:
        side = sidefiles.load(module)
        if side is not None:
            out[module] = side
    return out


def walk_entries(entries: List[dict]) -> Iterator[dict]:
    for entry in entries:
        yield entry
        yield from walk_entries(entry.get("fields") or [])


def anchor_collisions(pages: List[Page]) -> List[dict]:
    out = []
    for page in pages:
        seen: Dict[str, list] = defaultdict(list)
        for entry in walk_entries(page.entries):
            seen[entry["anchor"]].append(".".join(entry["path"]))
        for anchor, paths in seen.items():
            if len(set(paths)) > 1:
                out.append(
                    {"route": page.route, "anchor": anchor, "paths": sorted(set(paths))}
                )
    return out


def lift_aliases(pages: List[Page], side_files: Dict[str, sidefiles.SideFile]) -> tuple:
    """The manifest aliases: each page's own, then every side-file ``was:`` line; returns (aliases, errors)."""
    aliases = [a for p in pages for a in p.facts.get("aliases", [])]
    errors: List[str] = []
    shown: Dict[tuple, list] = defaultdict(list)
    for page in pages:
        for entry in walk_entries(page.entries):
            where = entry.get("declared_in")
            if where and not entry.get("link"):
                shown[(where["module"], where["model"], entry["key"])].append(
                    (page.route, entry["anchor"])
                )
    for module, side in side_files.items():
        for section in side.sections.values():
            for key, side_entry in section.keys.items():
                for was in side_entry.was:
                    name = f"{side.path.name}: {section.name}.{side_entry.heading}"
                    places = list(
                        dict.fromkeys(
                            shown.get(
                                (module.rsplit(".", 1)[-1], section.name, key), []
                            )
                        )
                    )
                    lifted, error = _lift(was, places)
                    if error:
                        errors.append(f'{name} was: "{was}" {error}')
                    aliases += lifted
    return aliases, errors


def _lift(was: str, places: List[tuple]) -> tuple:
    route, hash_, fragment = was.lstrip("/").partition("#")
    if not was or (hash_ and not fragment) or " " in was:
        return [], (
            "is not an old anchor; write the page route and fragment, as "
            "reference/solver/equations/turbulence#model, or the fragment alone, as #model"
        )
    if not places:
        return [], "belongs to a key that no reference page shows"
    if not route:
        return [{"from": f"{r}#{fragment}", "field": a} for r, a in places], None
    old = f"{route}#{fragment}" if fragment else route
    same = [a for r, a in places if r == route]
    if same:
        return [{"from": old, "field": same[0]}], None
    if len(places) > 1:
        routes = ", ".join(r for r, _ in places)
        return [], (
            f"names a page that does not show this key, and the key is on {len(places)} pages "
            f"({routes}), so the alias has no single target"
        )
    target, anchor = places[0]
    return [{"from": old, "route": target, "field": anchor}], None


def gap_list(pages: List[Page], walker: Walker, example_gaps: List[dict]) -> dict:
    descriptions: Dict[tuple, dict] = {}
    options: Dict[tuple, dict] = {}
    for page in pages:
        if page.kind == "output-variables":
            for entry in page.entries:
                if entry["gap"]:
                    descriptions.setdefault(
                        ("output_variables", entry["key"]),
                        {
                            "model": "OUTPUT_FIELDS",
                            "module": "output_variables",
                            "key": entry["key"],
                            "anchors": [],
                        },
                    )["anchors"].append(f"{page.route}#{entry['anchor']}")
            continue
        for entry in walk_entries(page.entries):
            where = entry["declared_in"]
            ref = f"{page.route}#{entry['anchor']}"
            if entry["gap"]:
                descriptions.setdefault(
                    (where["model"], entry["key"]),
                    {
                        "model": where["model"],
                        "module": where["module"],
                        "key": entry["key"],
                        "anchors": [],
                    },
                )["anchors"].append(ref)
            if entry.get("link") or not entry.get("options"):
                continue
            if entry["facts"]["discriminator"] and len(entry["options"]) == 1:
                continue
            for option in entry["options"]:
                missing = [k for k in ("what", "when") if not option[k]]
                if "what" in missing:
                    options.setdefault(
                        (where["model"], entry["key"], str(option["value"])),
                        {
                            "model": where["model"],
                            "module": where["module"],
                            "key": entry["key"],
                            "option": option["value"],
                            "missing": missing,
                        },
                    )
    examples: Dict[tuple, dict] = {}
    for gap in example_gaps:
        examples.setdefault((gap["model"], gap["key"]), {**gap, "anchors": []})[
            "anchors"
        ] += gap["anchors"]
    return {
        "description": list(descriptions.values()),
        "option": list(options.values()),
        "example": list(examples.values()),
    }


DRIFT = (
    "unknown_sections",
    "extra_keys",
    "extra_options",
    "stale_fingerprints",
    "unplaced_text",
    "restated_options",
    "unknown_schema_options",
)


def mismatch_report(
    modules: Dict[str, List[type]],
    side_files: Dict[str, sidefiles.SideFile],
    walker: Walker,
) -> dict:
    report: Dict[str, list] = {k: [] for k in DRIFT}
    pending = 0
    for module, models in modules.items():
        for model in models:
            report["unknown_schema_options"] += unknown_option_definitions(model)
        side = side_files.get(module)
        if side is None:
            continue
        part = sidefiles.check(side, models)
        for key, items in part.items():
            report[key] += items
        pending += sidefiles.pending_count(side)
    report["pending_fingerprints"] = pending
    return report


def add_outlines(pages: List[Page]) -> None:
    data = [p.to_json() for p in pages]
    routes = block_routes(data)
    for page, json_page in zip(pages, data):
        page.outline = page_outline(json_page, routes)
        if page.route == "reference/input-deck":
            page.walkthrough = walkthrough(data)


def build(
    out: Optional[Path] = None, testcases: Optional[Path] = DEFAULT_TESTCASES
) -> BuildResult:
    probe = Walker()
    probe.build()
    modules = schema_modules(probe)
    side_files = load_side_files(list(modules))
    walker = Walker(side_files)
    pages = walker.build()
    corpus = load_corpus(testcases) if testcases and testcases.exists() else []
    example_gaps: List[dict] = []
    loaded = sum(1 for d in corpus if d.parameters is not None)
    blocks_at: Dict[str, list] = {}
    for page in pages:
        if page.kind in ("output-variables", "type-index"):
            continue
        if page.split_from:
            parent, key = page.split_from
            blocks = items_of(
                [
                    (deck, value)
                    for deck, block in blocks_at[parent]
                    for ok, value in [lookup(block, key)]
                    if ok
                ],
                page.facts["container"],
            )
        else:
            blocks = page_blocks(corpus, page.to_json(), page.extra_deck_paths)
        blocks_at[page.route] = blocks
        derive(page.entries, blocks, example_gaps, page.route, loaded)
    add_outlines(pages)
    gaps = gap_list(pages, walker, example_gaps)
    mismatches = mismatch_report(modules, side_files, walker)
    aliases, alias_errors = lift_aliases(pages, side_files)
    result = BuildResult(
        pages,
        walker,
        gaps,
        mismatches,
        anchor_collisions(pages),
        corpus,
        aliases,
        alias_errors,
    )
    if out is not None:
        write(result, out)
    return result


def write(result: BuildResult, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    anchors = []
    for page in result.pages:
        data = page.to_json()
        path = out / f"{page.route}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _dump(path, data)
        anchors += [f"{page.route}#{e['anchor']}" for e in walk_entries(page.entries)]
    manifest = {
        "format": FORMAT,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "pydantic": pydantic.VERSION,
        "placeholder": PLACEHOLDER,
        "pages": [
            {
                "route": p.route,
                "title": p.title,
                "kind": p.kind,
                "parent": p.parent,
                "order": p.order,
            }
            for p in result.pages
        ],
        "anchors": anchors,
        "aliases": result.aliases,
    }
    _dump(out / "manifest.json", manifest)
    _dump(out / "gaps.json", result.gaps)
    _dump(
        out / "mismatches.json",
        {**result.mismatches, "anchor_collisions": result.collisions},
    )
    _dump(
        out / "unresolved.json",
        [
            {"location": e.location, "members": e.members, "reason": e.reason}
            for e in result.walker.unresolved
        ],
    )
    _dump(
        out / "corpus.json",
        [
            {
                "case": d.case,
                "controlfile": d.controlfile,
                "label": d.label,
                "loaded": d.parameters is not None,
                "error": d.error,
            }
            for d in result.corpus
        ],
    )


def _dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=repr) + "\n")


def summary(result: BuildResult) -> str:
    entries = sum(1 for p in result.pages for _ in walk_entries(p.entries))
    m = result.mismatches
    lines = [
        f"pages: {len(result.pages)}   entries: {entries}",
        f"gaps: {len(result.gaps['description'])} descriptions, {len(result.gaps['option'])} options, {len(result.gaps['example'])} examples",
        f"corpus: {sum(1 for d in result.corpus if d.parameters is not None)}/{len(result.corpus)} published decks loaded",
        f"side files: {len(m['unknown_sections'])} unknown sections, {len(m['extra_keys'])} unknown keys, {len(m['extra_options'])} unknown options, {len(m['stale_fingerprints'])} stale, {m['pending_fingerprints']} pending, {len(m['unplaced_text'])} blocks of unplaced text, {len(m['restated_options'])} restated options",
    ]
    for err in result.walker.unresolved:
        lines.append(f"UNRESOLVED UNION {err}")
    for kind in DRIFT:
        for item in m[kind]:
            lines.append(f"SIDE FILE {kind.upper().replace('_', ' ')} {item}")
    for err in result.alias_errors:
        lines.append(f"ALIAS {err}")
    for c in result.collisions:
        lines.append(
            f"ANCHOR COLLISION {c['route']}#{c['anchor']}: {', '.join(c['paths'])}"
        )
    return "\n".join(lines)
