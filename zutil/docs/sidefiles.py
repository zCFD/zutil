"""
Side files: the Markdown beside each schema module that owns per-option
what/when, the demonstration id and the guidance link for each dict key.

Format: one ``#`` section per model (class name), one ``##`` per dict key,
one ``###`` per option value of a Literal field; ``name: value`` lines
under each heading.

A ``was:`` line under a key names an anchor the key used to have, written
either as ``reference/solver/equations/turbulence#model`` or as
``#old-name``, which covers an old fragment on every page that shows the key;
a key may carry several. The build copies each into the manifest's
``aliases``.

Every entry is optional: a side file holds only what it adds. An entry for a
model, key or option the module does not define, a stale fingerprint, text
outside an attribute, or a ``what`` for an option the schema already defines
fails the docs build.
"""

import importlib
import inspect
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .anchors import normalise_key
from .introspect import (
    fingerprint,
    is_model,
    literal_values,
    option_definitions,
    own_fields,
)

ATTRIBUTES = ("fingerprint", "guide", "demo", "what", "when", "was")
_LINE = re.compile(r"^(" + "|".join(ATTRIBUTES) + r"):\s?(.*)$")


PROSE = ("what", "when")

_MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>~])")


def heading_text(line: str, level: int) -> str:
    """A heading's text with Markdown backslash escapes removed, as a formatter may add them."""
    return _MARKDOWN_ESCAPE.sub(r"\1", line[level + 1 :].strip())


@dataclass
class OptionEntry:
    value: str
    attrs: Dict[str, str] = field(default_factory=dict)
    unplaced: List[str] = field(default_factory=list)


@dataclass
class KeyEntry:
    key: str
    heading: str
    line: int
    attrs: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, OptionEntry] = field(default_factory=dict)
    unplaced: List[str] = field(default_factory=list)
    was: List[str] = field(default_factory=list)


@dataclass
class ModelSection:
    name: str
    keys: Dict[str, KeyEntry] = field(default_factory=dict)


@dataclass
class SideFile:
    path: Path
    module: str
    sections: Dict[str, ModelSection] = field(default_factory=dict)


def parse(text: str, path: Path = Path("<memory>"), module: str = "") -> SideFile:
    """Parse a side file; a later duplicate heading is merged into the first."""
    side = SideFile(path, module)
    section: Optional[ModelSection] = None
    key: Optional[KeyEntry] = None
    target = None
    last_attr: Optional[str] = None
    blank = False
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if line.startswith("### "):
            if key is None:
                continue
            value = heading_text(line, 3)
            target, last_attr = key.options.setdefault(value, OptionEntry(value)), None
        elif line.startswith("## "):
            if section is None:
                continue
            heading = heading_text(line, 2)
            key = section.keys.setdefault(
                normalise_key(heading),
                KeyEntry(normalise_key(heading), heading, number),
            )
            target, last_attr = key, None
        elif line.startswith("# "):
            name = heading_text(line, 1)
            section = side.sections.setdefault(name, ModelSection(name))
            key, target, last_attr = None, None, None
        elif target is not None:
            match = _LINE.match(line)
            if match and match.group(1) == "was":
                last_attr = "was"
                if isinstance(target, KeyEntry):
                    target.was.append(match.group(2).strip())
                else:
                    target.unplaced.append(line)
            elif match:
                last_attr = match.group(1)
                target.attrs[last_attr] = match.group(2).strip()
            elif not line.strip():
                blank = True
                continue
            elif last_attr in PROSE:
                joiner = "\n\n" if blank else " "
                target.attrs[last_attr] = (
                    target.attrs[last_attr] + joiner + line.strip()
                ).strip()
            else:
                target.unplaced.append(line)
        blank = False
    return side


def module_path(module: str) -> Path:
    return Path(inspect.getfile(importlib.import_module(module)))


def side_file_path(module: str) -> Path:
    return module_path(module).with_suffix(".md")


def load(module: str) -> Optional[SideFile]:
    path = side_file_path(module)
    if not path.exists():
        return None
    return parse(path.read_text(), path, module)


def is_fixed_discriminator(
    model: type, key: str, discriminators: Dict[type, set]
) -> bool:
    values = literal_values(model.model_fields[key].annotation) or []
    return len(values) == 1 and key in discriminators.get(model, set())


def check(side: SideFile, models: List[type]) -> dict:
    """Entries a side file holds for keys, options or models its module does not define, and stale fingerprints."""
    report: Dict[str, list] = {
        k: []
        for k in (
            "unknown_sections",
            "extra_keys",
            "extra_options",
            "stale_fingerprints",
            "unplaced_text",
            "restated_options",
        )
    }
    by_name = {m.__name__: m for m in models}
    file = str(side.path)
    for name, section in side.sections.items():
        if name not in by_name:
            report["unknown_sections"].append({"file": file, "section": name})
        for key, entry in section.keys.items():
            for owner, lines in [(None, entry.unplaced)] + [
                (o.value, o.unplaced) for o in entry.options.values()
            ]:
                if lines:
                    report["unplaced_text"].append(
                        {
                            "file": file,
                            "model": name,
                            "key": entry.heading,
                            "option": owner,
                            "lines": lines,
                        }
                    )
    for model in models:
        section = side.sections.get(model.__name__)
        keys = own_fields(model)
        present = section.keys if section else {}
        for key in keys:
            entry = present.get(key)
            if entry is None:
                continue
            options = [
                str(v) for v in literal_values(model.model_fields[key].annotation) or []
            ]
            defined = option_definitions(model.model_fields[key])
            for option, option_entry in entry.options.items():
                where = {
                    "file": file,
                    "model": model.__name__,
                    "key": key,
                    "option": option,
                }
                if option not in options:
                    report["extra_options"].append(where)
                elif option in defined and option_entry.attrs.get("what"):
                    report["restated_options"].append(where)
            stored = entry.attrs.get("fingerprint", "pending")
            current = fingerprint(model.model_fields[key], key)
            if stored != "pending" and stored != current:
                report["stale_fingerprints"].append(
                    {
                        "file": file,
                        "model": model.__name__,
                        "key": key,
                        "stored": stored,
                        "current": current,
                    }
                )
        for key in present:
            if key not in keys:
                report["extra_keys"].append(
                    {"file": file, "model": model.__name__, "key": present[key].heading}
                )
    return report


def pending_count(side: SideFile) -> int:
    return sum(
        1
        for s in side.sections.values()
        for k in s.keys.values()
        if k.attrs.get("fingerprint", "pending") == "pending"
    )


def restamp(text: str, targets: Dict[tuple, str], only_pending: bool = False) -> tuple:
    """Rewrite the fingerprint line of each (model, key) in ``targets``; returns (text, stamped keys)."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    headings = []
    section = None
    for i, line in enumerate(lines):
        if line.startswith("# "):
            section = heading_text(line, 1)
            headings.append((i, None))
        elif line.startswith("## "):
            headings.append((i, (section, normalise_key(heading_text(line, 2)))))
        elif line.startswith("### "):
            headings.append((i, None))
    headings.append((len(lines), None))
    edits = {}
    stamped = []
    for (start, target), (stop, _) in zip(headings, headings[1:]):
        if target not in targets:
            continue
        found = None
        for j in range(start + 1, stop):
            if lines[j].startswith("fingerprint:"):
                found = j
                break
        if found is not None:
            if only_pending and lines[found].split(":", 1)[1].strip() != "pending":
                continue
            edits[found] = f"fingerprint: {targets[target]}\n"
        else:
            edits[start] = lines[start] + f"fingerprint: {targets[target]}\n"
        stamped.append(target)
    return "".join(edits.get(i, line) for i, line in enumerate(lines)), stamped


def stamp(
    names: List[str],
    modules: List[str],
    new_files: bool = False,
    paths: Optional[List[Path]] = None,
) -> List[str]:
    """Stamp the named entries (``Model.key``), or every pending entry in new side files."""
    classes = _classes_by_name(modules)
    done: List[str] = []
    if new_files:
        files = paths or [
            side_file_path(m)
            for m in modules
            if side_file_path(m).exists() and not _tracked(side_file_path(m))
        ]
        for path in files:
            module = _module_for(path, modules)
            side = parse(path.read_text(), path, module)
            targets = {}
            for section in side.sections.values():
                model = classes.get(section.name)
                if model is None:
                    continue
                for key, entry in section.keys.items():
                    if (
                        key in model.model_fields
                        and entry.attrs.get("fingerprint", "pending") == "pending"
                    ):
                        targets[(section.name, key)] = fingerprint(
                            model.model_fields[key], key
                        )
            text, stamped = restamp(path.read_text(), targets, only_pending=True)
            path.write_text(text)
            done += [f"{m}.{k}" for m, k in stamped]
        return done
    for name in names:
        model_name, _, key = name.partition(".")
        model = classes.get(model_name)
        key = normalise_key(key)
        if model is None or key not in model.model_fields:
            raise KeyError(f"no schema field {name}")
        path = side_file_path(model.__module__)
        text, stamped = restamp(
            path.read_text(),
            {(model_name, key): fingerprint(model.model_fields[key], key)},
        )
        if not stamped:
            raise KeyError(f"{name} has no entry in {path}")
        path.write_text(text)
        done.append(name)
    return done


def scaffold(
    models: List[type], side: Optional[SideFile], discriminators: Dict[type, set]
) -> str:
    """Markdown for every model, key and option the side file lacks."""
    chunks = []
    for model in models:
        section = side.sections.get(model.__name__) if side else None
        lines = []
        for key in own_fields(model):
            if is_fixed_discriminator(model, key, discriminators):
                continue
            present = section.keys.get(key) if section else None
            options = [
                str(v) for v in literal_values(model.model_fields[key].annotation) or []
            ]
            missing = [o for o in options if not present or o not in present.options]
            if present and not missing:
                continue
            if not present:
                lines += [f"## {key}", "fingerprint: pending", ""]
            else:
                lines += [f"## {key}", ""]
            for option in missing:
                lines += [f"### {option}", "what:", "when:", ""]
        if lines:
            chunks.append("\n".join([f"# {model.__name__}", ""] + lines))
    return "\n".join(chunks)


def _classes_by_name(modules: List[str]) -> Dict[str, type]:
    out = {}
    for module in modules:
        mod = importlib.import_module(module)
        for name, obj in vars(mod).items():
            if is_model(obj) and obj.__module__ == module:
                out[name] = obj
    return out


def _module_for(path: Path, modules: List[str]) -> str:
    for module in modules:
        if side_file_path(module).resolve() == path.resolve():
            return module
    raise KeyError(f"{path} is not beside a schema module")


def _tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path.name],
        cwd=path.parent,
        capture_output=True,
    )
    return result.returncode == 0
