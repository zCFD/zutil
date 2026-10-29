"""Convert one v1 control deck to config version 2."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib.abc
import importlib.machinery
import importlib.util
import os
import re
import sys
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Union

from ..validate.validation_utils import lowercase_keys_recursive
from . import report
from .mesh import read_mesh
from .model import (
    UNCONVERTED,
    Conversion,
    ConversionError,
    MeshInfo,
    V1Deck,
    Value,
    norm_key,
)


PathLike = Union[str, os.PathLike]

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class LegacyDeckError(Exception):
    """The v1 deck fails the legacy validator, so it is not converted."""

    def __init__(self, deck_name: str, errors: list[str]) -> None:
        self.deck_name = deck_name
        self.errors = errors
        lines = [
            f"{deck_name} is not a valid v1 deck, so update_input will not convert it:"
        ]
        lines += [f"  {e}" for e in errors]
        lines.append(
            "Fix these in the v1 deck, or convert it as it is with --ignore-legacy-errors."
        )
        super().__init__("\n".join(lines))


class ConverterBugError(Exception):
    """A conversion with no open decisions produced a deck that does not validate."""

    def __init__(self, deck_name: str, errors: str) -> None:
        self.deck_name = deck_name
        self.errors = errors
        super().__init__(
            f"update_input converted {deck_name}, but the result is wrong:\n"
            f"{errors}\n"
            f"This is a bug in update_input, not in your deck. "
            f"Please send {deck_name} to Zenotech so it can be fixed."
        )


@dataclass
class ConversionResult:
    """The new deck text and everything said about how it was made.

    ``validated`` is true only when the deck had no open decisions and the
    new text passed ``validate_zcfd_input``. A base that the legacy validator
    rejects on its own is not a whole deck, so it is checked only through
    its variant. ``bases`` are the converted decks this one takes
    its parameters from, innermost first; ``summary`` covers them all.
    For an overset group, ``members`` are the v1 decks its pairs name, with
    their bases; they are read, and nothing is written for them.
    """

    text: str
    report: str
    summary: str
    conversion: Conversion
    validated: bool
    output_path: Path
    report_path: Path
    deck_path: Optional[Path] = None
    bases: list["ConversionResult"] = field(default_factory=list)
    own_summary: str = ""
    members: list[Path] = field(default_factory=list)
    deck_file: object = None

    @property
    def files(self) -> list[tuple[Path, str]]:
        """Every file to write, with its text, bases first."""
        out = []
        for result in self.bases + [self]:
            out.append((result.output_path, result.text))
            out.append((result.report_path, result.report))
        return out


def default_output(deck_path: PathLike) -> Path:
    deck_path = Path(deck_path)
    return deck_path.with_name(f"{deck_path.stem}_v2.py")


def report_path_for(output_path: PathLike) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}.conversion.md")


@contextlib.contextmanager
def in_deck_dir(deck_path: Path) -> Iterator[None]:
    """Run with the deck's directory as the working directory and first on ``sys.path``."""
    deck_dir = str(Path(deck_path).resolve().parent)
    cwd = os.getcwd()
    sys.path.insert(0, deck_dir)
    os.chdir(deck_dir)
    try:
        yield
    finally:
        os.chdir(cwd)
        with contextlib.suppress(ValueError):
            sys.path.remove(deck_dir)


def _legacy_error_text(err: BaseException) -> str:
    msg = _ANSI.sub("", str(getattr(err, "msg", err)))
    path = getattr(err, "path", None)
    if path:
        return " > ".join(str(k) for k in path) + ": " + msg
    return msg


UNLISTED_V1_KEYS = ("mesh transform function",)


def legacy_errors(evaluated: dict, deck_path: Path) -> tuple[list[str], list[str]]:
    """The legacy validator's errors for the evaluated v1 dict, both empty when it passes.

    Returns the errors that make the deck invalid, then the errors that only
    say a file the deck names is missing or unreadable from here. Keys in
    ``UNLISTED_V1_KEYS`` were read by the v1 solver but never listed in the
    legacy schema, so they are left out of the check.
    """
    from voluptuous.error import FileInvalid
    from zutil.validate.zcfd_validate_legacy import validate

    try:
        data = copy.deepcopy(evaluated)
    except Exception:
        data = dict(evaluated)
    data = {
        k: v
        for k, v in data.items()
        if not (isinstance(k, str) and norm_key(k) in UNLISTED_V1_KEYS)
    }
    try:
        with in_deck_dir(deck_path), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate(lowercase_keys_recursive(data), case_name=str(deck_path))
    except Exception as e:
        errors = getattr(e, "errors", None)
        if not (isinstance(errors, list) and errors):
            errors = [e]
        files = [err for err in errors if isinstance(err, FileInvalid)]
        return (
            [_legacy_error_text(err) for err in errors if err not in files],
            [_legacy_error_text(err) for err in files],
        )
    return [], []


def _config_version(evaluated: dict) -> object:
    return evaluated.get("config_version", evaluated.get("config version", 0))


def _set_mesh(c: Conversion, deck_path: Path, named: Optional[str] = None) -> None:
    v2_path = ("model", c.model, "mesh")
    if named is not None:
        c.set(v2_path, Value(named))
        return
    if c.mesh is not None:
        c.set(v2_path, Value(c.mesh.path))
        return
    c.decide(
        [],
        v2_path,
        "Which mesh file does this deck run on?",
        details=(
            "v1 decks take the mesh from the command line. "
            f"Run update_input {deck_path.name} --mesh <mesh>.h5, or write the mesh file name here."
        ),
    )


class _StagedFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves converted base decks that are not yet on disk, by module name."""

    def __init__(self, staged: dict[Path, str]) -> None:
        self.by_name = {p.stem: (p, text) for p, text in staged.items()}

    def find_spec(
        self, name: str, path: object = None, target: object = None
    ) -> Optional[importlib.machinery.ModuleSpec]:
        if name not in self.by_name:
            return None
        return importlib.util.spec_from_loader(
            name, self, origin=str(self.by_name[name][0])
        )

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        file, text = self.by_name[module.__name__]
        module.__file__ = str(file)
        exec(compile(text, str(file), "exec"), module.__dict__)


def _staged_path(control_file_path: object, namespace: Optional[dict]) -> Path:
    named = str(control_file_path)
    if not named.endswith(".py"):
        named += ".py"
    path = Path(named)
    if not path.is_absolute():
        anchor = (namespace or {}).get("__file__")
        path = (Path(anchor).parent if anchor else Path.cwd()) / path
    return path.resolve()


@contextlib.contextmanager
def _staged(staged: dict[Path, str]) -> Iterator[None]:
    """Let a deck load the converted bases in ``staged`` as if they were written."""
    import zutil
    import zutil.fileutils as fileutils

    staged = {p.resolve(): text for p, text in staged.items()}
    original = fileutils.get_parameters_from_file
    exported = getattr(zutil, "get_parameters_from_file", None)

    def loader(control_file_path: object, *args: object, **kwargs: object) -> dict:
        namespace = kwargs.get("namespace", args[1] if len(args) > 1 else None)
        path = _staged_path(control_file_path, namespace)
        if path not in staged:
            return original(control_file_path, *args, **kwargs)
        ns = dict(namespace) if namespace else {}
        ns["__file__"] = str(path)
        exec(compile(staged[path], str(path), "exec"), ns)
        return ns["parameters"]

    finder = _StagedFinder(staged)
    names = set(finder.by_name)
    for name in names:
        sys.modules.pop(name, None)
    sys.meta_path.insert(0, finder)
    fileutils.get_parameters_from_file = loader
    zutil.get_parameters_from_file = loader
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        fileutils.get_parameters_from_file = original
        if exported is not None:
            zutil.get_parameters_from_file = exported
        for name in names:
            sys.modules.pop(name, None)


def _check_v2(
    text: str, deck_path: Path, output_path: Path, staged: dict[Path, str]
) -> None:
    from zutil.validate import in_case_directory, validate_zcfd_input

    namespace = {"__file__": str(output_path), "__name__": "__update_input__"}
    with in_deck_dir(deck_path), _staged(staged):
        try:
            exec(compile(text, str(output_path), "exec"), namespace)
        except Exception as e:
            raise ConverterBugError(
                deck_path.name, f"the new deck does not run: {type(e).__name__}: {e}"
            ) from e
        parameters = namespace.get("parameters")
        if not isinstance(parameters, dict):
            raise ConverterBugError(
                deck_path.name, "the new deck sets no parameters dict"
            )
        try:
            with in_case_directory(None):
                validate_zcfd_input(parameters)
        except Exception as e:
            raise ConverterBugError(deck_path.name, str(e)) from e


def mesh_value(
    mesh_path: PathLike, output_path: PathLike, base: Optional[PathLike] = None
) -> str:
    """The mesh as the new deck names it.

    An absolute path stays as given. A relative path, taken from ``base`` (the
    working directory by default), is written relative to the new deck when
    the mesh sits under the new deck's directory, and absolute otherwise.
    """
    given = Path(mesh_path)
    if given.is_absolute():
        return given.as_posix()
    mesh = Path(os.path.abspath(Path(base or os.getcwd()) / given))
    out_dir = Path(os.path.abspath(output_path)).parent
    if mesh.is_relative_to(out_dir):
        return mesh.relative_to(out_dir).as_posix()
    return mesh.as_posix()


_TOP_LEVEL_ORDER = ("config version", "solver", "model")


def _top_level_rank(path: tuple) -> int:
    if path[0] in _TOP_LEVEL_ORDER:
        return _TOP_LEVEL_ORDER.index(path[0])
    if path[0] == UNCONVERTED:
        return len(_TOP_LEVEL_ORDER) + 1
    return len(_TOP_LEVEL_ORDER)


def ordered_v2(v2: dict) -> dict:
    """The v2 tree with its top-level blocks in deck order, each block's order kept."""
    return dict(sorted(v2.items(), key=lambda item: _top_level_rank(item[0])))


@dataclass
class _Run:
    """What every deck of one set shares during a single conversion."""

    label: str
    mesh: Optional[MeshInfo]
    mesh_file: Optional[Path]
    ignore_legacy_errors: bool
    root_dir: Path
    done: dict[Path, ConversionResult] = field(default_factory=dict)
    staged: dict[Path, str] = field(default_factory=dict)
    mesh_name: Optional[str] = None
    standalone: bool = True
    member_decks: tuple[Path, ...] = ()

    def name(self, path: Path) -> str:
        return relative(path, self.root_dir)


def relative(path: PathLike, root: Path) -> str:
    """``path`` relative to ``root``, as a POSIX string."""
    return Path(os.path.relpath(Path(path).resolve(), root)).as_posix()


def deck_chain(deck_path: Path, deck_file: object) -> list[Path]:
    """``deck_path`` and the decks it takes its parameters from, nearest first."""
    from . import source

    chain = [deck_path]
    seen = {deck_path.resolve()}
    while True:
        related = source.related_decks(deck_file)
        if not related:
            return chain
        path = Path(related[0])
        if path.resolve() in seen:
            return chain
        seen.add(path.resolve())
        chain.append(path)
        _, deck_file = source.read_related(deck_file, path)


def _base_output(source: types.ModuleType, base_path: Path) -> Path:
    return base_path.with_name(source.v2_name(base_path.name))


def _convert_one(
    run: _Run,
    deck_path: Path,
    v1: V1Deck,
    deck_file: object,
    output_path: Path,
    variant: str = "",
) -> ConversionResult:
    from . import rules, source

    bases: list[ConversionResult] = []
    base_v2 = None
    for related in source.related_decks(deck_file) if run.standalone else []:
        base_path = Path(related)
        key = base_path.resolve()
        base = run.done.get(key)
        if base is None:
            base_v1, base_file = source.read_related(deck_file, base_path)
            base = _convert_one(
                run,
                base_path,
                base_v1,
                base_file,
                _base_output(source, base_path),
                variant=run.name(deck_path),
            )
            run.done[key] = base
        for result in base.bases + [base]:
            if all(result is not b for b in bases):
                bases.append(result)
        base_v2 = base.conversion.v2

    errors, missing_files = legacy_errors(v1.evaluated, deck_path)
    if errors and not variant and not run.ignore_legacy_errors:
        raise LegacyDeckError(deck_path.name, errors)

    mesh = None
    if run.mesh is not None:
        mesh = dataclasses.replace(
            run.mesh, path=mesh_value(run.mesh_file, output_path)
        )
    c = Conversion(v1, run.label, mesh)
    c.legacy_errors = tuple(errors)
    c.deck_files = (
        deck_path.resolve(),
        *(b.deck_path.resolve() for b in bases),
        *(p.resolve() for p in run.member_decks),
    )
    for b in bases:
        c.rewrites.extend(r for r in b.conversion.rewrites if r not in c.rewrites)
    for err in errors:
        if variant:
            c.note(f"The legacy validator rejects this base on its own: {err}")
        else:
            c.note(
                f"The legacy validator rejected the deck, and the conversion went ahead: {err}"
            )
    for err in missing_files:
        c.note(
            f"The legacy validator cannot read a file the deck names from here; "
            f"check it is in place before the run: {err}"
        )

    try:
        _set_mesh(c, deck_path, run.mesh_name)
        for module in rules.rule_modules():
            module.convert(c)
        c.finish(variant)
        c.set(("config version",), Value(2))
    except ConversionError as e:
        raise ConverterBugError(deck_path.name, str(e)) from e
    c.v2 = ordered_v2(c.v2)
    if not run.standalone:
        return ConversionResult(
            text="",
            report="",
            summary="",
            conversion=c,
            validated=False,
            output_path=output_path,
            report_path=report_path_for(output_path),
            deck_path=deck_path,
        )

    for b in bases:
        _rewrite_base_functions(run, b, c)
    text = _render(deck_file, deck_path, c, base_v2)

    validated = False
    unvalidated = ""
    if not c.decisions:
        if variant and errors:
            unvalidated = (
                f"not validated on its own, as it is not a whole deck; "
                f"{variant} is validated with it"
            )
        elif errors:
            try:
                _check_v2(text, deck_path, output_path, run.staged)
                validated = True
            except ConverterBugError as e:
                c.note(f"The new deck fails config version 2 validation: {e.errors}")
                unvalidated = (
                    "not valid config version 2, as the v1 deck had legacy errors "
                    "(see the report)"
                )
        else:
            _check_v2(text, deck_path, output_path, run.staged)
            validated = True
    run.staged[output_path] = text

    report_path = report_path_for(output_path)
    names = report.Names(
        deck=run.name(deck_path),
        output=run.name(output_path),
        report=run.name(report_path),
        config_version=_config_version(v1.evaluated),
        bases=tuple((run.name(b.output_path), run.name(b.report_path)) for b in bases),
        unvalidated=unvalidated,
    )
    own_summary = report.summary(c, names, validated)
    return ConversionResult(
        text=text,
        report=report.markdown(c, names, validated),
        summary="\n\n".join([b.own_summary for b in bases] + [own_summary]),
        conversion=c,
        validated=validated,
        output_path=output_path,
        report_path=report_path,
        deck_path=deck_path,
        bases=bases,
        own_summary=own_summary,
        deck_file=deck_file,
    )


def _render(
    deck_file: object, deck_path: Path, c: Conversion, base_v2: Optional[dict]
) -> str:
    from . import source

    own = deck_path.resolve()
    functions = [(r.start, r.end, r.text) for r in c.rewrites if r.file == own]
    if functions:
        deck_file = dataclasses.replace(
            deck_file, rewrites=[*deck_file.rewrites, *functions]
        )
    return source.render_deck(deck_file, c.v2, base_v2=base_v2)


def _rewrite_base_functions(run: _Run, base: ConversionResult, c: Conversion) -> None:
    own = base.deck_path.resolve()
    added = [
        r for r in c.rewrites if r.file == own and r not in base.conversion.rewrites
    ]
    if not added:
        return
    base.conversion.rewrites.extend(added)
    inner = base.bases[-1].conversion.v2 if base.bases else None
    base.text = _render(base.deck_file, base.deck_path, base.conversion, inner)
    run.staged[base.output_path] = base.text


def convert_deck(
    path: PathLike,
    mesh_path: Optional[PathLike] = None,
    ignore_legacy_errors: bool = False,
    output_path: Optional[PathLike] = None,
) -> ConversionResult:
    """Convert the v1 deck at ``path``, and every base it takes parameters from.

    Nothing is written to disk. ``output_path`` names the new deck in the
    report and the summary; it defaults to ``<stem>_v2.py`` beside the deck.
    Each base is converted once, innermost first, to its ``_v2`` name beside
    it; :attr:`ConversionResult.files` lists everything to write. The whole
    set shares one model label: the mesh's, or the innermost deck's stem.

    An overset group (an ``override`` dict with ``mesh_case_pair``) becomes
    one deck with a model per pair; see :mod:`zutil.update_input.group`.

    Raises:
        LegacyDeckError: a deck fails the legacy validator, unless
            ``ignore_legacy_errors`` is set.
        ConverterBugError: a rule broke its contract, or a deck with no open
            decisions does not pass v2 validation.
        source.DeckReadError: a deck cannot be read, or ``mesh_path`` is
            given for a group.
        source.RenderError: a converted deck cannot be written as source.
    """
    from . import group, source

    deck_path = Path(path).absolute()
    output_path = (
        Path(output_path) if output_path is not None else default_output(deck_path)
    )

    pairs = group.read_group(deck_path)
    if pairs is not None:
        if mesh_path is not None:
            raise source.DeckReadError(
                f"{deck_path.name} is an overset group, which names its own meshes; "
                "run update_input without --mesh"
            )
        return group.convert_group(pairs, ignore_legacy_errors, output_path)

    with in_deck_dir(deck_path):
        v1, deck_file = source.read_deck(deck_path)
    if _config_version(v1.evaluated) == 2:
        raise source.DeckReadError(
            f"{deck_path.name} is already config version 2. "
            f"Check it with: validate_input {deck_path.name}"
        )

    mesh = read_mesh(mesh_path) if mesh_path is not None else None

    run = _Run(
        label=mesh.label if mesh else deck_chain(deck_path, deck_file)[-1].stem,
        mesh=mesh,
        mesh_file=Path(mesh_path) if mesh_path is not None else None,
        ignore_legacy_errors=ignore_legacy_errors,
        root_dir=deck_path.resolve().parent,
    )
    return _convert_one(run, deck_path, v1, deck_file, output_path)


__all__ = [
    "UNCONVERTED",
    "ConversionResult",
    "ConverterBugError",
    "LegacyDeckError",
    "convert_deck",
    "default_output",
    "ordered_v2",
    "report_path_for",
]
