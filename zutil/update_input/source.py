"""Read a v1 control deck into leaves, and write the converted deck back.

:func:`read_deck` runs the deck one top-level statement at a time, as the
solver's loader runs it, and walks the statements that build ``parameters``,
so each leaf keeps the text the deck wrote for it. :func:`render_deck` writes
the deck again with only those statements regenerated.

A deck that takes its parameters from another deck file is a variant of that
base: through ``get_parameters_from_file``, ``from base import parameters``,
``{**base.parameters, ...}``, ``dict(base.parameters, ...)`` or a copy of
``base.parameters``. :func:`related_decks` names the base, :func:`read_related`
returns it as read with the namespace the variant handed it, and
:func:`render_deck` writes the variant as its reference to ``<base>_v2`` plus
the difference between the variant's v2 tree and the base's.
"""

from __future__ import annotations

import __future__ as future_features
import ast
import contextlib
import importlib.util
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from .model import Expr, Leaf, Note, Placeholder, Source, V1Deck, V1Path, Value
from .model import V2Path, V2Value, norm_path

PARAMETERS = "parameters"
LOADER = "get_parameters_from_file"
REFUSED_MODULES = ("zcfd", "paraview")

_ANY = "*"
_MISSING = object()
_INDENT = "    "
_LINE = 88

Keys = tuple


class DeckReadError(Exception):
    """The deck cannot be read: it does not run, or it uses what update_input cannot convert."""


class RenderError(Exception):
    """The v2 tree cannot be written as Python source."""


@dataclass(frozen=True)
class BlockOverride(Note):
    """A variant replaces a whole v1 block that its base defines.

    The variant's leaves under its ``v1_paths`` are the replacing block alone, so the
    base's other keys in that block are absent. ``base`` is the base file name.
    """

    base: str = ""


@dataclass
class BaseDeck:
    """The deck a variant takes its parameters from, read as the variant ran it.

    ``form`` is ``"loader"`` for ``get_parameters_from_file`` and ``"import"``
    for the import forms. ``shared`` is true when the base ran in the variant's
    own globals, so the base's names are the variant's names.
    """

    path: Path
    form: str
    shared: bool
    namespace: Optional[dict]
    v1: V1Deck
    file: "DeckFile"


@dataclass
class DeckFile:
    """Where the ``parameters`` statements sit in the deck text, and how to rewrite them.

    ``style`` is ``"literal"`` (the statement at ``target`` becomes one v2
    literal), ``"spread"`` (it becomes a nested ``{**base, ...}`` literal) or
    ``"overrides"`` (v2 assignments over the base's dict go at ``target``).
    ``removed`` are the spans of statements folded into the regenerated text.
    ``rewrites`` replace a span with new text, such as a base's ``_v2`` name.
    ``inherited_text`` maps each v1 path the variant takes unchanged from its
    base to the base's source text for it.
    """

    path: Path
    text: str
    style: str
    target: tuple[int, int]
    removed: list[tuple[int, int]] = field(default_factory=list)
    rewrites: list[tuple[int, int, str]] = field(default_factory=list)
    base: Optional[BaseDeck] = None
    spread_operand: str = ""
    inherited_text: dict[V1Path, Optional[str]] = field(default_factory=dict)
    insert_after: bool = False


def related_decks(deck: DeckFile) -> list[Path]:
    """The v1 deck files this deck takes its parameters from, nearest first."""
    return [deck.base.path] if deck.base is not None else []


def read_related(deck: DeckFile, path: Path) -> tuple[V1Deck, DeckFile]:
    """The related deck at ``path``, read with the namespace this deck handed it."""
    if deck.base is None or deck.base.path.resolve() != Path(path).resolve():
        raise ValueError(f"{deck.path.name} does not take its parameters from {path}")
    return deck.base.v1, deck.base.file


def v2_name(name: str) -> str:
    """The converted file or module name for a v1 deck name, keeping any ``.py``."""
    if name.endswith(".py"):
        return name[:-3] + "_v2.py"
    return name + "_v2"


@dataclass
class _Call:
    raw: Any
    named: Path
    path: Path
    namespace: Optional[dict]
    shared: bool
    result: Any


@dataclass
class _Step:
    stmt: ast.stmt
    params: Any
    flat: dict
    changed: set
    rebound: bool
    calls: list


@dataclass
class _Src:
    text: Optional[str]
    line: int
    origin: int
    names: frozenset
    inherited: bool = False


def read_deck(
    path: Union[str, Path], namespace: Optional[dict] = None
) -> tuple[V1Deck, DeckFile]:
    """Run the deck at ``path`` and return its leaves and its layout.

    ``namespace`` is injected as ``get_parameters_from_file`` injects it; the
    caller's dict is not changed.

    Raises:
        DeckReadError: the deck imports a module only the solver or ParaView
            provides, does not run, or sets no ``parameters`` dict.
    """
    given = Path(path).absolute()
    path = given.resolve()
    text = path.read_bytes().decode("utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        raise DeckReadError(
            f"{path.name} is not valid Python: line {e.lineno}: {e.msg}"
        ) from e
    _refuse_imports(tree, path)
    steps, module_files = _run(path, tree, namespace)
    return _Reader(given, text, tree, steps, module_files).read()


def _refuse_imports(tree: ast.Module, path: Path) -> None:
    for node in _walk_outside_functions(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules = [node.module]
        for module in modules:
            top = module.split(".")[0]
            if top in REFUSED_MODULES:
                raise DeckReadError(
                    f"{path.name} imports {module} at line {node.lineno}. "
                    f"update_input runs the deck to read it, and {top} is only "
                    "available inside the solver or ParaView, so this deck cannot "
                    "be converted. Remove the import, or move it into the function "
                    "that uses it."
                )


def _walk_outside_functions(node: ast.AST) -> Iterator[ast.AST]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _walk_outside_functions(child)


def _future_flags(tree: ast.Module) -> int:
    flags = 0
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            for alias in stmt.names:
                feature = getattr(future_features, alias.name, None)
                if feature is not None:
                    flags |= feature.compiler_flag
    return flags


@contextlib.contextmanager
def _deck_dir(deck_dir: Path) -> Iterator[None]:
    """Put the deck's directory on ``sys.path``, and forget the modules it imported."""
    entry = str(deck_dir)
    added = entry not in sys.path
    if added:
        sys.path.insert(0, entry)
    before = set(sys.modules)
    try:
        yield
    finally:
        for name in set(sys.modules) - before:
            file = getattr(sys.modules[name], "__file__", None)
            if file and Path(file).resolve().parent == deck_dir.resolve():
                del sys.modules[name]
        if added:
            with contextlib.suppress(ValueError):
                sys.path.remove(entry)


@contextlib.contextmanager
def _loader_hook(calls: list) -> Iterator[None]:
    """Record each top-level ``get_parameters_from_file`` call the deck makes."""
    import zutil
    import zutil.fileutils as fileutils

    original = fileutils.get_parameters_from_file
    depth = [0]

    def loader(control_file_path, *args, **kwargs):
        namespace = kwargs.get("namespace", args[1] if len(args) > 1 else None)
        caller = sys._getframe(1).f_globals
        resolved = control_file_path
        if isinstance(resolved, str) and not resolved.endswith(".py"):
            resolved += ".py"
        resolved = Path(resolved)
        if not resolved.is_absolute():
            anchor = (namespace or {}).get("__file__") or caller.get("__file__")
            if anchor:
                resolved = Path(anchor).parent / resolved
        named = resolved.absolute()
        resolved = named.resolve()
        snapshot = dict(namespace) if namespace is not None else None
        depth[0] += 1
        try:
            result = original(resolved, *args, **kwargs)
        finally:
            depth[0] -= 1
        if depth[0] == 0:
            calls.append(
                _Call(
                    control_file_path,
                    named,
                    resolved,
                    snapshot,
                    namespace is not None and namespace is caller,
                    result,
                )
            )
        return result

    fileutils.get_parameters_from_file = loader
    exported = getattr(zutil, LOADER, None)
    zutil.get_parameters_from_file = loader
    try:
        yield
    finally:
        fileutils.get_parameters_from_file = original
        if exported is not None:
            zutil.get_parameters_from_file = exported


@contextlib.contextmanager
def _without_solver_modules() -> Iterator[None]:
    """Make the solver's own modules unimportable while a deck runs.

    A deck that loads one is refused the same way whether or not it happens to
    be importable where update_input runs.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in REFUSED_MODULES
    }
    for name in saved:
        del sys.modules[name]
    for name in REFUSED_MODULES:
        sys.modules[name] = None
    try:
        yield
    finally:
        for name in REFUSED_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def _run(
    path: Path, tree: ast.Module, namespace: Optional[dict]
) -> tuple[list[_Step], dict[str, Path]]:
    ns = dict(namespace) if namespace else {}
    ns["__file__"] = str(path)
    flags = _future_flags(tree)
    steps: list[_Step] = []
    module_files: dict[str, Path] = {}
    prev_params: Any = _MISSING
    prev_prints: dict = {}
    calls: list[_Call] = []
    with _deck_dir(path.parent), _loader_hook(calls), _without_solver_modules():
        for stmt in tree.body:
            calls.clear()
            code = compile(
                ast.Module(body=[stmt], type_ignores=[]),
                str(path),
                "exec",
                flags=flags,
                dont_inherit=True,
            )
            try:
                exec(code, ns)
            except Exception as e:
                raise DeckReadError(_run_error(path, stmt, e)) from e
            params = ns.get(PARAMETERS, _MISSING)
            flat = _flatten(params) if isinstance(params, dict) else {}
            prints = {k: _fingerprint(v) for k, v in flat.items()}
            changed = {
                k
                for k in prints.keys() | prev_prints.keys()
                if prints.get(k, _MISSING) != prev_prints.get(k, _MISSING)
            }
            steps.append(
                _Step(
                    stmt, params, flat, changed, params is not prev_params, list(calls)
                )
            )
            prev_params, prev_prints = params, prints
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                _record_modules(stmt, module_files)
    if not steps or not isinstance(steps[-1].params, dict):
        raise DeckReadError(_no_parameters(path, ns))
    return steps, module_files


def _run_error(path: Path, stmt: ast.stmt, e: Exception) -> str:
    lines = [
        f.lineno
        for f in traceback.extract_tb(e.__traceback__)
        if f.filename == str(path)
    ]
    line = lines[-1] if lines else stmt.lineno
    missing = getattr(e, "name", None) if isinstance(e, ImportError) else None
    if missing and missing.split(".")[0] in REFUSED_MODULES:
        return (
            f"{path.name} line {line} runs code that imports {missing}. "
            f"update_input runs the deck to read it, and {missing.split('.')[0]} is "
            "only available inside the solver or ParaView, so this deck cannot be "
            "converted."
        )
    text = f"{path.name} does not run: line {line}: {type(e).__name__}: {e}"
    if isinstance(e, NameError):
        text += (
            ". If another deck loads this one with get_parameters_from_file "
            "and supplies that name, convert that deck instead."
        )
    return text


def _no_parameters(path: Path, ns: dict) -> str:
    near = [
        k
        for k, v in ns.items()
        if isinstance(v, dict) and k != PARAMETERS and k.lower().startswith("param")
    ]
    group = ns.get("override")
    if isinstance(group, dict) and "mesh_case_pair" in group:
        return (
            f"{path.name} is an overset group (override > mesh_case_pair), not a "
            "single deck; update_input converts it with every deck it lists"
        )
    text = f"{path.name} sets no parameters dict"
    if near:
        text += f"; it sets {', '.join(near)}, which the solver does not read"
    return text


def _record_modules(stmt: ast.stmt, module_files: dict[str, Path]) -> None:
    if isinstance(stmt, ast.Import):
        names = [a.name for a in stmt.names]
    else:
        names = [stmt.module] if stmt.module and not stmt.level else []
    for name in names:
        file = getattr(sys.modules.get(name), "__file__", None)
        if file and file.endswith(".py") and not _installed(Path(file)):
            module_files[name] = Path(file).absolute()


def _installed(path: Path) -> bool:
    return any(part in ("site-packages", "dist-packages") for part in path.parts)


def _flatten(tree: dict, prefix: Keys = ()) -> dict:
    out: dict = {}
    for key, value in tree.items():
        keys = prefix + (key,)
        if isinstance(value, dict) and value:
            out.update(_flatten(value, keys))
        else:
            out[keys] = value
    return out


def _fingerprint(value: Any) -> str:
    try:
        return f"{type(value).__name__}:{value!r}"
    except Exception:
        return f"{type(value).__name__}@{id(value)}"


def _bound(node: ast.AST) -> set[str]:
    """Names a top-level statement binds in the module namespace."""
    names: set[str] = set()

    def visit(n: ast.AST) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
            return
        if isinstance(
            n, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            names.update(_ANY if a.name == "*" else a.asname or a.name for a in n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return names


def _stores(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    }


def _loads(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _free_names(node: ast.AST) -> frozenset:
    inner: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Lambda):
            args = n.args
            for a in args.posonlyargs + args.args + args.kwonlyargs:
                inner.add(a.arg)
            for a in (args.vararg, args.kwarg):
                if a is not None:
                    inner.add(a.arg)
        elif isinstance(n, ast.comprehension):
            inner |= {t.id for t in ast.walk(n.target) if isinstance(t, ast.Name)}
        elif isinstance(n, ast.NamedExpr):
            inner.add(n.target.id)
    return frozenset(_loads(node) - inner)


def _text_names(text: Optional[str]) -> frozenset:
    if text is None:
        return frozenset()
    try:
        return _free_names(ast.parse(f"(\n{text}\n)", mode="eval"))
    except SyntaxError:
        return frozenset({_ANY})


def _subscript_keys(node: ast.expr) -> Optional[Keys]:
    """The constant keys of ``parameters[k1][k2]...``, or None for any other expression."""
    keys: list = []
    while isinstance(node, ast.Subscript):
        if not isinstance(node.slice, ast.Constant):
            return None
        keys.append(node.slice.value)
        node = node.value
    if isinstance(node, ast.Name) and node.id == PARAMETERS:
        return tuple(reversed(keys))
    return None


def _is_loader_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name == LOADER


class _Reader:
    def __init__(self, path, text, tree, steps, module_files):
        self.path: Path = path
        self.text: str = text
        self.tree: ast.Module = tree
        self.steps: list[_Step] = steps
        self.module_files: dict[str, Path] = module_files
        self.line_starts = _line_starts(text)
        self.bound = [
            _bound(s.stmt) | ({_ANY} if any(c.shared for c in s.calls) else set())
            for s in steps
        ]
        self.src: dict[Keys, _Src] = {}
        self.inherited: set[Keys] = set()
        self.base: Optional[BaseDeck] = None
        self.base_leaves: dict[Keys, Leaf] = {}
        self.set_line: dict[Keys, int] = {}
        self.notes: list[Note] = []
        self.absorbed: list[int] = []
        self.rewrites: list[tuple[int, int, str]] = []
        self.style = "literal"
        self.spread_operand = ""

    def segment(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.text, node)

    def span(self, node: ast.AST) -> tuple[int, int]:
        return (
            _offset(self.text, self.line_starts, node.lineno, node.col_offset),
            _offset(self.text, self.line_starts, node.end_lineno, node.end_col_offset),
        )

    def read(self) -> tuple[V1Deck, DeckFile]:
        start = max(i for i, s in enumerate(self.steps) if s.rebound)
        self.start = start
        self.start_step(start)
        for index in range(start + 1, len(self.steps)):
            self.later_step(index)
        render_at = (
            self.absorbed[-1] if self.style == "overrides" and self.absorbed else start
        )
        leaves = self.leaves(render_at)
        evaluated = self.steps[-1].params
        try:
            v1 = V1Deck(leaves, evaluated, self.notes)
        except ValueError as e:
            raise DeckReadError(f"{self.path.name}: {e}") from e
        return v1, self.deck_file(start)

    def deck_file(self, start: int) -> DeckFile:
        start_span = self.span(self.steps[start].stmt)
        removed = [self.span(self.steps[i].stmt) for i in self.absorbed]
        insert_after = False
        if self.style == "overrides":
            if removed:
                target = removed.pop()
            else:
                target = (start_span[1], start_span[1])
                insert_after = True
        else:
            target = start_span
        inherited_text = {
            norm_path(*keys): self.base_leaves[keys].source
            for keys in self.inherited
            if keys in self.base_leaves
        }
        return DeckFile(
            path=self.path,
            text=self.text,
            style=self.style,
            target=target,
            removed=removed,
            rewrites=self.rewrites,
            base=self.base,
            spread_operand=self.spread_operand,
            inherited_text=inherited_text,
            insert_after=insert_after,
        )

    def start_step(self, index: int) -> None:
        step = self.steps[index]
        stmt = step.stmt
        for keys in step.flat:
            self.set_line[keys] = stmt.lineno
        calls = [c for c in step.calls if c.result is step.params]
        if calls:
            self.loader_base(index, calls[-1])
            return
        if isinstance(stmt, ast.ImportFrom) and any(
            a.name == PARAMETERS and (a.asname or a.name) == PARAMETERS
            for a in stmt.names
        ):
            if self.import_base(index, stmt.module or ""):
                self.style = "overrides"
                return
        value = self.assigned_value(stmt)
        if value is not None and self.assignment(index, value):
            return
        self.opaque(index)

    def assigned_value(self, stmt: ast.stmt) -> Optional[ast.expr]:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target = stmt.target
        else:
            return None
        if isinstance(target, ast.Name) and target.id == PARAMETERS:
            return stmt.value
        return None

    def assignment(self, index: int, value: ast.expr) -> bool:
        if isinstance(value, ast.Dict):
            if value.keys and value.keys[0] is None:
                module = self.base_module(value.values[0])
                if module and self.import_base(index, module):
                    self.style = "spread"
                    self.spread_operand = self.segment(value.values[0])
                    rest = ast.Dict(keys=value.keys[1:], values=value.values[1:])
                    self.walk_dict(rest, (), index)
                    return True
            self.walk_dict(value, (), index)
            self.mark_literal(value, ())
            return True
        if isinstance(value, ast.Call):
            func = self.segment(value.func)
            if func == "dict" and value.args:
                module = self.base_module(value.args[0])
                if module and len(value.args) == 1 and self.import_base(index, module):
                    self.style = "spread"
                    self.spread_operand = self.segment(value.args[0])
                    self.keyword_entries(value.keywords, index)
                    return True
                local = self.local_literal(value.args[0], index)
                if local is not None and len(value.args) == 1:
                    self.walk_dict(local[1], (), local[0])
                    self.keyword_entries(value.keywords, index)
                    self.local_base_note(value.args[0], local[0])
                    return True
                return False
            copied = self.copied(value)
            if copied is not None:
                return self.reference(index, copied)
            return False
        return self.reference(index, value)

    def copied(self, call: ast.Call) -> Optional[ast.expr]:
        func = self.segment(call.func)
        if (
            func in ("copy.deepcopy", "deepcopy", "copy.copy", "dict")
            and len(call.args) == 1
            and not call.keywords
        ):
            return call.args[0]
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "copy"
            and not call.args
        ):
            return call.func.value
        return None

    def reference(self, index: int, node: ast.expr) -> bool:
        module = self.base_module(node)
        if module and self.import_base(index, module):
            self.style = "overrides"
            return True
        local = self.local_literal(node, index)
        if local is not None:
            self.walk_dict(local[1], (), local[0])
            self.local_base_note(node, local[0])
            return True
        return False

    def local_base_note(self, node: ast.expr, origin: int) -> None:
        self.notes.append(
            Note(
                f"{self.path.name} line {self.steps[origin].stmt.lineno} defines "
                f"{self.segment(node)}, which the new parameters no longer use; "
                "its values were written into the new parameters."
            )
        )

    def keyword_entries(self, keywords: list[ast.keyword], index: int) -> None:
        for kw in keywords:
            if kw.arg is not None:
                self.assign((kw.arg,), kw.value, index)
            elif isinstance(kw.value, ast.Dict):
                self.walk_dict(kw.value, (), index)
            else:
                self.drop(())

    def base_module(self, node: ast.expr) -> Optional[str]:
        """The module whose ``parameters`` dict ``node`` names, if it is a deck file."""
        aliases: dict[str, str] = {}
        from_params: dict[str, str] = {}
        for stmt in self.tree.body:
            if isinstance(stmt, ast.Import):
                for a in stmt.names:
                    if "." not in a.name:
                        aliases[a.asname or a.name] = a.name
            elif isinstance(stmt, ast.ImportFrom) and stmt.module and not stmt.level:
                for a in stmt.names:
                    if a.name == PARAMETERS:
                        from_params[a.asname or a.name] = stmt.module
        module = None
        if (
            isinstance(node, ast.Attribute)
            and node.attr == PARAMETERS
            and isinstance(node.value, ast.Name)
        ):
            module = aliases.get(node.value.id)
        elif isinstance(node, ast.Name):
            module = from_params.get(node.id)
        if module in self.module_files:
            return module
        return None

    def import_base(self, index: int, module: str) -> bool:
        file = self.module_files.get(module)
        if file is None:
            return False
        v1, deck_file = read_deck(file)
        self.inherit(index, BaseDeck(file, "import", False, None, v1, deck_file))
        for stmt in self.tree.body:
            new = _rewrite_import(stmt, module)
            if new is not None:
                start, end = self.span(stmt)
                self.rewrites.append((start, end, new))
        return True

    def loader_base(self, index: int, call: _Call) -> None:
        stmt = self.steps[index].stmt
        loader_calls = [n for n in ast.walk(stmt) if _is_loader_call(n)]
        if len(loader_calls) != 1:
            raise DeckReadError(
                f"{self.path.name} line {stmt.lineno} has {len(loader_calls)} "
                f"{LOADER} calls; update_input converts a deck that loads one base."
            )
        self.rewrite_loader_argument(index, loader_calls[0], call)
        v1, deck_file = read_deck(call.named, namespace=call.namespace)
        self.inherit(
            index,
            BaseDeck(call.named, "loader", call.shared, call.namespace, v1, deck_file),
        )
        self.style = "overrides"

    def rewrite_loader_argument(self, index: int, node: ast.Call, call: _Call) -> None:
        arg = (
            node.args[0]
            if node.args
            else next(
                (k.value for k in node.keywords if k.arg == "control_file_path"), None
            )
        )
        if not isinstance(call.raw, str) or arg is None:
            raise DeckReadError(
                f"{self.path.name} line {node.lineno} names its base deck with "
                "something other than a file name, so update_input cannot point "
                "it at the converted base."
            )
        new_name = v2_name(call.raw)
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            self.rewrite_string(arg, call.raw, new_name)
            return
        if isinstance(arg, ast.Name):
            definition = self.constant_definition(arg.id, index)
            if definition is not None and self.loaded_once(arg.id):
                self.rewrite_string(definition, call.raw, new_name)
                return
        start, end = self.span(arg)
        self.rewrites.append((start, end, _str_text(new_name)))

    def rewrite_string(self, node: ast.Constant, old: str, new: str) -> None:
        start, end = self.span(node)
        segment = self.text[start:end]
        if old in segment and node.value == old:
            self.rewrites.append((start, end, segment.replace(old, new, 1)))
        else:
            self.rewrites.append((start, end, _str_text(new)))

    def constant_definition(self, name: str, use: int) -> Optional[ast.Constant]:
        for i in range(use - 1, -1, -1):
            if name in self.bound[i] or _ANY in self.bound[i]:
                stmt = self.steps[i].stmt
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    return stmt.value
                return None
        return None

    def loaded_once(self, name: str) -> bool:
        count = sum(
            1
            for n in ast.walk(self.tree)
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
        )
        return count == 1

    def inherit(self, index: int, base: BaseDeck) -> None:
        self.base = base
        line = self.steps[index].stmt.lineno
        for leaf in base.v1:
            self.base_leaves[leaf.keys] = leaf
            self.inherited.add(leaf.keys)
            if leaf.source is not None:
                names = _text_names(leaf.source)
                self.src[leaf.keys] = _Src(leaf.source, line, index, names, True)

    def walk_dict(self, node: ast.Dict, prefix: Keys, origin: int) -> None:
        for k, v in zip(node.keys, node.values):
            if k is None:
                local = self.local_literal(v, origin)
                if local is not None:
                    self.walk_dict(local[1], prefix, local[0])
                else:
                    self.drop(prefix)
                continue
            if not isinstance(k, ast.Constant):
                self.drop(prefix)
                continue
            self.assign(prefix + (k.value,), v, origin)

    def assign(self, keys: Keys, node: ast.expr, origin: int) -> None:
        self.block_override(keys, node, origin)
        self.drop(keys)
        if isinstance(node, ast.Dict):
            if not node.keys:
                self.src[keys] = _Src("{}", node.lineno, origin, frozenset())
            self.walk_dict(node, keys, origin)
            return
        local = self.local_literal(node, origin)
        if local is not None:
            self.walk_dict(local[1], keys, local[0])
            return
        self.src[keys] = _Src(
            self.segment(node), node.lineno, origin, _free_names(node)
        )

    def block_override(self, keys: Keys, node: ast.expr, origin: int) -> None:
        if self.base is None or not keys:
            return
        n = len(keys)
        if any(len(p) > n and p[:n] == keys for p in self.inherited):
            line = node.lineno
            self.notes.append(
                BlockOverride(
                    f"{self.path.name} line {line} replaces the whole "
                    f"'{' > '.join(map(str, keys))}' block of {self.base.path.name}; "
                    "the base's other keys in it are not used.",
                    (norm_path(*keys),),
                    self.base.path.name,
                )
            )

    def drop(self, prefix: Keys) -> None:
        n = len(prefix)
        for keys in [k for k in self.src if k[:n] == prefix]:
            del self.src[keys]
        self.inherited = {k for k in self.inherited if k[:n] != prefix}

    def mark_literal(self, node: ast.Dict, prefix: Keys) -> None:
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant):
                keys = prefix + (k.value,)
                self.set_line[keys] = v.lineno
                if isinstance(v, ast.Dict):
                    self.mark_literal(v, keys)

    def local_literal(self, node: ast.expr, use: int) -> Optional[tuple[int, ast.Dict]]:
        """The top-level dict literal a name holds at statement ``use``, if unchanged since."""
        if not isinstance(node, ast.Name):
            return None
        name = node.id
        for i in range(use - 1, -1, -1):
            if name in self.bound[i] or _ANY in self.bound[i]:
                stmt = self.steps[i].stmt
                if not (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Dict)
                ):
                    return None
                if any(name in _loads(self.steps[j].stmt) for j in range(i + 1, use)):
                    return None
                return i, stmt.value
        return None

    def later_step(self, index: int) -> None:
        step = self.steps[index]
        stmt = step.stmt
        ops = _edit_ops(stmt)
        if ops is not None and not step.calls:
            prefixes = [keys for _, keys, _ in ops]
            if all(any(p[: len(k)] == k for k in prefixes) for p in step.changed):
                self.absorbed.append(index)
                for kind, keys, value in ops:
                    if kind == "set":
                        self.assign(keys, value, index)
                    else:
                        self.drop(keys)
                        self.notes.append(
                            Note(
                                f"{self.path.name} line {stmt.lineno} deletes "
                                f"'{' > '.join(map(str, keys))}', so it is absent here.",
                                (norm_path(*keys),),
                            )
                        )
                for keys in step.changed:
                    self.set_line[keys] = stmt.lineno
                return
        if step.changed:
            self.opaque(index)
            return
        if PARAMETERS in _loads(stmt) and not _is_main_block(stmt):
            self.notes.append(
                Note(
                    f"{self.path.name} line {stmt.lineno} reads parameters and is "
                    "copied as written; check that it reads the config version 2 keys."
                )
            )

    def opaque(self, index: int) -> None:
        step = self.steps[index]
        stmt = step.stmt
        others = (self.bound[index] - {PARAMETERS}) & (
            self.loads_elsewhere(index) | {_ANY}
        )
        if others:
            raise DeckReadError(
                f"{self.path.name} line {stmt.lineno} changes parameters and also sets "
                f"{', '.join(sorted(others))}, which the deck uses elsewhere; "
                "update_input cannot separate the two. Move the change to "
                "parameters into a statement of its own."
            )
        changed = step.changed if index != self.start else set(step.flat)
        for keys in changed:
            self.drop(keys)
            self.set_line[keys] = stmt.lineno
        if index != self.start:
            self.absorbed.append(index)

    def loads_elsewhere(self, index: int) -> set[str]:
        names: set[str] = set()
        for i, step in enumerate(self.steps):
            if i != index:
                names |= _loads(step.stmt) - _stores(step.stmt)
        return names

    def leaves(self, render_at: int) -> list[Leaf]:
        final = self.steps[-1].flat
        leaves = []
        literal_lines: dict[int, list[str]] = {}
        dict_sources: dict[Keys, _Src] = {}
        for keys, value in final.items():
            src = self.src.get(keys)
            if src is not None:
                reason = self.unusable(src, render_at)
                if reason is None:
                    leaves.append(Leaf(keys, value, src.text, src.line))
                    continue
                self.notes.append(
                    Note(
                        f"'{' > '.join(map(str, keys))}' is written as a literal value: {reason}.",
                        (norm_path(*keys),),
                    )
                )
                leaves.append(Leaf(keys, value, None, src.line))
                continue
            holder = next(
                (keys[:n] for n in range(len(keys) - 1, 0, -1) if keys[:n] in self.src),
                None,
            )
            if holder is not None:
                dict_sources.setdefault(holder, self.src[holder])
                leaves.append(Leaf(keys, value, None, self.src[holder].line))
                continue
            line = self.set_line.get(keys, self.steps[self.start].stmt.lineno)
            if keys in self.inherited and self.base is not None:
                leaves.append(Leaf(keys, value, None, line))
                continue
            literal_lines.setdefault(line, [])
            if str(keys[0]) not in literal_lines[line]:
                literal_lines[line].append(str(keys[0]))
            leaves.append(Leaf(keys, value, None, line))
        for keys, src in dict_sources.items():
            self.notes.append(
                Note(
                    f"{self.path.name} line {src.line}: '{' > '.join(map(str, keys))}' "
                    f"is {src.text}, a dict built in code, so its keys are written as "
                    "literal values.",
                    (norm_path(*keys),),
                )
            )
        for line, tops in sorted(literal_lines.items()):
            self.notes.append(
                Note(
                    f"{self.path.name} line {line} builds {', '.join(repr(t) for t in tops)} "
                    "in code, so update_input writes them as literal values."
                )
            )
        return leaves

    def unusable(self, src: _Src, render_at: int) -> Optional[str]:
        if (
            src.inherited
            and self.base is not None
            and not self.base.shared
            and src.names
        ):
            return (
                f"its text in {self.base.path.name} uses names of that deck, "
                f"which {self.path.name} does not share"
            )
        if not src.names:
            return None
        low, high = sorted((src.origin, render_at))
        changed: set[str] = set()
        for i in range(low + 1, high):
            if i not in self.absorbed:
                changed |= self.bound[i]
        if _ANY in changed:
            changed = set(src.names)
        hit = sorted(src.names & changed)
        if hit:
            lines = (self.steps[low].stmt.lineno, self.steps[high].stmt.lineno)
            return (
                f"{', '.join(hit)} changes between line {lines[0]} and line {lines[1]}"
            )
        return None


def _is_main_block(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.If)
        and isinstance(stmt.test, ast.Compare)
        and isinstance(stmt.test.left, ast.Name)
        and stmt.test.left.id == "__name__"
    )


def _edit_ops(stmt: ast.stmt) -> Optional[list[tuple[str, Keys, Optional[ast.expr]]]]:
    """A foldable change to ``parameters`` with constant keys, as set and delete operations."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        keys = _subscript_keys(stmt.targets[0])
        if keys:
            return [("set", keys, stmt.value)]
        return None
    if isinstance(stmt, ast.Delete):
        ops = []
        for target in stmt.targets:
            keys = _subscript_keys(target)
            if not keys:
                return None
            ops.append(("del", keys, None))
        return ops
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return None
    call = stmt.value
    if not isinstance(call.func, ast.Attribute):
        return None
    keys = _subscript_keys(call.func.value)
    if keys is None:
        return None
    if call.func.attr == "update":
        ops = []
        if len(call.args) > 1:
            return None
        if call.args:
            arg = call.args[0]
            if not isinstance(arg, ast.Dict):
                return None
            for k, v in zip(arg.keys, arg.values):
                if not isinstance(k, ast.Constant):
                    return None
                ops.append(("set", keys + (k.value,), v))
        for kw in call.keywords:
            if kw.arg is None:
                return None
            ops.append(("set", keys + (kw.arg,), kw.value))
        return ops
    if call.func.attr == "pop" and call.args and isinstance(call.args[0], ast.Constant):
        return [("del", keys + (call.args[0].value,), None)]
    return None


def _rewrite_import(stmt: ast.stmt, module: str) -> Optional[str]:
    new = v2_name(module)
    if isinstance(stmt, ast.Import) and any(a.name == module for a in stmt.names):
        parts = []
        for a in stmt.names:
            if a.name == module:
                parts.append(f"{new} as {a.asname or module}")
            else:
                parts.append(a.name + (f" as {a.asname}" if a.asname else ""))
        return "import " + ", ".join(parts)
    if isinstance(stmt, ast.ImportFrom) and stmt.module == module and not stmt.level:
        names = ", ".join(
            a.name + (f" as {a.asname}" if a.asname else "") for a in stmt.names
        )
        return f"from {new} import {names}"
    return None


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for m in re.finditer(r"\r\n|\r|\n", text):
        starts.append(m.end())
    return starts


def _offset(text: str, starts: list[int], lineno: int, col: int) -> int:
    start = starts[lineno - 1]
    end = starts[lineno] if lineno < len(starts) else len(text)
    line = text[start:end]
    return start + len(line.encode("utf-8")[:col].decode("utf-8"))


def _nest(v2: dict[V2Path, V2Value]) -> dict:
    root: dict = {}
    for path, value in v2.items():
        node = root
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return root


def _str_text(text: str) -> str:
    if '"' not in text and "\\" not in text and text.isprintable():
        return f'"{text}"'
    return repr(text)


def _literal_text(value: Any, where: str) -> str:
    """Python source for a plain value: numbers, strings, lists, tuples and dicts of them."""
    if value is None or isinstance(value, bool):
        return repr(value)
    if isinstance(value, str):
        return _str_text(value)
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return f'float("{value}")'
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_literal_text(v, where) for v in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(_literal_text(v, where) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{_literal_text(k, where)}: {_literal_text(v, where)}"
                for k, v in value.items()
            )
            + "}"
        )
    if hasattr(value, "tolist") and type(value).__module__.startswith("numpy"):
        return _literal_text(value.tolist(), where)
    if callable(value):
        raise RenderError(
            f"{where} is a callable with no source text in the deck, so it cannot be written."
        )
    raise RenderError(
        f"{where} holds a {type(value).__name__}, which cannot be written as a literal."
    )


def _value_text(value: V2Value, path: V2Path) -> str:
    where = " > ".join(map(str, path))
    if isinstance(value, Source):
        if value.leaf.source is not None:
            return value.leaf.source
        return _literal_text(value.leaf.value, where)
    if isinstance(value, Value):
        try:
            return _literal_text(value.value, where)
        except RenderError:
            return repr(value.value)
    if isinstance(value, Expr):
        return value.text
    if isinstance(value, Placeholder):
        return _str_text(value.text)
    raise RenderError(f"{where} holds {value!r}, which is not a v2 value")


def _signature(value: V2Value, inherited: dict[V1Path, Optional[str]]) -> tuple:
    if isinstance(value, Source):
        leaf = value.leaf
        text = inherited[leaf.path] if leaf.path in inherited else leaf.source
        return ("value", text, leaf.value)
    if isinstance(value, Value):
        return ("value", None, value.value)
    if isinstance(value, Expr):
        return ("value", value.text, value.value)
    if isinstance(value, Placeholder):
        return ("decide", without_item(value.text), None)
    return ("other", repr(value), None)


def without_item(text: str) -> str:
    """``text`` with its item numbers removed, so one question about two items matches."""
    return re.sub(r"item \d+", "item", text)


def same_value(a: Any, b: Any) -> bool:
    """Equality for evaluated deck values: functions by code, arrays by element."""
    if a is b:
        return True
    if hasattr(a, "__code__") and hasattr(b, "__code__"):
        return a.__code__ == b.__code__
    if callable(a) or callable(b):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(same_value, a, b))
    if hasattr(a, "shape") or hasattr(b, "shape"):
        try:
            import numpy as np

            return bool(np.array_equal(a, b))
        except Exception:
            return False
    try:
        return bool(a == b)
    except Exception:
        return False


class _Writer:
    """Writes v2 trees as source, on one line for ruff or in fixed indent."""

    def __init__(self, pretty: bool):
        self.pretty = pretty

    def node(self, node: Any, path: V2Path, level: int, lead: int = 0) -> str:
        """``lead`` is the column the node starts at, for the line-length check."""
        if isinstance(node, dict):
            return self.entries(
                [
                    (f"{_literal_text(k, 'key')}: ", v, path + (k,))
                    for k, v in node.items()
                ],
                level,
                lead,
            )
        return _value_text(node, path)

    def entries(
        self, items: list[tuple[str, Any, V2Path]], level: int, lead: int = 0
    ) -> str:
        """``items`` are (prefix, node, path); a node of None writes the prefix alone.

        In fixed indent a dict stays on one line when that line fits, as ruff
        lays it out.
        """
        if not items:
            return "{}"
        flat = (
            "{"
            + ", ".join(
                prefix if node is None else prefix + _Writer(False).node(node, path, 0)
                for prefix, node, path in items
            )
            + "}"
        )
        if not self.pretty or ("\n" not in flat and lead + len(flat) + 1 <= _LINE):
            return flat
        pad = _INDENT * (level + 1)
        parts = [
            prefix
            if node is None
            else prefix + self.node(node, path, level + 1, len(pad) + len(prefix))
            for prefix, node, path in items
        ]
        return "{\n" + "".join(f"{pad}{p},\n" for p in parts) + _INDENT * level + "}"

    def spread(
        self, operand: str, base: dict, var: dict, path: V2Path, level: int, inherited
    ) -> str:
        if any(k not in var for k in base):
            return self.node(var, path, level)
        items: list[tuple[str, Any, V2Path]] = [(f"**{operand}", None, path)]
        for key, value in var.items():
            key_text = _literal_text(key, "key")
            sub = path + (key,)
            if key not in base:
                items.append((f"{key_text}: ", value, sub))
            elif isinstance(value, dict) and isinstance(base[key], dict):
                if _differs(base[key], value, inherited):
                    text = self.spread(
                        f"{operand}[{key_text}]",
                        base[key],
                        value,
                        sub,
                        level + 1,
                        inherited,
                    )
                    items.append((f"{key_text}: {text}", None, sub))
            elif _differs_leaf(base[key], value, inherited):
                items.append((f"{key_text}: ", value, sub))
        return self.entries(items, level)


def _differs_leaf(base: Any, var: Any, inherited) -> bool:
    if isinstance(base, dict) or isinstance(var, dict):
        return not (isinstance(base, dict) and isinstance(var, dict)) or _differs(
            base, var, inherited
        )
    sb, sv = _signature(base, {}), _signature(var, inherited)
    return not (sb[0] == sv[0] and sb[1] == sv[1] and same_value(sb[2], sv[2]))


def _differs(base: dict, var: dict, inherited) -> bool:
    if base.keys() != var.keys():
        return True
    return any(_differs_leaf(base[k], var[k], inherited) for k in var)


def _overrides(
    base: dict, var: dict, inherited, path: V2Path = ()
) -> list[tuple[str, V2Path, Any]]:
    out: list[tuple[str, V2Path, Any]] = []
    removed = [k for k in base if k not in var]
    if removed and path:
        return [("set", path, var)]
    out.extend(("del", path + (k,), None) for k in removed)
    for key, value in var.items():
        sub = path + (key,)
        if key not in base:
            out.append(("set", sub, value))
        elif isinstance(value, dict) and isinstance(base[key], dict):
            out.extend(_overrides(base[key], value, inherited, sub))
        elif _differs_leaf(base[key], value, inherited):
            out.append(("set", sub, value))
    return out


def _ruff_command() -> Optional[list[str]]:
    exe = shutil.which("ruff")
    if exe:
        return [exe]
    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]
    return None


def _ruff_format(code: str) -> Optional[str]:
    command = _ruff_command()
    if command is None:
        return None
    try:
        done = subprocess.run(
            command
            + ["format", "--isolated", "--stdin-filename", "parameters.py", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.rstrip("\n")


def _statement(
    deck: DeckFile, tree: dict, base_tree: Optional[dict], pretty: bool
) -> str:
    writer = _Writer(pretty)
    if deck.style == "literal":
        return f"{PARAMETERS} = {writer.node(tree, (), 0, len(PARAMETERS) + 3)}"
    if deck.style == "spread":
        body = writer.spread(
            deck.spread_operand, base_tree, tree, (), 0, deck.inherited_text
        )
        return f"{PARAMETERS} = {body}"
    lines = []
    for kind, path, value in _overrides(base_tree, tree, deck.inherited_text):
        target = PARAMETERS + "".join(f"[{_literal_text(k, 'key')}]" for k in path)
        if kind == "del":
            lines.append(f"del {target}")
        else:
            lines.append(f"{target} = {writer.node(value, path, 0, len(target) + 3)}")
    return "\n".join(lines)


def render_deck(
    deck: DeckFile,
    v2: dict[V2Path, V2Value],
    base_v2: Optional[dict[V2Path, V2Value]] = None,
) -> str:
    """The whole text of the converted deck.

    Everything outside the ``parameters`` statements is copied as written. For a
    variant, ``base_v2`` is the v2 tree written to the base's ``_v2`` file, and
    only the difference from it is written.

    Raises:
        RenderError: a variant is rendered without its base's tree, or a value
            cannot be written as source.
    """
    tree = _nest(v2)
    base_tree = None
    if deck.base is not None:
        if base_v2 is None:
            raise RenderError(
                f"{deck.path.name} takes its parameters from {deck.base.path.name}; "
                f"convert {deck.base.path.name} first and pass its v2 tree."
            )
        base_tree = _nest(base_v2)
    statement = _statement(deck, tree, base_tree, pretty=False)
    if statement:
        formatted = _ruff_format(statement)
        statement = (
            formatted
            if formatted is not None
            else _statement(deck, tree, base_tree, pretty=True)
        )
    edits = list(deck.rewrites)
    edits.extend((start, end, "") for start, end in deck.removed)
    start, end = deck.target
    if deck.insert_after:
        if statement:
            edits.append((start, end, "\n" + statement))
    else:
        edits.append((start, end, statement))
    return _splice(deck.text, edits)


def _splice(text: str, edits: list[tuple[int, int, str]]) -> str:
    starts = _line_starts(text)
    kept = [e for e in edits if e[2] != ""]
    removals = sorted(
        _whole_lines(text, starts, s, e) for s, e, new in edits if new == ""
    )
    merged: list[list[int]] = []
    for start, end in removals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for span in merged:
        span[1] = _past_blank_line(text, starts, span[0], span[1])
    widened = kept + [(start, end, "") for start, end in merged]
    for start, end, new in sorted(widened, key=lambda e: (e[0], e[1]), reverse=True):
        text = text[:start] + new + text[end:]
    return text


def _past_blank_line(text: str, starts: list[int], start: int, end: int) -> int:
    """Extend a removal of whole lines over one blank line when blank lines enclose it."""
    if start == 0 or start not in starts or end not in starts:
        return end
    previous = max(s for s in starts if s < start)
    following = [s for s in starts if s > end]
    next_end = following[0] if following else len(text)
    if (
        text[previous:start].strip() == ""
        and end < len(text)
        and text[end:next_end].strip() == ""
    ):
        return next_end
    return end


def _whole_lines(text: str, starts: list[int], start: int, end: int) -> tuple[int, int]:
    line_start = max(s for s in starts if s <= start)
    following = [s for s in starts if s > end]
    line_end = following[0] if following else len(text)
    before = text[line_start:start]
    after = text[end:line_end]
    if before.strip() == "" and (after.strip() == "" or after.strip().startswith("#")):
        return line_start, line_end
    return start, end


__all__ = [
    "BaseDeck",
    "BlockOverride",
    "DeckFile",
    "DeckReadError",
    "RenderError",
    "read_deck",
    "read_related",
    "related_decks",
    "render_deck",
    "same_value",
    "v2_name",
    "without_item",
]
