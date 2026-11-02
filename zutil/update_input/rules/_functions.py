"""Move a v1 deck function to v2, rewriting its ``def`` line where v2 needs it.

v2 calls every deck function with keyword arguments only. A function that
already takes those keywords, by the contract's own
:meth:`~zutil.validate.schemas.callables.CallableContract.signature_error`,
moves unchanged. A function v1 called positionally is rewritten where its
deck defines it when its arguments map onto the v2 keywords mechanically: the
``def`` line becomes ``def name(*, k1, k2):`` and the body reads the
keywords, every other byte kept. The mechanical forms are:

- the positional arguments v1 passed, in order, renamed to the v2 keywords;
- the three coordinates v1 passed as ``(x, y, z)``, unpacked from the one v2
  vector keyword by a line ``x, y, z = location``.

Any other function is kept as written and becomes a decision that names the
``def`` line to write and what v2 reads back.
"""

from __future__ import annotations

import ast
import inspect
import io
import tokenize
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Callable, Optional, Union

from zutil.validate.schemas.callables import (
    CFL_RAMP,
    FORCE_TRANSFORM,
    INITIAL_CONDITIONS,
    LEAF_AREA_DENSITY,
    MESH_DEFORMATION,
    MOMENTUM_SOURCE,
    PLACEMENT,
    CallableContract,
)

from ..model import Conversion, FunctionRewrite, Leaf, V2Path
from ..source import _line_starts, _offset


@dataclass(frozen=True)
class _V1Call:
    """How v1 called one kind of deck function.

    Attributes:
        call: The v1 call, as the engineer reads it in a decision.
        by_keyword: v1 passed the v2 keywords already, by name.
        positional: The v2 keywords that v1's positional arguments become,
            in v1's order.
        unpacked: The v2 keyword holding the three coordinates v1 passed as
            three positional arguments.
    """

    call: str
    by_keyword: bool = False
    positional: tuple[str, ...] = ()
    unpacked: str = ""
    alternative: str = ""


_V1_CALLS: tuple[tuple[CallableContract, _V1Call], ...] = (
    (
        INITIAL_CONDITIONS,
        _V1Call(
            call="func(**kwargs) with pressure, temperature, velocity, "
            "wall_distance and location",
            by_keyword=True,
        ),
    ),
    (
        MESH_DEFORMATION,
        _V1Call(call="func(x, y, z)", unpacked="location"),
    ),
    (
        FORCE_TRANSFORM,
        _V1Call(call="transform(x, y, z)", unpacked="vector"),
    ),
    (
        MOMENTUM_SOURCE,
        _V1Call(
            call="func(cell_centres) with one (x, y, z) tuple per cell",
            positional=("locations",),
        ),
    ),
    (
        LEAF_AREA_DENSITY,
        _V1Call(
            call="func(cells) with one (x, y, z, wall distance) tuple per cell",
        ),
    ),
    (
        PLACEMENT,
        _V1Call(call='func(**{"coord": (x, y, z)}) and read back {"coord": (x, y, z)}'),
    ),
    (
        CFL_RAMP,
        _V1Call(
            call="ramp_func(solve_cycle, real_time_cycle, cfl) and let it set the "
            "transport and coarse CFL on the cfl object",
            alternative=" A ramp list can replace the function: "
            '[{"type": "exponential", "factor": <growth>, "initial": <initial>}].',
        ),
    ),
)


def _v1_call(contract: CallableContract) -> _V1Call:
    for known, call in _V1_CALLS:
        if known is contract:
            return call
    raise KeyError(f"no v1 calling convention for {contract!r}")


def port_function(
    c: Conversion,
    leaf: Leaf,
    path: V2Path,
    contract: CallableContract,
    respelled: bool = False,
) -> None:
    """Move the function in ``leaf`` to ``path`` with a ``def`` line v2 can call.

    A value that is not callable moves as it is. Every function that moves
    gets a report line with its ``def`` line, rewritten or not, and what v2
    reads back. A function with no mechanical rewrite stays in the draft as
    written, and ``path`` carries the DECIDE placeholder naming the ``def``
    line v2 calls and what it reads back.
    """
    func = leaf.value
    if not callable(func):
        c.move(leaf, path, respelled=respelled)
        return
    call = _v1_call(contract)
    name = getattr(func, "__name__", type(func).__name__)
    if contract.signature_error(func) is None and (
        call.by_keyword or not _collects_positional(func)
    ):
        c.move(leaf, path, respelled=respelled)
        c.note(
            f"{name}: `{_def_line(func, name)}` already takes the keywords v2 "
            f"calls it with{_returns(contract, lambda: func(**contract.inputs))}",
            leaf.path,
        )
        return
    found = _definition(c, func)
    plan = _plan(found, contract, call) if isinstance(found, _Definition) else found
    if isinstance(plan, FunctionRewrite):
        existing = c.rewrite_at(plan.file, plan.line)
        if existing is None or existing == plan:
            c.rewrite(plan)
            c.move(
                leaf,
                path,
                respelled=respelled,
                reason="def line rewritten to the v2 call",
            )
            c.note(
                f"{name}: `{plan.old_def}` is now `{plan.new_def}`"
                f"{_returns(contract, lambda: func(*_v1_arguments(contract, call)))}",
                leaf.path,
            )
            return
        plan = f"is already rewritten as `{existing.new_def}` for another key"
    c.decide(
        [leaf],
        path,
        f"rewrite {name} as {contract.definition(name)}",
        details=(
            f"v1 called it as {call.call}. {name} {plan}, so update_input "
            f"did not rewrite it. v2 calls it with keyword arguments only, as "
            f"{contract.signature_text(name)}, and reads back {contract.reads}."
            f"{call.alternative}"
        ),
    )


def _v1_arguments(contract: CallableContract, call: _V1Call) -> list:
    if call.unpacked:
        return list(contract.inputs[call.unpacked])
    return [contract.inputs[k] for k in call.positional]


def _returns(contract: CallableContract, sample: Callable[[], object]) -> str:
    """What the function's body is seen to return, or nothing when it cannot be called."""
    try:
        result = sample()
    except Exception:
        return ""
    if contract.check_result(result) is not None:
        return ""
    if isinstance(result, Mapping) and set(result) == {"v1", "v2", "v3"}:
        return "; its body still returns the v1/v2/v3 dict, which v2 accepts"
    return f", and it returns {contract.reads}"


def _collects_positional(func: object) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind is p.VAR_POSITIONAL for p in parameters)


def _def_line(func: object, name: str) -> str:
    try:
        return f"def {name}{inspect.signature(func)}:"
    except (TypeError, ValueError):
        return f"def {name}(...):"


@dataclass
class _Definition:
    file: Path
    text: str
    node: ast.FunctionDef
    starts: list[int]

    def offset(self, lineno: int, col: int) -> int:
        return _offset(self.text, self.starts, lineno, col)

    def line_start(self, lineno: int) -> int:
        return self.starts[lineno - 1] if lineno <= len(self.starts) else len(self.text)


def top_level_def(
    func: object,
) -> Union[tuple[Path, str, ast.Module, ast.FunctionDef], str]:
    """The file, source, module and ``def`` node of a deck function, or why there are none."""
    if not inspect.isfunction(func):
        return "is not a plain function"
    if func.__name__ == "<lambda>":
        return "is a lambda"
    if func.__qualname__ != func.__name__:
        return "is not defined at the top level of its deck"
    file = Path(func.__code__.co_filename).resolve()
    try:
        text = file.read_bytes().decode("utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return "has no source that update_input can read"
    node = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == func.__name__
            and n.lineno == func.__code__.co_firstlineno
        ),
        None,
    )
    if node is None:
        return "is not a plain function at the top level of its deck"
    if node.decorator_list:
        return "is decorated"
    return file, text, tree, node


def _definition(c: Conversion, func: object) -> Union[_Definition, str]:
    found = top_level_def(func)
    if isinstance(found, str):
        return found
    file, text, tree, node = found
    if file not in {Path(p).resolve() for p in c.deck_files}:
        return f"is defined in {file.name}, outside the decks update_input converts"
    if _used_by_other_code(c, file, tree, node):
        return "is called by other code in its deck"
    return _Definition(file, text, node, _line_starts(text))


def _used_by_other_code(
    c: Conversion, file: Path, tree: ast.Module, node: ast.FunctionDef
) -> bool:
    trees = {file: tree}
    for other in {Path(p).resolve() for p in c.deck_files} - {file}:
        try:
            trees[other] = ast.parse(other.read_bytes().decode("utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
    name = node.name
    for path, module in trees.items():
        if path != file and not _imports(module, name):
            continue
        if any(_calls(n, name) for n in ast.walk(module)):
            return True
        for scope in ast.walk(module):
            if scope is node or not isinstance(scope, _SCOPES):
                continue
            if any(
                isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
                for n in ast.walk(scope)
            ):
                return True
    return False


def _imports(module: ast.Module, name: str) -> bool:
    return any(
        isinstance(n, (ast.Import, ast.ImportFrom))
        and any(
            (a.asname or a.name) == name or a.name == "*" or a.name == name
            for a in n.names
        )
        for n in ast.walk(module)
    ) or any(isinstance(n, ast.Attribute) and n.attr == name for n in ast.walk(module))


def _calls(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == name) or (
        isinstance(func, ast.Attribute) and func.attr == name
    )


_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.GeneratorExp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
)


class _Scan:
    """Every use of every name in a function body, by kind."""

    def __init__(self, node: ast.FunctionDef) -> None:
        self.names: dict[str, list[ast.Name]] = {}
        self.bindings: dict[str, int] = {}
        self.declared: set[str] = set()
        self.own: set[str] = set()
        self.in_fstring: set[int] = set()
        for stmt in node.body:
            self._visit(stmt, own=True, fstring=False)

    def _bind(self, name: Optional[str], own: bool) -> None:
        if name is None:
            return
        self.bindings[name] = self.bindings.get(name, 0) + 1
        if own:
            self.own.add(name)

    def _visit(self, node: ast.AST, own: bool, fstring: bool) -> None:
        if isinstance(node, ast.Name):
            self.names.setdefault(node.id, []).append(node)
            if fstring:
                self.in_fstring.add(id(node))
            if own and not isinstance(node.ctx, ast.Load):
                self.own.add(node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            self.declared.update(node.names)
            for name in node.names:
                self._bind(name, own=False)
        elif isinstance(node, ast.arg):
            self._bind(node.arg, own=False)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._bind(node.name, own)
        elif isinstance(node, ast.alias):
            self._bind((node.asname or node.name).split(".")[0], own)
        elif isinstance(node, ast.ExceptHandler):
            self._bind(node.name, own)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            self._bind(node.name, own)
        elif isinstance(node, ast.MatchMapping):
            self._bind(node.rest, own)
        inner = own and not isinstance(node, _SCOPES)
        fstring = fstring or isinstance(node, ast.JoinedStr)
        for child in ast.iter_child_nodes(node):
            self._visit(child, inner, fstring)

    def loads(self, name: str) -> list[ast.Name]:
        return [n for n in self.names.get(name, []) if isinstance(n.ctx, ast.Load)]

    def appears(self, name: str) -> bool:
        return name in self.names or name in self.bindings


Edit = tuple[int, int, str]


def _takes(count: int) -> str:
    return f"{count} argument" if count == 1 else f"{count} arguments"


def _plan(
    d: _Definition, contract: CallableContract, call: _V1Call
) -> Union[FunctionRewrite, str]:
    node = d.node
    args = node.args
    scan = _Scan(node)
    inputs = list(contract.inputs)
    edits: list[Edit] = []
    if call.by_keyword:
        reason = "cannot take the keywords v1 passed, which v2 passes too"
    elif call.positional:
        reason = _positional_edits(d, scan, call, edits)
    elif call.unpacked:
        reason = _unpacked_edits(d, scan, call, edits)
    else:
        reason = "takes the v1 arguments, which have no keyword-for-keyword match in v2"
    if reason is not None:
        return reason
    start = d.offset(node.lineno, node.col_offset)
    end = d.offset(node.end_lineno, node.end_col_offset)
    head = _head_end(d)
    if head is None:
        return "has a def line update_input cannot read"
    returns = (
        f" -> {ast.get_source_segment(d.text, node.returns)}"
        if node.returns is not None
        else ""
    )
    new_def = f"def {node.name}(*, {', '.join(inputs)}){returns}:"
    old_def = f"def {node.name}({ast.unparse(args)}){returns}:"
    edits.append((start, head, new_def))
    kept = sorted(edits, key=lambda e: (e[0], e[1]))
    for (_, first_end, _), (second_start, _, _) in zip(kept, kept[1:]):
        if second_start < first_end:
            return "cannot be rewritten without overlapping edits"
    pieces = []
    at = start
    for s, t, new in kept:
        pieces.append(d.text[at:s])
        pieces.append(new)
        at = t
    pieces.append(d.text[at:end])
    text = "".join(pieces)
    if not _rewritten_signature(text, node.name, inputs):
        return "cannot be rewritten to a function that parses"
    return FunctionRewrite(
        file=d.file,
        line=node.lineno,
        name=node.name,
        start=start,
        end=end,
        text=text,
        old_def=old_def,
        new_def=new_def,
    )


def _rewritten_signature(text: str, name: str, inputs: list[str]) -> bool:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    a = tree.body[0].args
    return (
        tree.body[0].name == name
        and not (a.posonlyargs or a.args or a.vararg or a.kwarg)
        and [k.arg for k in a.kwonlyargs] == inputs
    )


def _head_end(d: _Definition) -> Optional[int]:
    node = d.node
    start = d.offset(node.lineno, node.col_offset)
    body = d.offset(node.body[0].lineno, node.body[0].col_offset)
    head = d.text[start:body]
    depth = 0
    seen_paren = False
    starts = _line_starts(head)
    try:
        for token in tokenize.generate_tokens(io.StringIO(head).readline):
            if token.type != tokenize.OP:
                continue
            if token.string in "([{":
                depth += 1
                seen_paren = True
            elif token.string in ")]}":
                depth -= 1
            elif token.string == ":" and depth == 0 and seen_paren:
                row, col = token.end
                return start + starts[row - 1] + col
    except (tokenize.TokenError, IndentationError):
        return None
    return None


def _positional_edits(
    d: _Definition, scan: _Scan, call: _V1Call, edits: list[Edit]
) -> Optional[str]:
    args = d.node.args
    params = [*args.posonlyargs, *args.args]
    if (
        args.kwonlyargs
        or args.vararg
        or args.kwarg
        or len(params) != len(call.positional)
    ):
        return (
            f"takes {_takes(len(params))}"
            + (
                " with *args, **kwargs or keywords"
                if args.kwonlyargs or args.vararg or args.kwarg
                else ""
            )
            + f", where v1 passed {len(call.positional)}"
        )
    for p, keyword in zip(params, call.positional):
        if p.arg == keyword:
            continue
        if scan.bindings.get(p.arg):
            return f"binds {p.arg} again inside the function"
        if scan.appears(keyword):
            if keyword not in scan.own and scan.loads(keyword):
                return f"reads {keyword} from its deck"
            return f"uses the name {keyword} already"
        for name in scan.names.get(p.arg, []):
            if id(name) in scan.in_fstring:
                return f"uses {p.arg} inside an f-string"
            edits.append(
                (
                    d.offset(name.lineno, name.col_offset),
                    d.offset(name.end_lineno, name.end_col_offset),
                    keyword,
                )
            )
    return None


def _unpacked_edits(
    d: _Definition, scan: _Scan, call: _V1Call, edits: list[Edit]
) -> Optional[str]:
    node = d.node
    args = node.args
    params = [*args.posonlyargs, *args.args]
    if args.kwonlyargs or args.vararg or args.kwarg:
        return "takes *args, **kwargs or keywords, where v1 passed x, y and z"
    if len(params) != 3:
        return f"takes {_takes(len(params))}, where v1 passed x, y and z"
    keyword = call.unpacked
    if scan.appears(keyword):
        if keyword not in scan.own and scan.loads(keyword):
            return f"reads {keyword} from its deck"
        return f"uses the name {keyword} already"
    for p in params:
        if p.arg in scan.declared:
            return f"declares {p.arg} global"
    body = node.body
    docstring = (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )
    anchor = body[1] if docstring and len(body) > 1 else body[0]
    if docstring and len(body) == 1:
        return "has only a docstring"
    head_line = d.node.body[0].lineno
    if head_line == node.lineno or anchor.lineno == body[0].end_lineno and docstring:
        return "has its body on the def line"
    begin = d.line_start(anchor.lineno)
    indent = d.text[begin : d.offset(anchor.lineno, anchor.col_offset)]
    if indent.strip():
        return "has its body on the def line"
    names = ", ".join(p.arg for p in params)
    edits.append((begin, begin, f"{indent}{names} = {keyword}\n"))
    return None


__all__ = ["port_function", "top_level_def"]
