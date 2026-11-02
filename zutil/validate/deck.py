"""The one route from a zCFD deck to a verdict.

:func:`check_deck` loads a deck, routes it on ``config_version`` (rejecting
legacy, override and deprecated-codepath decks), normalises its keys once,
validates it against the v2 schema and returns a :class:`DeckReport`. The
solver, live reload, the ``validate_input`` CLI and zui all render that
report; none of them routes a deck on its own.

A deck reaches the schema only when it declares ``config_version: 2``.
Anything that stops it earlier is reported as one error node and kept as
:attr:`DeckReport.cause`, the exception that stopped it, so callers that
raise can re-raise it unchanged.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .error_handling import UnsupportedDeckError, ValidationErrorHandler
from .error_mapping import format_errors_as_text
from .schemas.base_types import in_solver_directory
from .validate_zcfd_schema import validate_zcfd_comprehensive
from .validation_utils import lowercase_keys_recursive, replace_spaces_in_keys

SUPPORTED_CONFIG_VERSION = 2

DeckSource = Union[str, Path, dict]


@dataclass(frozen=True)
class DeckReport:
    """The verdict on one deck.

    ``errors`` and ``warnings`` are nodes in the shape
    :func:`validate_zcfd_comprehensive` returns; a node with no
    ``severity`` counts as an error. ``model`` is the validated model, or
    ``None`` when there are errors. ``parameters`` is the deck with its keys
    normalised, or ``None`` when the deck never reached the schema.
    ``origin`` is the file the deck was read from, if known. ``cause`` is
    the exception that stopped the deck before the schema, or ``None``.
    """

    errors: tuple = ()
    warnings: tuple = ()
    model: object = None
    parameters: Optional[dict] = None
    origin: Optional[Path] = None
    cause: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        """True when the deck has no errors; warnings never make it false."""
        return not self.errors

    def text(self) -> str:
        """The errors as the terminal text the CLI and the solver log print."""
        if self.cause is not None:
            return str(self.cause)
        return format_errors_as_text(list(self.errors), self.parameters, self.origin)

    def warning_lines(self) -> list:
        """One log line per warning, as the CLI and the solver print them."""
        return [w.get("message") or w.get("display_message") for w in self.warnings]

    def log_warnings(self) -> None:
        """Log each warning line on the ``zutil.validate`` logger."""
        for line in self.warning_lines():
            logging.getLogger("zutil.validate").warning(line)

    def model_or_raise(self) -> object:
        """The validated model, or the exception a raising caller expects.

        Raises ``cause`` when the deck stopped before the schema, and
        :class:`ValidationErrorHandler` carrying :meth:`text` when the schema
        rejected it.
        """
        if self.cause is not None:
            raise self.cause
        if self.errors:
            raise ValidationErrorHandler(self.text())
        return self.model


def check_deck(source: DeckSource, origin: Union[str, Path, None] = None) -> DeckReport:
    """Load, route and validate one deck.

    ``source`` is a path to a deck file, or a parameters dict. ``origin`` is
    the file a dict was read from; it names the deck in messages and locates
    the sub-cases of an override deck. Never raises for a bad deck.
    """
    routed = route_deck(source, origin)
    if isinstance(routed, DeckReport):
        return routed
    parameters, origin_path = routed

    with in_solver_directory():
        model, nodes = validate_zcfd_comprehensive(parameters)
    errors = tuple(n for n in nodes if (n.get("severity") or "error") == "error")
    warnings = tuple(n for n in nodes if n.get("severity") == "warning")
    return DeckReport(
        errors=errors,
        warnings=warnings,
        model=None if errors else model,
        parameters=parameters,
        origin=origin_path,
    )


def route_deck(
    source: DeckSource, origin: Union[str, Path, None] = None
) -> Union[tuple, DeckReport]:
    """Everything :func:`check_deck` does before the schema.

    Returns ``(parameters, origin)`` for a deck the v2 schema should see,
    with ``parameters`` normalised, or a failed :class:`DeckReport`.
    """
    origin_path = Path(origin).resolve() if origin else None
    try:
        raw, origin_path, is_override = _load(source, origin_path)
        _route(raw, origin_path, is_override)
        return replace_spaces_in_keys(raw), origin_path
    except Exception as e:
        return DeckReport(errors=(_error_node(str(e)),), origin=origin_path, cause=e)


def _load(source: DeckSource, origin: Optional[Path]) -> tuple:
    """``(raw parameters, origin, is_override)`` for one source."""
    from zutil.fileutils import get_parameters_from_file

    if isinstance(source, (str, Path)):
        origin = Path(source).resolve()
        original_cwd = os.getcwd()
        try:
            os.chdir(origin.parent)
            raw, key_found = get_parameters_from_file(
                origin, parameters_key=["parameters", "override"]
            )
        finally:
            os.chdir(original_cwd)
        return raw, origin, key_found == "override" or _names_mesh_case_pairs(raw)
    if isinstance(source, dict):
        return source, origin, _names_mesh_case_pairs(source)
    raise TypeError(f"A deck is a file path or a dict, not {type(source).__name__}.")


def _names_mesh_case_pairs(raw: object) -> bool:
    return isinstance(raw, dict) and "mesh_case_pair" in raw


def _normalised(key: object) -> object:
    return key.replace(" ", "_").lower() if isinstance(key, str) else key


def _lookup(data: dict, name: str, default: object = None) -> object:
    """``data[name]`` under any spelling that normalises to ``name``."""
    return next((v for k, v in data.items() if _normalised(k) == name), default)


def _route(raw: object, origin: Optional[Path], is_override: bool) -> None:
    """Raise for any deck the v2 schema must not see."""
    if is_override:
        _reject_override(raw, origin)
    if not isinstance(raw, dict):
        raise TypeError(f"'parameters' must be a dict, not {type(raw).__name__}.")

    config_version = _lookup(raw, "config_version", 0)
    if not isinstance(config_version, int):
        config_version = 0

    _reject_deprecated_codepaths(raw)

    if config_version == 0:
        _reject_legacy(raw, origin)
    if config_version != SUPPORTED_CONFIG_VERSION:
        raise UnsupportedDeckError(
            f"Unsupported configuration version: {config_version}. "
            f"Supported versions are: {SUPPORTED_CONFIG_VERSION}"
        )


def _reject_deprecated_codepaths(data: dict) -> None:
    """Raise for the DG, FRPM, CAA and immersed boundary codepaths, whatever the config_version."""
    immersed = _immersed_boundary_errors(data)
    if immersed:
        raise ValidationErrorHandler(
            f"Configuration validation failed with {len(immersed)} error(s):\n"
            + "\n".join(f"  • {e}" for e in immersed)
        )

    solver_data = _lookup(data, "solver", {})
    if not isinstance(solver_data, dict):
        return

    equations = _lookup(solver_data, "equations", "")
    eq_type = (
        _lookup(equations, "type", "") if isinstance(equations, dict) else equations
    )
    if str(eq_type).lower().startswith("dg") or any(
        str(_normalised(k)).startswith("frpm") or _normalised(k) == "caa" for k in data
    ):
        raise ValidationErrorHandler(
            "Configuration validation failed with 1 error(s):\n"
            "  • high order and CAA codepaths are deprecated "
            "and no longer supported."
        )


def _immersed_boundary_errors(data: dict) -> list[str]:
    """One message for each immersed wall and immersed boundary surface in a deck."""
    try:
        d = replace_spaces_in_keys(data)
    except ValueError:
        return []
    errors = []
    models = d.get("model")
    for name, model in models.items() if isinstance(models, dict) else ():
        bcs = model.get("boundary_conditions") if isinstance(model, dict) else None
        for label, bc in bcs.items() if isinstance(bcs, dict) else ():
            if isinstance(bc, dict) and bc.get("type") == "immersed wall":
                errors.append(
                    f"model → {name} → boundary conditions → {label}: the "
                    "'immersed wall' boundary condition has been removed; mesh "
                    "the body and use a 'wall' boundary."
                )
    solver = d.get("solver")
    output = solver.get("output_settings") if isinstance(solver, dict) else None
    solution = output.get("solution") if isinstance(output, dict) else None
    if isinstance(solution, dict) and "immersed_boundary_surface" in solution:
        errors.append(
            "solver → output settings → solution: 'immersed boundary surface' "
            "has been removed; delete it."
        )
    return errors


def _reject_legacy(data: dict, origin: Optional[Path]) -> None:
    """Raise for a config_version 0 deck, saying whether the legacy validator accepts it.

    The legacy validator is given the deck with lower-cased keys, as the v1
    solver read it, keeping the legacy spaced spellings.
    """
    legacy_errors = None
    try:
        from .zcfd_validate_legacy import validate as legacy_validate

        legacy_validate(
            lowercase_keys_recursive(data), case_name=str(origin or "unknown")
        )
    except Exception as e:
        legacy_errors = str(e)

    file_name = origin.name if origin else "input"
    command = f"update_input {origin.name if origin else '<deck>.py'}"
    if legacy_errors is None:
        raise UnsupportedDeckError(
            f"Legacy configuration detected (Config Version 0).\n"
            f"  '{file_name}' is valid for the legacy system.\n"
            f"  It is NOT compatible with the new schema.\n"
            f"  Please update to 'config_version: 2'. "
            f"Convert it with: {command}"
        )
    raise UnsupportedDeckError(
        f"Legacy configuration detected (Config Version 0).\n"
        f"  '{file_name}' also has legacy validation errors:\n"
        f"     {legacy_errors}\n"
        f"  Please update to 'config_version: 2'. "
        f"Fix the errors above, then convert it with: {command}"
    )


def _reject_override(data: dict, origin: Optional[Path]) -> None:
    """Raise for an override (overset) deck, summarising its mesh-case pairs.

    Each sub-case is checked against the legacy validator when the deck's
    file is known, so the message says which ones would still load.
    """
    from zutil.fileutils import get_parameters_from_file

    file_name = origin.name if origin else "input"
    mesh_case_pair = data.get("mesh_case_pair", [])

    errors = []
    if not isinstance(mesh_case_pair, list):
        errors.append("'mesh_case_pair' must be a list of (mesh, case) tuples")
    elif len(mesh_case_pair) == 0:
        errors.append("'mesh_case_pair' is empty")
    else:
        for i, pair in enumerate(mesh_case_pair):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                errors.append(
                    f"  mesh_case_pair[{i}]: expected (mesh, case) tuple, "
                    f"got {type(pair).__name__}"
                )

    sub_case_results = []
    if not errors and origin:
        for mesh_file, case_file in mesh_case_pair:
            case_path = origin.parent / case_file
            if not case_path.exists():
                sub_case_results.append((case_file, mesh_file, "file not found"))
                continue

            original_cwd = os.getcwd()
            try:
                os.chdir(case_path.parent)
                case_params = get_parameters_from_file(
                    case_path, parameters_key="parameters"
                )
                from .zcfd_validate_legacy import validate as legacy_validate

                legacy_validate(
                    lowercase_keys_recursive(case_params), case_name=str(case_path)
                )
                sub_case_results.append((case_file, mesh_file, "valid"))
            except Exception as e:
                sub_case_results.append((case_file, mesh_file, str(e)))
            finally:
                os.chdir(original_cwd)

    lines = ["Overset configuration detected (override file)."]
    if errors:
        lines.append(f"  '{file_name}' has structural errors:")
        lines.extend(f"     {err}" for err in errors)
    elif sub_case_results:
        valid_count = sum(1 for _, _, s in sub_case_results if s == "valid")
        total = len(sub_case_results)
        lines.append(
            f"  Found {total} mesh-case pair(s), "
            f"{valid_count}/{total} valid for the legacy system:"
        )
        lines.extend(
            f"    {mesh_file} -> {case_file}: {status}"
            for case_file, mesh_file, status in sub_case_results
        )
    else:
        lines.append(
            f"  Found {len(mesh_case_pair)} mesh-case pair(s) "
            f"(sub-case validation skipped — no file path available)."
        )
    lines.append(
        "  Overset configurations are no longer supported in the new schema.\n"
        "  Please migrate to 'config_version: 2' — "
        "see documentation for migration details."
    )
    raise UnsupportedDeckError("\n".join(lines))


def _error_node(message: str) -> dict:
    """An error node, in the shape the schema returns, for a deck stopped before it."""
    return {
        "nodes": [],
        "field_name": "",
        "display_message": message,
        "message": message,
        "field_path": "",
        "anchor": None,
        "severity": "error",
    }


__all__ = ["DeckReport", "check_deck", "route_deck", "SUPPORTED_CONFIG_VERSION"]
