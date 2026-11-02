"""
Command-line interface for zCFD validation.
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

from colorama import Fore, Style

from .deck import DeckReport, check_deck
from .error_handling import ValidationErrorHandler as ZCFDValidationError


def _use_color(stream=None) -> bool:
    """True if decorative colour/emoji output should be used on `stream`.

    Only when the stream is a TTY and the user hasn't set NO_COLOR
    (https://no-color.org/). Presentation lives here, at the CLI edge —
    never inside exception messages (see validate_input.py). This is the
    single gate for both colorama codes and emoji: never emit either when
    piped/redirected, or when NO_COLOR is set.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    stream = stream if stream is not None else sys.stdout
    try:
        return stream.isatty()
    except Exception:
        return False


def _colorize(text: str, code: str, active: bool) -> str:
    """Wrap `text` in a colorama colour `code`, only when `active`."""
    if not active:
        return text
    return f"{code}{text}{Style.RESET_ALL}"


def _configure_warning_logging(color: bool):
    """Attach a handler to the `zutil.validate` logger so any warnings it
    emits (per the warnings contract) are printed distinctly on stderr,
    without ever failing the run.
    """
    logger = logging.getLogger("zutil.validate")

    class _WarningHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            prefix = "⚠  " if color else "[WARNING] "
            line = f"{prefix}{msg}"
            print(_colorize(line, Fore.YELLOW, color), file=sys.stderr)

    handler = _WarningHandler()
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False


def _run_json(config_file: Path) -> int:
    """Validate `config_file` and print a single structured JSON document
    to stdout: {"valid": bool, "errors": [...], "warnings": [...]}.

    No emoji/colour/decorative text is ever printed in this mode.
    Exit code is 1 only if error-severity nodes are present; a config with
    only warnings exits 0.
    """
    report = check_deck(config_file)
    print(
        json.dumps(
            {
                "valid": report.ok,
                "errors": list(report.errors),
                "warnings": list(report.warnings),
            }
        )
    )
    return 0 if report.ok else 1


def _failure(report: DeckReport, fail: str) -> tuple:
    """``(headline, kind)`` for a deck that did not validate.

    ``kind`` is ``"schema"``, ``"known"`` for a load or routing error, or
    ``"unexpected"``.
    """
    cause = report.cause
    if isinstance(cause, FileNotFoundError):
        return f"{fail}File error: {cause}", "known"
    if isinstance(cause, ValueError):
        return f"{fail}Configuration error: {cause}", "known"
    if cause is None or isinstance(cause, ZCFDValidationError):
        return f"{fail}Configuration validation failed:", "schema"
    return f"{fail}Unexpected error: {cause}", "unexpected"


def _dump_resolved(
    model: object,
    file_path: Optional[Path],
    print_verbose: bool,
    dump_verbose: Optional[str],
) -> None:
    """Pretty-print or dump the resolved configuration dictionary."""

    def _pretty_dict(obj: object, indent: int = 0) -> str:
        space = " " * indent
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            lines = ["{"]
            for k, v in obj.items():
                lines.append(f"{space}    {repr(k)}: {_pretty_dict(v, indent + 4)},")
            lines.append(f"{space}}}")
            return "\n".join(lines)
        elif isinstance(obj, list):
            if not obj:
                return "[]"
            lines = ["["]
            for item in obj:
                lines.append(f"{space}    {_pretty_dict(item, indent + 4)},")
            lines.append(f"{space}]")
            return "\n".join(lines)
        elif callable(obj):
            try:
                name = obj.__name__
            except AttributeError:
                name = str(obj)
            return f"'{name}'"
        else:
            return repr(obj)

    formatted_dict = _pretty_dict(model.model_dump())

    if print_verbose:
        print("\n\n==== FULLY RESOLVED DICTIONARY ====\n")
        print(formatted_dict)
        print("\n===================================\n")

    if dump_verbose is not None:
        output_path = dump_verbose
        if output_path == "AUTO":
            if file_path:
                output_path = file_path.with_name(f"{file_path.stem}_resolved_dict.py")
            else:
                output_path = "resolved_dict.py"

        with open(output_path, "w") as f:
            f.write("parameters = \\\n")
            f.write(formatted_dict)
            f.write("\n")
        print(f"📝 Resolved dictionary saved to {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate zCFD configuration files", prog="validate_input"
    )

    parser.add_argument(
        "casename", help="Path to zCFD configuration file (casename.py)"
    )

    parser.add_argument(
        "-m",
        "--mesh",
        help="Path to mesh file (meshname.h5) - currently unused but preserved for compatibility",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--print-verbose",
        action="store_true",
        help="Dump the full resolved configuration dictionary",
    )

    parser.add_argument(
        "--dump-verbose",
        nargs="?",
        const="AUTO",
        default=None,
        metavar="FILE",
        help="Dump the full resolved configuration dictionary to a file. Optionally provide a filename.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit a single JSON document on stdout instead of human-readable "
            'output: {"valid": bool, "errors": [...], "warnings": [...]}. '
            "Exit code 1 if any errors are present, 0 otherwise (including "
            "warnings-only). No colour/emoji is printed in this mode."
        ),
    )

    args = parser.parse_args()

    config_file = Path(args.casename)

    if args.json:
        sys.exit(_run_json(config_file))

    color_out = _use_color(sys.stdout)
    color_err = _use_color(sys.stderr)
    ok = "✅ " if color_out else ""
    fail = "❌ " if color_err else ""
    search = "\U0001f50d " if color_out else ""
    bug = "\U0001f41b " if color_err else ""

    _configure_warning_logging(color_err)

    print(f"{search}Validating zCFD configuration: {config_file}")

    if args.verbose:
        print(f"{search}Loading and validating configuration...")
    report = check_deck(config_file)
    report.log_warnings()

    if not report.ok:
        headline, kind = _failure(report, fail)
        print(_colorize(headline, Fore.RED, color_err), file=sys.stderr)
        if kind == "schema":
            print(report.text(), file=sys.stderr)
        elif args.verbose and kind == "unexpected":
            print(f"\n{bug}Full traceback:", file=sys.stderr)
            traceback.print_exception(
                type(report.cause), report.cause, report.cause.__traceback__
            )
        sys.exit(1)

    validated_config = report.model
    if args.print_verbose or args.dump_verbose is not None:
        _dump_resolved(
            validated_config, report.origin, args.print_verbose, args.dump_verbose
        )

    print(_colorize(f"{ok}Configuration validation passed!", Fore.GREEN, color_out))

    if args.verbose:
        clipboard = "\U0001f4cb " if color_out else ""
        wrench = "\U0001f527 " if color_out else ""
        ruler = "\U0001f4d0 " if color_out else ""

        print(
            f"{clipboard}Validated configuration type: {type(validated_config).__name__}"
        )

        solver_block = getattr(validated_config, "solver", None)
        if solver_block is not None:
            if hasattr(solver_block, "solver_settings"):
                solver_type = getattr(solver_block.solver_settings, "type", "unknown")
                print(f"{wrench}Solver type: {solver_type}")

            if hasattr(solver_block, "equations"):
                equation_type = getattr(solver_block.equations, "type", "unknown")
                print(f"{ruler}Equation type: {equation_type}")


if __name__ == "__main__":
    main()
