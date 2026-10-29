"""Command line for converting a v1 zCFD control deck to config version 2.

Exit codes: 0 when the new deck is written (valid, or a draft with open
decisions); 1 when a deck or the mesh cannot be read, or for a usage
problem; 2 when the v1 deck fails the legacy validator; 3 for a converter
bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_LEGACY = 2
EXIT_BUG = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="update_input",
        description=(
            "Convert a v1 zCFD control deck to config version 2. "
            "Writes <deck>_v2.py and <deck>_v2.conversion.md beside the deck; "
            "the input is never changed."
        ),
    )
    parser.add_argument(
        "deck",
        help="the v1 control deck (deck.py), or an overset group (override.py)",
    )
    parser.add_argument(
        "--mesh",
        help=(
            "the deck's native h5 mesh; names the model and fills BC zones from 'ref' "
            "codes. An overset group names its own meshes"
        ),
    )
    parser.add_argument(
        "--ignore-legacy-errors",
        action="store_true",
        help="convert a deck that fails the legacy validator, listing its errors in the report",
    )
    parser.add_argument(
        "--output", help="where to write the new deck (default <deck>_v2.py)"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing output and report"
    )
    return parser


def _error(text: str) -> None:
    print(f"update_input: {text}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .convert import (
        ConverterBugError,
        LegacyDeckError,
        convert_deck,
        default_output,
        report_path_for,
    )
    from .mesh import MeshReadError

    args = _parser().parse_args(argv)
    deck = Path(args.deck)
    if not deck.is_file():
        _error(f"{deck} does not exist")
        return EXIT_USAGE

    output = Path(args.output) if args.output else default_output(deck)
    refusal = _refusal([output, report_path_for(output)], [deck], args.force)
    if refusal:
        _error(refusal)
        return EXIT_USAGE

    from . import source

    try:
        result = convert_deck(
            deck,
            mesh_path=args.mesh,
            ignore_legacy_errors=args.ignore_legacy_errors,
            output_path=output,
        )
    except LegacyDeckError as e:
        print(str(e), file=sys.stderr)
        return EXIT_LEGACY
    except ConverterBugError as e:
        print(str(e), file=sys.stderr)
        return EXIT_BUG
    except (MeshReadError, source.DeckReadError, source.RenderError) as e:
        _error(f"cannot convert {deck.name}: {e}")
        return EXIT_USAGE

    decks = [r.deck_path for r in result.bases + [result] if r.deck_path is not None]
    decks += result.members
    files = result.files
    refusal = _refusal([path for path, _ in files], decks, args.force)
    if refusal:
        _error(refusal)
        return EXIT_USAGE

    for path, text in files:
        path.write_text(text)
    print(result.summary)
    return EXIT_OK


def _refusal(outputs: list[Path], inputs: list[Path], force: bool) -> str:
    """Why writing ``outputs`` is refused, or an empty string."""
    resolved = {Path(p).resolve() for p in inputs}
    for path in outputs:
        if Path(path).resolve() in resolved:
            return f"writing {path.name} would overwrite an input deck; choose another --output"
    existing = [p for p in outputs if Path(p).exists()]
    if existing and not force:
        names = " and ".join(p.name for p in existing)
        return f"{names} already exists; pass --force to overwrite"
    return ""


if __name__ == "__main__":
    sys.exit(main())
