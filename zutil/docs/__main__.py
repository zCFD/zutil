import argparse
import sys
from pathlib import Path

from . import fragments, sidefiles
from .build import DEFAULT_TESTCASES, build, schema_modules, summary
from .tree import Walker


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zutil.docs", description="Reference data for the zCFD docs"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="emit the reference payload")
    p_build.add_argument("--out", required=True, type=Path)
    p_build.add_argument("--testcases", type=Path, default=DEFAULT_TESTCASES)

    p_fragments = sub.add_parser(
        "fragments", help="check the guide's deck fragments against the reference"
    )
    p_fragments.add_argument("--pages", required=True, type=Path)
    p_fragments.add_argument("--reference", required=True, type=Path)

    p_stamp = sub.add_parser(
        "stamp", help="re-stamp named side-file entries (Model.key)"
    )
    p_stamp.add_argument("names", nargs="*")
    p_stamp.add_argument(
        "--new-files",
        nargs="*",
        type=Path,
        default=None,
        metavar="SIDE_FILE",
        help="stamp every pending entry in these side files (default: untracked side files)",
    )

    p_scaffold = sub.add_parser(
        "scaffold", help="print the entries a module's side file lacks"
    )
    p_scaffold.add_argument(
        "module", help="schema module stem, e.g. boundary_conditions_schemas"
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        result = build(args.out, args.testcases)
        print(summary(result))
        print(f"wrote {args.out}")
        return 1 if result.failed else 0

    if args.command == "fragments":
        problems = fragments.check_pages(args.pages, args.reference)
        for problem in problems:
            print(problem)
        print(f"deck fragments: {len(problems)} not matching the reference")
        return 1 if problems else 0

    walker = Walker()
    walker.build()
    modules = schema_modules(walker)

    if args.command == "stamp":
        if args.new_files is None and not args.names:
            parser.error("stamp needs entry names or --new-files")
        try:
            if args.new_files is not None:
                done = sidefiles.stamp(
                    [], list(modules), new_files=True, paths=args.new_files or None
                )
            else:
                done = sidefiles.stamp(args.names, list(modules))
        except KeyError as err:
            parser.exit(2, f"{err.args[0]}\n")
        for name in done:
            print(f"stamped {name}")
        return 0

    matches = [m for m in modules if m.rsplit(".", 1)[-1] == args.module]
    if not matches:
        parser.error(f"no schema module {args.module}")
    module = matches[0]
    print(
        sidefiles.scaffold(
            modules[module], sidefiles.load(module), walker.discriminators
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
