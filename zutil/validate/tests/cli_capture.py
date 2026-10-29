"""Run the validate_input CLI over a fixed set of decks and record what it prints.

Usage, from the repository root::

    /usr/bin/python3 zutil/zutil/validate/tests/cli_capture.py <zutil project dir> <output.json>

Recording with this repository's zutil into ``cli_expected.json`` updates what
``test_deck_cli_output.py`` compares against. Every command runs from the
repository root, so the relative paths printed by the CLI are the same
whichever zutil is under test; the root's absolute path is stored as
``<ROOT>``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DECKS = "zutil/zutil/validate/tests/decks"
TESTCASES = "testcases/testcases"

CASES = [
    ["valid", f"{TESTCASES}/naca0012/n0012_449_a0p0_imp.py"],
    ["valid_verbose", "-v", f"{TESTCASES}/naca0012/n0012_449_a0p0_imp.py"],
    ["invalid", f"{TESTCASES}/dpw/dpw_base.py"],
    ["missing_sections", f"{DECKS}/missing_sections.py"],
    ["warning_only", f"{TESTCASES}/plate/plate_coarse.py"],
    ["legacy", f"{TESTCASES}/cavity/buoyancy_cavity.py"],
    ["no_parameters", f"{TESTCASES}/naca0012/batch_stub.py"],
    ["deck_raises", f"{TESTCASES}/turbine/turbine_bet.py"],
    ["missing_file", "does_not_exist.py"],
    ["no_config_version", f"{DECKS}/no_config_version.py"],
    ["unsupported_version", f"{DECKS}/unsupported_version.py"],
    ["deprecated_dg", f"{DECKS}/deprecated_dg.py"],
    ["override", f"{DECKS}/override_case.py"],
    ["duplicate_keys", f"{DECKS}/duplicate_keys.py"],
    ["capitalised_bc_type", f"{DECKS}/capitalised_bc_type.py"],
]


def run_cli(zutil_root: str, args: list) -> dict:
    """One CLI run as ``{"rc", "stdout", "stderr"}`` with ``<ROOT>`` substituted."""
    env = dict(os.environ, PYTHONPATH=zutil_root)
    env.pop("NO_COLOR", None)
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "zutil.validate.cli", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    def clean(text):
        return text.replace(str(ROOT), "<ROOT>").replace(zutil_root, "<ZUTIL>")

    return {
        "rc": result.returncode,
        "stdout": clean(result.stdout),
        "stderr": clean(result.stderr),
    }


def capture(zutil_root: str) -> dict:
    """Text and ``--json`` runs for every case."""
    runs = {}
    for name, *args in CASES:
        runs[f"{name}:text"] = run_cli(zutil_root, args)
        if "-v" not in args:
            runs[f"{name}:json"] = run_cli(zutil_root, ["--json", *args])
    return runs


if __name__ == "__main__":
    zutil_root, out = sys.argv[1], sys.argv[2]
    Path(out).write_text(json.dumps(capture(zutil_root), indent=1, sort_keys=True))
    print(f"{len(CASES)} cases written to {out}")
