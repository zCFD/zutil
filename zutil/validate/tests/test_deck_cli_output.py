"""The validate_input CLI prints exactly what ``cli_expected.json`` records.

Each case runs the CLI under test from the repository root and compares exit
code, stdout and stderr byte for byte. Re-record the file with
``cli_capture.py`` when a message changes on purpose.
"""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cli_capture import CASES, run_cli  # noqa: E402

ZUTIL_UNDER_TEST = str(HERE.parents[2])
EXPECTED = json.loads((HERE / "cli_expected.json").read_text())


def _runs():
    for name, *args in CASES:
        yield f"{name}:text", args
        if "-v" not in args:
            yield f"{name}:json", ["--json", *args]


RUNS = [pytest.param(key, args, id=key) for key, args in _runs()]


@pytest.mark.parametrize("key,args", RUNS)
def test_cli_output_matches_the_recording(key, args):
    assert run_cli(ZUTIL_UNDER_TEST, args) == EXPECTED[key]


def test_duplicate_keys_json_is_an_error_document():
    deck = next(args[-1] for name, *args in CASES if name == "duplicate_keys")
    result = run_cli(ZUTIL_UNDER_TEST, ["--json", deck])

    assert result["rc"] == 1
    assert result["stderr"] == ""
    payload = json.loads(result["stdout"])
    assert payload["valid"] is False
    assert payload["warnings"] == []
    [error] = payload["errors"]
    assert error["display_message"] == (
        "Duplicate key: 'numerical scheme' and 'numerical_scheme' are the same "
        "setting spelt two ways. Use one spelling."
    )


def test_the_recording_covers_a_valid_and_an_invalid_testcase():
    assert EXPECTED["valid:text"]["rc"] == 0
    assert "Configuration validation passed!" in EXPECTED["valid:text"]["stdout"]
    assert EXPECTED["invalid:text"]["rc"] == 1
    assert json.loads(EXPECTED["invalid:json"]["stdout"])["valid"] is False
