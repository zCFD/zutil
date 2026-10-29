"""
Subprocess tests for the `zutil.validate.cli` command-line entry point.

Runs the CLI exactly as a user/CI job would (``python -m zutil.validate.cli``)
against:
  - a real, valid config_version 2 testcase (testcases/testcases/cavity/re400.py)
  - an invalid config_version 2 dict constructed in tmp_path
  - a rejected config_version (3)

Covers --json output shape/exit codes, absence of ANSI escapes when piped,
stderr vs stdout routing, and the verbose path.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
ZUTIL_SRC = (
    Path(__file__).resolve().parents[3]
)  # .../zutil (outer, contains the `zutil` package)
VALID_TESTCASE = REPO_ROOT / "testcases" / "testcases" / "cavity" / "re400.py"

ANSI_ESCAPE = "\x1b"


def _run_cli(args, cwd=None):
    """Run `python -m zutil.validate.cli <args>` as a subprocess.

    The repo's `zutil` source is put first on PYTHONPATH so the subprocess
    picks up the in-repo package rather than any stale installed copy.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ZUTIL_SRC) + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-m", "zutil.validate.cli", *args],
        cwd=str(cwd or REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture()
def invalid_config(tmp_path):
    """A structurally invalid config_version 2 config (missing every
    required solver/model section).
    """
    path = tmp_path / "invalid_case.py"
    path.write_text(
        "parameters = {\n"
        '    "config_version": 2,\n'
        '    "solver": {"equations": {"type": "viscous"}},\n'
        '    "model": {},\n'
        "}\n"
    )
    return path


@pytest.fixture()
def unsupported_version_config(tmp_path):
    """A config declaring an unsupported config_version (3)."""
    path = tmp_path / "v3_case.py"
    path.write_text(
        "parameters = {\n"
        '    "config_version": 3,\n'
        '    "solver": {},\n'
        '    "model": {},\n'
        "}\n"
    )
    return path


def _assert_no_ansi(*texts):
    for text in texts:
        assert ANSI_ESCAPE not in text, f"unexpected ANSI escape in: {text!r}"


class TestValidConfig:
    def test_exit_zero_and_no_ansi_when_piped(self):
        assert VALID_TESTCASE.exists(), f"missing fixture testcase: {VALID_TESTCASE}"
        result = _run_cli([str(VALID_TESTCASE)])
        _assert_no_ansi(result.stdout, result.stderr)
        assert result.returncode == 0, result.stderr
        assert "passed" in result.stdout.lower()
        assert result.stderr == ""

    def test_verbose_shows_solver_and_equation_type(self):
        """Verbose must print the solver and equation type.

        They live under `.solver.` on the root model, not at the top level.
        """
        result = _run_cli(["-v", str(VALID_TESTCASE)])
        assert result.returncode == 0, result.stderr
        assert "Solver type:" in result.stdout
        assert "Equation type:" in result.stdout

    def test_json_valid_shape_and_no_decoration(self):
        result = _run_cli(["--json", str(VALID_TESTCASE)])
        assert result.returncode == 0, result.stderr
        _assert_no_ansi(result.stdout, result.stderr)
        assert "✅" not in result.stdout and "❌" not in result.stdout
        payload = json.loads(result.stdout)
        assert payload == {"valid": True, "errors": [], "warnings": []}


class TestInvalidConfig:
    def test_exit_one_and_errors_on_stderr(self, invalid_config):
        result = _run_cli([str(invalid_config)])
        _assert_no_ansi(result.stdout, result.stderr)
        assert result.returncode == 1
        assert result.stderr != ""
        assert "failed" in result.stderr.lower()
        # The success banner must never appear when validation failed.
        assert "passed" not in result.stdout.lower()

    def test_json_invalid_shape_and_exit_code(self, invalid_config):
        result = _run_cli(["--json", str(invalid_config)])
        _assert_no_ansi(result.stdout, result.stderr)
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["valid"] is False
        assert isinstance(payload["errors"], list)
        assert len(payload["errors"]) > 0
        assert isinstance(payload["warnings"], list)
        for node in payload["errors"]:
            assert {
                "nodes",
                "field_name",
                "display_message",
                "message",
                "field_path",
                "anchor",
            } <= set(node.keys())
            assert "›" not in node["field_path"]
        assert any(
            node["anchor"] and node["anchor"].startswith("reference/")
            for node in payload["errors"]
        )

    def test_json_stdout_is_single_json_document(self, invalid_config):
        """--json mode must suppress all the decorative banner prints —
        stdout should be parseable as exactly one JSON document.
        """
        result = _run_cli(["--json", str(invalid_config)])
        # json.loads succeeds only if stdout is *just* the JSON document.
        json.loads(result.stdout)


class TestConfigVersion:
    def test_unsupported_version_rejected(self, unsupported_version_config):
        """config_version 3 is not accepted; the message says version
        2 is the only supported version, and the code must match that.
        """
        result = _run_cli([str(unsupported_version_config)])
        assert result.returncode == 1
        assert "Unsupported configuration version: 3" in result.stderr
        assert "Supported versions are: 2" in result.stderr

    def test_unsupported_version_rejected_json(self, unsupported_version_config):
        result = _run_cli(["--json", str(unsupported_version_config)])
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["valid"] is False
        assert any(
            "Unsupported configuration version: 3" in n["message"]
            for n in payload["errors"]
        )


class TestMissingFile:
    def test_missing_file_exits_one_on_stderr(self, tmp_path):
        missing = tmp_path / "does_not_exist.py"
        result = _run_cli([str(missing)])
        assert result.returncode == 1
        assert result.stderr != ""
        assert "passed" not in result.stdout.lower()
