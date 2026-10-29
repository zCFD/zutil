"""Every top-level key the legacy validator accepts has exactly one rule owner."""

from __future__ import annotations

import ast
import fnmatch
import importlib
import pathlib

import pytest

from zutil.update_input.model import norm_key
from zutil.update_input.rules import owner, rule_modules
from zutil.validate import zcfd_validate_legacy

LEGACY = pathlib.Path(zcfd_validate_legacy.__file__)

TOP_LEVEL_SCHEMAS = {
    "all_solver_schema",
    "all_cfd_schema",
    "all_fv_schema",
    "restart_schema",
    "interpolate_restart_schema",
    "fsi_schema",
    "linear_solver_schema",
    "incomp_linear_solver_schema",
    "equations_to_schema",
}

OUTSIDE_THE_SCHEMA = {"air", "mesh transform function"}

EXPECTED_OWNER = {
    "conditions": [
        "reference",
        "initial",
        "ic *",
        "restart",
        "restart casename",
        "restart ignore history",
        "interpolate restart",
        "restart meshname",
        "solution smooth cycles",
    ],
    "equations": [
        "equations",
        "euler",
        "viscous",
        "rans",
        "les",
        "incomp viscous",
        "incomp rans",
        "incomp les",
        "material",
        "air",
    ],
    "time": ["time marching"],
    "boundaries": [
        "bc *",
        "modal model",
        "fsi",
        "sponge layer damping",
        "mapping",
        "create overset halos",
    ],
    "zones": [
        "fz *",
        "tr *",
        "scale",
        "mesh transform matrix",
        "mesh transform function",
    ],
    "output": [
        "write output",
        "report",
        "units",
        "partitioner",
        "safe",
        "linear solver options",
        "mesh quality remediation angle threshold",
        "reset face area",
        "append mesh name",
        "zone id to name",
        "variable name alias",
    ],
}

ASSIGNED = {key: module for module, keys in EXPECTED_OWNER.items() for key in keys}


def _literal_key(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"Required", "Optional"}
        and node.args
    ):
        return _literal_key(node.args[0])
    return None


def _is_deck_key_scan(comprehension: ast.comprehension) -> bool:
    source = ast.unparse(comprehension.iter)
    return source == "list(d.keys())"


def legacy_top_level_keys() -> set[str]:
    """Every top-level key the voluptuous schemas name, from the validator's AST."""
    tree = ast.parse(LEGACY.read_text())
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & TOP_LEVEL_SCHEMAS and isinstance(node.value, ast.Dict):
                for key in node.value.keys:
                    text = _literal_key(key) if key is not None else None
                    if text is not None:
                        keys.add(norm_key(text))
        if isinstance(node, ast.DictComp):
            if not any(_is_deck_key_scan(g) for g in node.generators):
                continue
            for generator in node.generators:
                for condition in generator.ifs:
                    if (
                        isinstance(condition, ast.Call)
                        and isinstance(condition.func, ast.Attribute)
                        and condition.func.attr == "startswith"
                        and isinstance(condition.args[0], ast.Constant)
                    ):
                        keys.add(norm_key(condition.args[0].value) + "*")
    return {_pattern(k) for k in keys}


def _pattern(key: str) -> str:
    return key[:-1] + " *" if key.endswith("*") else key


def all_keys() -> set[str]:
    return legacy_top_level_keys() | OUTSIDE_THE_SCHEMA


def representative(key: str) -> str:
    return key.replace("*", "1")


def missing_modules() -> list[str]:
    absent = []
    for name in EXPECTED_OWNER:
        try:
            importlib.import_module(f"zutil.update_input.rules.{name}")
        except ModuleNotFoundError as error:
            if error.name != f"zutil.update_input.rules.{name}":
                raise
            absent.append(name)
    return absent


def test_the_enumeration_finds_the_documented_key_set():
    keys = legacy_top_level_keys()
    fixed = {k for k in keys if "*" not in k}
    patterns = {k for k in keys if "*" in k}
    assert patterns == {"bc *", "ic *", "fz *", "tr *"}
    assert {
        "write output",
        "report",
        "units",
        "partitioner",
        "safe",
        "linear solver options",
        "create overset halos",
        "mesh quality remediation angle threshold",
        "reset face area",
        "append mesh name",
        "zone id to name",
        "variable name alias",
        "time marching",
        "mapping",
        "scale",
        "mesh transform matrix",
        "sponge layer damping",
        "modal model",
        "fsi",
        "reference",
        "initial",
        "material",
        "restart",
        "interpolate restart",
        "euler",
        "incomp les",
    } <= fixed
    assert "mm *" not in patterns and "fr *" not in patterns


def test_every_key_is_assigned_to_an_expected_package():
    unassigned = sorted(all_keys() - set(ASSIGNED))
    assert not unassigned, f"keys the coverage table does not assign: {unassigned}"
    stale = sorted(set(ASSIGNED) - all_keys())
    assert not stale, (
        f"coverage table names keys the validator does not accept: {stale}"
    )


def test_no_key_has_two_owners():
    doubled = {
        key: [m.__name__ for m in owner(representative(key))]
        for key in sorted(all_keys())
        if len(owner(representative(key))) > 1
    }
    assert not doubled, f"keys owned by more than one rule module: {doubled}"


def test_each_pattern_matches_its_own_representative():
    for module in rule_modules():
        for pattern in module.OWNS:
            assert fnmatch.fnmatchcase(representative(pattern), pattern)


def test_every_key_has_an_owner_once_all_rule_modules_exist():
    absent = missing_modules()
    unowned = sorted(k for k in all_keys() if not owner(representative(k)))
    waiting = sorted(k for k in unowned if ASSIGNED[k] in absent)
    broken = sorted(set(unowned) - set(waiting))
    assert not broken, f"keys with no owner although their rule module exists: {broken}"
    if absent:
        pytest.xfail(
            f"rule modules not written yet: {absent}; keys with no owner yet: {waiting}"
        )
    assert unowned == []


def test_each_existing_module_owns_exactly_its_assigned_keys():
    for module in rule_modules():
        short = module.__name__.rsplit(".", 1)[-1]
        if short not in EXPECTED_OWNER:
            continue
        claimed = {k for k in all_keys() if module in owner(representative(k))}
        assert claimed == set(EXPECTED_OWNER[short]), short
