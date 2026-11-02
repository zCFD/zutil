"""The variable registry and the solver's variable map must name the same set.

`VariableAccess::variableMap` in the solver decides which output variables a
run will actually write: a name it does not hold is logged as
"Output Variable <name> not recognised" and dropped, and the case runs on
without that field. `output_variables.py` decides which names a deck is
allowed to ask for. Nothing but this test ties the two together, and while
they were free to drift they did — in both directions.

The test reads the solver source, so it is skipped where zutil is installed
away from the repository.
"""

import re
from pathlib import Path

import pytest

from zutil.validate.schemas.output_variables import (
    TIERS_BY_EQUATION,
    TRANSITION_TURBULENCE_MODELS,
    get_valid_variables_for_equation_type,
)

#: Both maps register the same names; the solver picks one by solver family.
SOLVER_VARIABLE_MAPS = (
    "src/solver/VariableAccess.cxx",
    "src/solver/output/VariableAccess.cxx",
)

#: var_1, resvar_2, ... are generated per solved equation on both sides, so
#: they are compared by count elsewhere rather than name-by-name here.
DYNAMIC_NAME = re.compile(r"^(var|resvar|resshare|vargrad)_\d+$")

#: `std::string("<key>"), std::make_tuple(...)` — the key a deck writes.
REGISTRATION = re.compile(r'std::string\("([^"]*)"\)\s*,\s*std::make_tuple')


def _repo_root():
    root = Path(__file__).resolve().parents[4]
    return root if (root / SOLVER_VARIABLE_MAPS[0]).exists() else None


def _solver_variable_names():
    root = _repo_root()
    if root is None:
        pytest.skip("solver sources are not next to zutil in this install")

    names = set()
    for relative in SOLVER_VARIABLE_MAPS:
        source = (root / relative).read_text()
        found = REGISTRATION.findall(source)
        # The registration style changed if this finds nothing, and an empty
        # set would agree with anything.
        assert len(found) > 100, f"{relative}: found {len(found)} registrations"
        names.update(found)

    assert "p" in names and "badcells" in names
    return {n for n in names if not DYNAMIC_NAME.match(n)}


def _requestable_names():
    """Every name some valid configuration may ask for."""
    names = set()
    for equation_type in TIERS_BY_EQUATION:
        for context in ("volume", "surface"):
            for turbulence_model in (None,) + TRANSITION_TURBULENCE_MODELS:
                names.update(
                    get_valid_variables_for_equation_type(
                        equation_type,
                        context,
                        turbulence_model,
                        has_mesh_velocity=True,
                        has_fsi=True,
                    )
                )
    return {n for n in names if not DYNAMIC_NAME.match(n)}


def test_every_requestable_name_is_one_the_solver_accepts():
    """Otherwise a deck passes validation and the solver drops the variable."""
    phantom = sorted(_requestable_names() - _solver_variable_names())
    assert phantom == []


def test_every_name_the_solver_accepts_can_be_requested():
    """Otherwise validation rejects a variable the solver would have written."""
    unreachable = sorted(_solver_variable_names() - _requestable_names())
    assert unreachable == []
