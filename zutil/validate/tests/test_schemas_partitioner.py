"""`partitioner` is a hidden development setting: a deck may set it, the
validator warns when it asks for metis, and neither the UI schema nor the
reference shows it."""

import copy
import runpy
from pathlib import Path

import pytest

from zutil.docs.tree import Walker
from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive
from zutil.validate.zcfd_schema import ZCFDInputModelBase

DECK = (
    Path(__file__).resolve().parents[4]
    / "testcases/testcases/naca0012/n0012_449_a0p0_imp.py"
)


@pytest.fixture(scope="module")
def deck():
    return runpy.run_path(str(DECK))["parameters"]


def _warnings(deck, partitioner):
    deck = copy.deepcopy(deck)
    if partitioner is not None:
        deck["solver"]["solver settings"]["partitioner"] = partitioner
    model, nodes = validate_zcfd_comprehensive(deck)
    assert model is not None
    return [n["message"] for n in nodes if n.get("severity") == "warning"]


def test_metis_warns_that_it_is_a_development_setting(deck):
    (message,) = _warnings(deck, "metis")
    assert "research versions of zCFD" in message
    assert "'scotch'" in message


@pytest.mark.parametrize("partitioner", [None, "scotch"])
def test_scotch_or_unset_is_silent(deck, partitioner):
    assert _warnings(deck, partitioner) == []


def test_the_ui_schema_leaves_it_out():
    assert "partitioner" not in str(ZCFDInputModelBase.model_json_schema())


def test_the_reference_leaves_it_out():
    (page,) = [
        p for p in Walker().build() if p.route == "reference/solver/solver-settings"
    ]
    keys = {e["key"] for e in page.entries}
    assert "type" in keys
    assert "partitioner" not in keys
