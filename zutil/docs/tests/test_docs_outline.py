"""Each reference page draws its block in place from ``parameters``."""

import pytest

from zutil.docs.build import build
from zutil.docs.outline import walkthrough


@pytest.fixture(scope="module")
def pages():
    return {p.route: p.to_json() for p in build(testcases=None).pages}


def _keys(node):
    return [c["key"] for c in node["children"]]


def _descend(node, *keys):
    for key in keys:
        node = next(c for c in node["children"] if c["key"] == key)
    return node


def test_root_outline_is_the_top_level_deck(pages):
    outline = pages["reference/input-deck"]["outline"]
    assert outline["key"] is None and not outline["more"]
    assert _keys(outline) == ["config version", "solver", "model"]
    assert _descend(outline, "config version")["value"] == "2"
    assert _descend(outline, "solver")["link"] == "reference/solver"
    assert _descend(outline, "model")["value"] == "{...}"


def test_solver_outline_lists_every_node_with_its_page(pages):
    page = pages["reference/solver"]
    solver = _descend(page["outline"], "solver")
    assert page["outline"]["more"]
    assert _keys(solver) == [e["deck_key"] for e in page["entries"]]
    assert all(
        c["link"] == e["link"] for c, e in zip(solver["children"], page["entries"])
    )


def test_model_outline_sits_under_a_mesh_name(pages):
    page = pages["reference/model"]
    mesh = _descend(page["outline"], "model", "<mesh>")
    assert _keys(mesh) == [e["deck_key"] for e in page["entries"]]
    assert _descend(mesh, "mesh")["link"] == "reference/model#mesh"


def test_type_page_names_its_type_and_links_entries_on_the_page(pages):
    page = pages["reference/model/boundary-conditions/wall"]
    block = _descend(
        page["outline"], "model", "<mesh>", "boundary conditions", "<name>"
    )
    assert _descend(block, "type")["value"] == '"wall"'
    assert _descend(block, "zones")["link"] == page["route"] + "#zones"


def test_pages_without_keys_have_no_outline(pages):
    assert pages["reference/model/boundary-conditions"]["outline"] is None
    assert pages["reference/output-variables"]["outline"] is None


def test_walkthrough_holds_only_required_solver_blocks(pages):
    w = walkthrough(list(pages.values()))
    solver = _descend(w["deck"], "solver")
    required = [
        e["deck_key"]
        for e in pages["reference/solver"]["entries"]
        if e["facts"]["required"]
    ]
    assert _keys(solver) == required and solver["more"]
    bc = _descend(w["deck"], "model", "<mesh>", "boundary conditions", "<name>")
    assert "type" in _keys(bc) and "zones" in _keys(bc)
    assert [b["key"] for b in w["blocks"]] == ["config version", "solver", "model"]
    assert pages["reference/input-deck"]["walkthrough"] == w
