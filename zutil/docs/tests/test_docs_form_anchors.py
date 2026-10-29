"""The form JSON Schema carries each field's reference anchor, so zui never derives one."""

import pytest

from zutil.docs.build import walk_entries
from zutil.docs.tree import Walker
from zutil.validate.zcfd_schema import ZCFDInputModelBase


@pytest.fixture(scope="module")
def schema():
    return ZCFDInputModelBase.model_json_schema()


@pytest.fixture(scope="module")
def emitted():
    out = set()
    for page in Walker().build():
        for entry in walk_entries(page.entries):
            out.add(f"{page.route}#{entry['anchor']}")
        out.add(page.route)
    return out


def properties(schema):
    yield "ZCFDInputModelBase", schema["properties"]
    for name, definition in schema["$defs"].items():
        yield name, definition.get("properties", {})


def anchors_of(prop):
    extra = prop.get("zcfd", {})
    if "anchor" in extra:
        return [extra["anchor"]]
    return list(extra.get("anchors", {}).values())


def test_a_wall_field(schema):
    prop = schema["$defs"]["WallBoundary"]["properties"]["temperature"]
    assert (
        prop["zcfd"]["anchor"] == "reference/model/boundary-conditions/wall/temperature"
    )


def test_a_top_level_node_links_to_its_page(schema):
    assert schema["properties"]["solver"]["zcfd"]["anchor"] == "reference/solver"


def test_a_model_reached_from_two_places_maps_each_place(schema):
    prop = schema["$defs"]["WallScalarField"]["properties"]["field"]
    wall = "model.<mesh>.boundary_conditions.<name>[wall]"
    page = "reference/model/boundary-conditions/wall"
    assert "anchor" not in prop["zcfd"]
    assert prop["zcfd"]["anchors"] == {
        f"{wall}.heat_flux.field": f"{page}/heat-flux#field",
        f"{wall}.temperature.field": f"{page}/temperature#field",
    }


def test_every_property_has_an_anchor(schema):
    missing = [
        f"{model}.{key}"
        for model, props in properties(schema)
        for key, prop in props.items()
        if not anchors_of(prop)
    ]
    assert missing == []


def test_every_anchor_is_one_the_reference_emits(schema, emitted):
    stray = {
        a
        for _, props in properties(schema)
        for prop in props.values()
        for a in anchors_of(prop)
        if a not in emitted
    }
    assert stray == set()
