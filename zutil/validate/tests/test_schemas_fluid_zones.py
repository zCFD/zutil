"""Cell selection of the volumetric sources.

A momentum, porous or canopy source selects its region with 'cell zones',
the mesh cell zone numbers, or 'definition', a closed-surface VTP file, and
applies to the whole mesh with neither. A canopy may give 'field' instead.
An ABL zone applies to the whole mesh. A rotating zone selects its cells
with 'cell zones' only.
"""

import pytest
from pydantic import ValidationError

from zutil.validate.applicability import find_applicability_violations
from zutil.validate.schemas.fluid_zones_schemas import (
    ABLZone,
    CanopyZone,
    IDWTransformZone,
    MomentumSourceZone,
    PorousMediaZone,
    RBFTransformZone,
    RotatingZone,
)


def _dummy_func(*, locations):
    return [(0.0, 0.0, 0.0) for _ in locations]


def _deform(*, location):
    return location


SOURCES = [
    (MomentumSourceZone, {"func": _dummy_func}),
    (PorousMediaZone, {}),
    (CanopyZone, {}),
]
ROTATION = {"axis": [0.0, 0.0, 1.0], "origin": [0.0, 0.0, 0.0], "omega": 1.0}


@pytest.mark.parametrize("model, extra", SOURCES)
def test_definition_selects_the_region(model, extra):
    zone = model(definition="region.vtp", **extra)
    assert zone.definition == "region.vtp"
    assert find_applicability_violations(zone) == []


@pytest.mark.parametrize("model, extra", SOURCES)
def test_cell_zones_select_the_region(model, extra):
    zone = model(cell_zones=[4, 5], **extra)
    assert zone.cell_zones == [4, 5]
    assert find_applicability_violations(zone) == []


@pytest.mark.parametrize("model, extra", SOURCES)
def test_neither_is_the_whole_mesh(model, extra):
    zone = model(**extra)
    assert zone.definition is None and zone.cell_zones is None


@pytest.mark.parametrize("model, extra", SOURCES)
def test_cell_zones_and_definition_are_alternatives(model, extra):
    zone = model(cell_zones=[4], definition="region.vtp", **extra)
    [(path, message)] = find_applicability_violations(zone)
    assert path == "cell_zones"
    assert "'cell zones'" in message and "'definition'" in message


def test_a_canopy_field_is_a_third_alternative():
    zone = CanopyZone(cell_zones=[4], field="height.vtp")
    assert find_applicability_violations(zone) == [
        ("cell_zones", "give only one of 'cell zones', 'definition' or 'field'")
    ]


@pytest.mark.parametrize("model, extra", SOURCES + [(RotatingZone, ROTATION)])
def test_an_empty_cell_zone_list_is_rejected(model, extra):
    with pytest.raises(ValidationError, match="at least 1 item"):
        model(cell_zones=[], **extra)


@pytest.mark.parametrize(
    "model, extra",
    SOURCES + [(ABLZone, {}), (RotatingZone, {**ROTATION, "cell_zones": [1]})],
    ids=lambda m: getattr(m, "__name__", ""),
)
def test_zones_rejected(model, extra):
    with pytest.raises(ValidationError, match="zones"):
        model(zones=[0], **extra)


def test_abl_takes_no_cell_zones():
    with pytest.raises(ValidationError, match="cell_zones"):
        ABLZone(cell_zones=[0])


def test_a_rotating_zone_is_a_rotation_on_cell_zones():
    zone = RotatingZone(cell_zones=[17], **ROTATION)
    assert zone.type == "rotating"
    assert zone.model_dump() == {"type": "rotating", "cell_zones": [17], **ROTATION}


def test_a_rotating_zone_needs_cell_zones():
    with pytest.raises(ValidationError, match="cell_zones"):
        RotatingZone(**ROTATION)


@pytest.mark.parametrize(
    "key, value", [("porosity", 0.5), ("direction", [1.0, 0.0, 0.0])]
)
def test_porous_zone_rejects_unread_key(key, value):
    with pytest.raises(ValidationError, match=key):
        PorousMediaZone(cell_zones=[0], **{key: value})


class TestMorphTransformZones:
    """The solver aborts on a morph transform that moves no zones."""

    @pytest.mark.parametrize("zone_cls", [IDWTransformZone, RBFTransformZone])
    def test_empty_zones_rejected(self, zone_cls):
        with pytest.raises(ValidationError, match="at least 1 item"):
            zone_cls(zones=[], func=_deform)

    @pytest.mark.parametrize("zone_cls", [IDWTransformZone, RBFTransformZone])
    def test_zones_given_accepted(self, zone_cls):
        assert zone_cls(zones=[3], func=_deform).zones == [3]
