"""The reference states each boundary-condition rule in the entry's facts."""

from zutil.docs.introspect import applies_when, fingerprint
from zutil.validate.schemas.boundary_conditions_schemas import (
    MassFlowInflowBoundary,
    RoughnessModel,
    WallBoundary,
)


def _facts(model, key):
    return applies_when(model.model_fields[key], key.replace("_", " "))


def test_plain_field_has_no_rule():
    assert _facts(WallBoundary, "velocity") is None


def test_required_clause_is_stated_both_ways():
    facts = _facts(WallBoundary, "most")
    assert facts["required"] is True
    assert (
        facts["text"] == "'most' is required when kind is 'most', and only applies then"
    )


def test_wall_temperature_excludes_heat_flux():
    facts = _facts(WallBoundary, "temperature")
    assert facts["conditions"] is None
    assert facts["exclusive"] == {"with": ["heat_flux"], "required": False}
    assert facts["text"] == "give either 'temperature' or 'heat flux', not both"


def test_required_clause_beside_an_exclusion():
    facts = _facts(MassFlowInflowBoundary, "mass_flow_rate")
    assert facts["conditions"] is None
    assert facts["text"] == (
        "'mass flow rate' is required when solver.solver settings.type is "
        "'compressible' and mass flow ratio is unset; "
        "give either 'mass flow rate' or 'mass flow ratio', not both"
    )


def test_required_exclusion_names_the_choice():
    assert _facts(RoughnessModel, "scalar")["text"] == "give either 'scalar' or 'field'"


def test_parent_clause_reads_from_the_sub_block():
    assert _facts(RoughnessModel, "thermal_model")["text"] == (
        "'thermal model' only applies when kind is 'wall function' and either "
        "temperature is set or heat flux is set"
    )


def test_exclusion_changes_the_fingerprint():
    field = MassFlowInflowBoundary.model_fields["mass_flow_rate"]
    plain = MassFlowInflowBoundary.model_fields["mass_flow_ratio"]
    assert fingerprint(field, "mass flow rate") != fingerprint(plain, "mass flow rate")
