"""
Union tags on the inflow branches, and the type choices that
BoundaryConditionBase offers.

Accept/reject coverage for the boundary-condition rules. The cross-field
ones are compatibility clauses (When, AnyOf, Exclusive and the Conditional*
annotations), enforced when the whole deck is validated, so those tests
validate a minimal deck carrying the boundary condition and assert on the
text the engineer reads.
"""

import copy
import typing

import pytest
from pydantic import TypeAdapter, ValidationError

from zutil.validate.schemas.boundary_conditions_schemas import (
    BoundaryCondition,
    BoundaryConditionBase,
    InflowBoundary,
    MassFlowInflowBoundary,
    MassFlowOutflowBoundary,
    OversetConnection,
    OversetInterface,
    PressureInflowBoundary,
    RoughnessModel,
    WallScalarField,
    WallBoundary,
)
from zutil.validate.schemas.base_types import in_case_directory
from zutil.validate.tests.test_case_files import ROUGHNESS_VTP
from zutil.validate.schemas.union_tags import tags_of
from zutil.validate.tests.test_buoyancy_placement import (
    _incompressible_buoyancy_config,
)
from zutil.validate.tests.test_core_orchestration import _minimal_valid_config
from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _deck(bc):
    deck = copy.deepcopy(_minimal_valid_config())
    deck["model"]["base"]["boundary_conditions"]["under_test"] = {"zones": [5], **bc}
    return deck


def _accepted(bc):
    model = ZCFDInputModelBase.model_validate(_deck(bc))
    return model.model["base"].boundary_conditions["under_test"]


def _rejected(bc):
    with pytest.raises(ValidationError) as info:
        ZCFDInputModelBase.model_validate(_deck(bc))
    return "\n".join(
        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in info.value.errors()
    )


WALL_FUNCTION = {"type": "wall", "kind": "wall function"}
ROUGH = {"type": "height", "scalar": 0.001}


def _offered_types():
    """The type choices BoundaryConditionBase offers the UI."""
    return set(typing.get_args(BoundaryConditionBase.model_fields["type"].annotation))


class TestPressureInflowTag:
    """The pressure inflow branch is tagged by its kind literal, "pressure"."""

    def test_pressure_inflow_tag_matches_kind_literal(self):
        tags = set(tags_of(InflowBoundary))
        assert "pressure" in tags
        assert "pressure inlet" not in tags

    def test_pressure_inflow_validates_via_type_adapter(self):
        ta = TypeAdapter(InflowBoundary)
        bc = ta.validate_python(
            {
                "zones": [0],
                "type": "inflow",
                "kind": "pressure",
                "reference": "IC_1",
                "total_pressure_ratio": 1.0,
                "total_temperature_ratio": 1.0,
            }
        )
        assert isinstance(bc, PressureInflowBoundary)

    def test_full_boundary_condition_union_resolves_pressure_inflow(self):
        ta = TypeAdapter(BoundaryCondition)
        bc = ta.validate_python(
            {
                "zones": [0],
                "type": "inflow",
                "kind": "pressure",
            }
        )
        assert isinstance(bc, PressureInflowBoundary)

    def test_massflow_inflow_still_resolves(self):
        ta = TypeAdapter(InflowBoundary)
        bc = ta.validate_python(
            {"zones": [0], "type": "inflow", "kind": "massflow", "mass_flow_rate": 1.0}
        )
        assert isinstance(bc, MassFlowInflowBoundary)


class TestPressureOutflowTakesOnlyWhatTheSolverReads:
    """The solver sets a pressure outflow from 'static_pressure_ratio' alone."""

    @pytest.mark.parametrize(
        "key, value", [("pressure", 90000.0), ("pressure_ramp", {"cycles": 10})]
    )
    def test_unread_key_is_rejected(self, key, value):
        message = _rejected(
            {"type": "outflow", "kind": "pressure", "reference": "IC_1", key: value}
        )
        assert key in message


class TestBoundaryConditionTypeChoices:
    """BoundaryConditionBase offers exactly the BoundaryCondition union's type tags."""

    def test_choices_match_union_tags_exactly(self):
        assert _offered_types() == set(tags_of(BoundaryCondition))

    def test_riemann_invariant_removed(self):
        """No model exists for 'riemann invariant' — it must not be listed."""
        literal_values = _offered_types()
        assert "riemann invariant" not in literal_values

    def test_interface_present(self):
        literal_values = _offered_types()
        assert "interface" in literal_values
        assert not {"overset", "sliding", "periodic"} & literal_values

    def test_pressure_outflow_not_a_real_type_tag(self):
        """'pressure outflow' was never a real 'type' tag (it's 'outflow' + kind='pressure')."""
        literal_values = _offered_types()
        assert "pressure outflow" not in literal_values

    def test_invalid_type_rejected(self):
        ta = TypeAdapter(BoundaryCondition)
        with pytest.raises(ValidationError):
            ta.validate_python({"zones": [0], "type": "riemann invariant"})

    def test_overset_interface_accepted(self):
        ta = TypeAdapter(BoundaryCondition)
        bc = ta.validate_python({"zones": [0], "type": "interface", "kind": "overset"})
        assert (bc.type, bc.kind) == ("interface", "overset")


class TestZoneVsZonesField:
    """`zones` is the sole canonical spelling for the zone-ID list on every
    boundary condition. The schema is the C++ dict — there is no translation
    layer, so the legacy singular `zone` spelling is rejected at parse time.
    """

    def test_legacy_zone_key_rejected_on_base_model(self):
        with pytest.raises(ValidationError):
            BoundaryConditionBase.model_validate({"zone": [0], "type": "interface"})

    def test_zones_key_accepted_on_base_model(self):
        bc = BoundaryConditionBase.model_validate({"zones": [0], "type": "interface"})
        assert bc.zones == [0]

    def test_legacy_zone_key_rejected_via_full_union(self):
        ta = TypeAdapter(BoundaryCondition)
        with pytest.raises(ValidationError):
            ta.validate_python({"zone": [0], "type": "interface", "kind": "overset"})

    def test_zones_key_accepted_via_full_union(self):
        ta = TypeAdapter(BoundaryCondition)
        bc = ta.validate_python({"zones": [0], "type": "interface", "kind": "overset"})
        assert bc.zones == [0]

    def test_mass_flow_report_dumps_with_zones_under_output_settings(self):
        """The solver reads output_settings.report.mass_flow and its
        per-report 'zones' list; the dumped dict must land at exactly that
        path.
        """
        from zutil.validate.schemas.model_schemas import ModelSettings

        model = ModelSettings.model_validate(
            {
                "mesh": "mesh.h5",
                "boundary_conditions": {"BC_1": {"zones": [0], "type": "symmetry"}},
                "output_settings": {
                    "solution": {},
                    "report": {
                        "frequency": 1,
                        "mass_flow": {"MF_1": {"zones": [1, 2]}},
                    },
                },
            }
        )
        dumped = model.model_dump(exclude_none=True)
        mass_flow = dumped["output_settings"]["report"]["mass_flow"]
        assert mass_flow["MF_1"]["zones"] == [1, 2]


class TestRoughnessScalarOrField:
    """Exactly one of 'scalar' and 'field': Exclusive("field", required=True)."""

    @pytest.fixture(autouse=True)
    def _roughness_map(self, tmp_path):
        (tmp_path / "r.vtp").write_text(ROUGHNESS_VTP.format(name="Roughness"))
        with in_case_directory(tmp_path):
            yield

    def test_scalar_only_accepted(self):
        bc = _accepted({**WALL_FUNCTION, "roughness": ROUGH})
        assert bc.roughness.scalar == 0.001
        assert bc.roughness.field is None

    def test_field_only_accepted(self):
        bc = _accepted(
            {**WALL_FUNCTION, "roughness": {"type": "height", "field": "r.vtp"}}
        )
        assert bc.roughness.field == "r.vtp"
        assert bc.roughness.scalar is None

    def test_both_set_rejected(self):
        message = _rejected({**WALL_FUNCTION, "roughness": {**ROUGH, "field": "r.vtp"}})
        assert "roughness.scalar: give either 'scalar' or 'field', not both" in message

    def test_neither_set_rejected(self):
        message = _rejected({**WALL_FUNCTION, "roughness": {"type": "height"}})
        assert "roughness.scalar: give either 'scalar' or 'field'" in message

    def test_unknown_thermal_model_rejected(self):
        with pytest.raises(ValidationError):
            RoughnessModel(type="height", scalar=0.001, thermal_model="nikuradse")

    def test_zero_scalar_accepted(self):
        bc = _accepted(
            {**WALL_FUNCTION, "roughness": {"type": "height", "scalar": 0.0}}
        )
        assert bc.roughness.scalar == 0.0

    def test_negative_scalar_rejected(self):
        message = _rejected(
            {**WALL_FUNCTION, "roughness": {"type": "height", "scalar": -0.001}}
        )
        assert "roughness.scalar" in message
        assert "greater than or equal to 0" in message

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_scalar_rejected(self, value):
        with pytest.raises(ValidationError) as info:
            RoughnessModel(type="height", scalar=value)
        assert info.value.errors()[0]["loc"] == ("scalar",)

    def test_non_negative_scalar_is_a_declared_bound(self):
        from zutil.docs.introspect import facts_of

        facts = facts_of(RoughnessModel.model_fields["scalar"], "scalar")
        assert facts["bounds"] == {"ge": 0.0}
        assert facts["units"] == "m"


class TestOversetConnectionNormalisation:
    def test_string_model_normalised_to_list(self):
        bc = OversetInterface(zones=[0], connection={"model": "donor_mesh"})
        assert bc.connection.model == ["donor_mesh"]

    def test_list_model_passed_through(self):
        bc = OversetInterface(
            zones=[0], connection=OversetConnection(model=["donor_a", "donor_b"])
        )
        assert bc.connection.model == ["donor_a", "donor_b"]

    def test_connection_is_optional(self):
        bc = OversetInterface(zones=[0])
        assert bc.connection is None


RATIO_CASES = [
    ("inflow", "pressure", "total_temperature_ratio", {}),
    ("inflow", "pressure", "static_pressure_ratio", {}),
    ("inflow", "massflow", "total_temperature_ratio", {"mass_flow_rate": 2.0}),
    ("inflow", "massflow", "mass_flow_ratio", {"total_temperature_ratio": 1.0}),
    ("outflow", "massflow", "mass_flow_ratio", {}),
]


class TestRatiosNeedAReference:
    """Every dimensionless ratio scales a reference condition, so it only
    applies when 'reference' is set: RatioToReference, stated once."""

    @pytest.mark.parametrize("bc_type, kind, ratio, extra", RATIO_CASES)
    def test_ratio_without_reference_rejected(self, bc_type, kind, ratio, extra):
        message = _rejected({"type": bc_type, "kind": kind, ratio: 1.2, **extra})
        spelled = ratio.replace("_", " ")
        assert f"'{spelled}' only applies when reference is set" in message

    @pytest.mark.parametrize("bc_type, kind, ratio, extra", RATIO_CASES)
    def test_ratio_with_reference_accepted(self, bc_type, kind, ratio, extra):
        bc = _accepted(
            {"type": bc_type, "kind": kind, "reference": "IC_1", ratio: 1.2, **extra}
        )
        assert getattr(bc, ratio) == 1.2

    def test_no_ratio_no_reference_accepted(self):
        bc = _accepted({"type": "inflow", "kind": "pressure"})
        assert bc.reference is None

    def test_pressure_inflow_ratio_written_at_its_default_needs_a_reference(self):
        """The pressure inlet's total temperature ratio defaults to 1.0; a deck
        that writes it, even as 1.0, without a reference is rejected."""
        message = _rejected(
            {"type": "inflow", "kind": "pressure", "total_temperature_ratio": 1.0}
        )
        assert "'total temperature ratio' only applies when reference is set" in message

    def test_pressure_inflow_ratio_left_at_its_default_needs_no_reference(self):
        bc = _accepted({"type": "inflow", "kind": "pressure"})
        assert bc.total_temperature_ratio == 1.0

    def test_standalone_boundary_condition_defers_the_rule_to_the_deck(self):
        bc = PressureInflowBoundary(zones=[0], static_pressure_ratio=0.9)
        assert bc.static_pressure_ratio == 0.9


MASSFLOW_TEMPERATURE = {
    "inflow": {"reference": "IC_1", "total_temperature_ratio": 1.0},
    "outflow": {},
}


class TestMassFlowRateOrRatio:
    """At most one of 'mass flow rate' and 'mass flow ratio', and the
    compressible solver aborts on a massflow boundary with neither."""

    @pytest.mark.parametrize("bc_type", ["inflow", "outflow"])
    def test_rate_only_accepted(self, bc_type):
        bc = _accepted(
            {
                "type": bc_type,
                "kind": "massflow",
                "mass_flow_rate": 2.0,
                **MASSFLOW_TEMPERATURE[bc_type],
            }
        )
        assert bc.mass_flow_rate == 2.0

    @pytest.mark.parametrize("bc_type", ["inflow", "outflow"])
    def test_ratio_only_accepted(self, bc_type):
        bc = _accepted(
            {
                "type": bc_type,
                "kind": "massflow",
                "reference": "IC_1",
                "mass_flow_ratio": 0.5,
                **MASSFLOW_TEMPERATURE[bc_type],
            }
        )
        assert bc.mass_flow_ratio == 0.5

    @pytest.mark.parametrize("bc_type", ["inflow", "outflow"])
    def test_both_set_rejected(self, bc_type):
        message = _rejected(
            {
                "type": bc_type,
                "kind": "massflow",
                "reference": "IC_1",
                "mass_flow_rate": 2.0,
                "mass_flow_ratio": 0.5,
            }
        )
        assert (
            "mass_flow_rate: give either 'mass flow rate' or 'mass flow ratio', not both"
            in message
        )

    @pytest.mark.parametrize("bc_type", ["inflow", "outflow"])
    def test_neither_set_rejected_for_compressible(self, bc_type):
        message = _rejected(
            {"type": bc_type, "kind": "massflow", **MASSFLOW_TEMPERATURE[bc_type]}
        )
        assert (
            "mass_flow_rate: Set 'mass flow rate' or 'mass flow ratio'; one is "
            "required when solver.solver settings.type is 'compressible'."
        ) in message

    @pytest.mark.parametrize("bc_type", ["inflow", "outflow"])
    def test_neither_set_accepted_for_incompressible(self, bc_type):
        model = ZCFDInputModelBase.model_validate(
            _incompressible_deck({"type": bc_type, "kind": "massflow"})
        )
        assert model.model["base"].boundary_conditions["under_test"].kind == "massflow"


class TestMassFlowIsPositive:
    """The boundary type gives the flow direction, so a mass flow is a
    magnitude: zero and negative values are rejected by a declared bound."""

    def _messages(self, bc):
        from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive

        model, nodes = validate_zcfd_comprehensive(_deck(bc))
        assert model is None
        return [(n["field_path"], n["display_message"]) for n in nodes]

    @pytest.mark.parametrize("bc_type", ["inflow", "outflow"])
    @pytest.mark.parametrize("value", [-2.0, 0.0])
    def test_non_positive_rate_rejected(self, bc_type, value):
        bc = {
            "type": bc_type,
            "kind": "massflow",
            "mass_flow_rate": value,
            **MASSFLOW_TEMPERATURE[bc_type],
        }
        assert self._messages(bc) == [
            (
                "model.base.boundary_conditions.under_test.mass_flow_rate",
                "Mass flow rate must be greater than 0 kg/s.",
            )
        ]

    @pytest.mark.parametrize("bc_type", ["inflow", "outflow"])
    def test_non_positive_ratio_rejected(self, bc_type):
        bc = {
            "type": bc_type,
            "kind": "massflow",
            "reference": "IC_1",
            "mass_flow_ratio": -0.5,
            **MASSFLOW_TEMPERATURE[bc_type],
        }
        assert self._messages(bc) == [
            (
                "model.base.boundary_conditions.under_test.mass_flow_ratio",
                "Mass flow ratio must be greater than 0.",
            )
        ]

    def test_mass_flow_is_not_a_boundary_type(self):
        assert self._messages({"type": "mass flow", "mass_flow": 2.0}) == [
            (
                "model.base.boundary_conditions.under_test",
                "Type must be one of 'wall', 'farfield', 'symmetry', 'inflow', "
                "'outflow', 'interface'.",
            )
        ]

    @pytest.mark.parametrize(
        "model, field",
        [
            (MassFlowInflowBoundary, "mass_flow_rate"),
            (MassFlowInflowBoundary, "mass_flow_ratio"),
            (MassFlowOutflowBoundary, "mass_flow_rate"),
            (MassFlowOutflowBoundary, "mass_flow_ratio"),
        ],
    )
    def test_positive_mass_flow_is_a_declared_bound(self, model, field):
        from zutil.docs.introspect import facts_of

        assert facts_of(model.model_fields[field], field)["bounds"] == {"gt": 0.0}

    @pytest.mark.parametrize("field", ["mass_flow_rate", "mass_flow_ratio"])
    def test_description_names_the_direction(self, field):
        for model in (MassFlowInflowBoundary, MassFlowOutflowBoundary):
            description = model.model_fields[field].description
            assert "the inflow or outflow type sets the direction" in description


def _incompressible_deck(bc):
    deck = _incompressible_buoyancy_config()
    deck["model"]["base"]["boundary_conditions"]["under_test"] = {"zones": [5], **bc}
    return deck


MASSFLOW_INFLOW = {
    "type": "inflow",
    "kind": "massflow",
    "reference": "IC_1",
    "mass_flow_rate": 2.0,
}


FARFIELD = {"type": "farfield", "condition": "IC_1"}


class TestFarfieldSolver:
    """The incompressible solver has no farfield kernels."""

    def test_compressible_accepts_it(self):
        assert _accepted(FARFIELD).type == "farfield"

    def test_incompressible_rejects_it(self):
        with pytest.raises(ValidationError) as info:
            ZCFDInputModelBase.model_validate(_incompressible_deck(FARFIELD))
        message = str(info.value)
        assert "'type': 'farfield' only applies when" in message
        assert (
            "'type': 'farfield' only applies when solver.solver settings.type is "
            "'compressible'; an incompressible domain is bounded with inflow, "
            "outflow, symmetry or wall"
        ) in message


class TestDefaultInflowKind:
    """The 'default' inflow takes its state from the reference condition; only
    the incompressible solver reads it."""

    def test_compressible_is_rejected_naming_the_kinds_to_use(self):
        message = _rejected({"type": "inflow", "kind": "default"})
        assert (
            "'kind': 'default' only applies when solver.solver settings.type is "
            "'incompressible'; a compressible inflow is kind 'pressure' or 'massflow'"
        ) in message

    def test_incompressible_is_accepted(self):
        model = ZCFDInputModelBase.model_validate(
            _incompressible_deck({"type": "inflow", "kind": "default"})
        )
        assert model.model["base"].boundary_conditions["under_test"].kind == "default"


class TestMassFlowInflowTotalTemperature:
    """The compressible solver aborts on a massflow inlet without a total
    temperature ratio; the incompressible solver never reads it."""

    def test_compressible_without_it_is_rejected(self):
        message = _rejected(MASSFLOW_INFLOW)
        assert (
            "total_temperature_ratio: 'total temperature ratio' is required when "
            "solver.solver settings.type is 'compressible'"
        ) in message

    def test_compressible_without_it_or_a_reference_says_both_in_one_message(self):
        inlet = {k: v for k, v in MASSFLOW_INFLOW.items() if k != "reference"}
        message = _rejected(inlet)
        assert (
            "total_temperature_ratio: 'total temperature ratio' is required when "
            "solver.solver settings.type is 'compressible', and only applies when "
            "reference is set"
        ) in message
        assert message.count("total temperature ratio") == 1

    def test_compressible_with_it_is_accepted(self):
        bc = _accepted({**MASSFLOW_INFLOW, "total_temperature_ratio": 1.008})
        assert bc.total_temperature_ratio == 1.008

    @pytest.mark.parametrize("extra", [{}, {"total_temperature_ratio": 1.008}])
    def test_incompressible_accepts_it_either_way(self, extra):
        model = ZCFDInputModelBase.model_validate(
            _incompressible_deck({**MASSFLOW_INFLOW, **extra})
        )
        bc = model.model["base"].boundary_conditions["under_test"]
        assert bc.total_temperature_ratio == extra.get("total_temperature_ratio")


class TestWallThermalState:
    """Mirrors the C++ reader in src/solver/WallScalarInput.h; the two must
    change together. 'temperature' and 'heat flux' share one value shape."""

    def test_unset_is_adiabatic(self):
        bc = WallBoundary(zones=[5])
        assert bc.temperature is None and bc.heat_flux is None

    def test_number_is_uniform(self):
        assert _accepted({"type": "wall", "temperature": 150.0}).temperature == 150.0
        bc = _accepted({**WALL_FUNCTION, "heat_flux": -5.0e3})
        assert bc.heat_flux == -5.0e3

    @pytest.mark.parametrize("key", ["temperature", "heat_flux"])
    def test_field_dict_is_a_map(self, key, tmp_path):
        (tmp_path / "m.vtp").write_text("")
        with in_case_directory(tmp_path):
            bc = _accepted({**WALL_FUNCTION, key: {"field": "m.vtp"}})
        assert isinstance(getattr(bc, key), WallScalarField)
        assert getattr(bc, key).field == "m.vtp"

    @pytest.mark.parametrize("key", ["temperature", "heat_flux"])
    def test_v1_scalar_dict_rejected_with_migration_hint(self, key):
        with pytest.raises(ValidationError, match="write the number 150.0 directly"):
            WallBoundary(zones=[5], kind="wall function", **{key: {"scalar": 150.0}})

    def test_adiabatic_string_rejected_as_redundant(self):
        with pytest.raises(ValidationError, match="adiabatic is the default"):
            WallBoundary(zones=[5], temperature="adiabatic")

    def test_other_string_rejected(self):
        with pytest.raises(ValidationError, match="'field'"):
            WallBoundary(zones=[5], temperature="hot_patch.vtp")

    def test_field_dict_with_extra_key_rejected(self):
        with pytest.raises(ValidationError):
            WallBoundary(zones=[5], temperature={"field": "a.vtp", "scale": 2.0})

    @pytest.mark.parametrize("kelvin", [0.0, -5.0])
    def test_non_positive_kelvin_rejected(self, kelvin):
        with pytest.raises(ValidationError, match="greater than 0"):
            WallBoundary(zones=[5], temperature=kelvin)

    def test_positive_kelvin_is_a_declared_bound(self):
        from zutil.docs.introspect import facts_of

        facts = facts_of(WallBoundary.model_fields["temperature"], "temperature")
        assert facts["bounds"] == {"gt": 0.0}
        assert facts["units"] == "K"

    def test_temperature_and_heat_flux_exclusive(self):
        message = _rejected({**WALL_FUNCTION, "temperature": 150.0, "heat_flux": 5.0e3})
        assert (
            "temperature: give either 'temperature' or 'heat flux', not both" in message
        )

    @pytest.mark.parametrize("kind", ["slip", "no slip", "most"])
    def test_heat_flux_needs_wall_function(self, kind):
        bc = {"type": "wall", "kind": kind, "heat_flux": 5.0e3}
        if kind == "most":
            bc["most"] = {"roughness_length": 0.1}
        message = _rejected(bc)
        assert (
            "heat_flux: 'heat flux' only applies when kind is 'wall function'"
            in message
        )

    @pytest.mark.parametrize("kind", ["slip", "no slip"])
    def test_resolved_wall_takes_a_temperature(self, kind):
        bc = _accepted({"type": "wall", "kind": kind, "temperature": 300.0})
        assert bc.temperature == 300.0

    def test_thermal_wall_function_accepts_roughness(self):
        """Ice roughness under an anti-iced (heated) wall is the icing use case."""
        bc = _accepted({**WALL_FUNCTION, "temperature": 150.0, "roughness": ROUGH})
        assert bc.roughness.thermal_model == "owen thomson"

    def test_roughness_thermal_model_selectable(self):
        bc = _accepted(
            {
                **WALL_FUNCTION,
                "heat_flux": 5.0e3,
                "roughness": {**ROUGH, "thermal_model": "reynolds analogy"},
            }
        )
        assert bc.roughness.thermal_model == "reynolds analogy"

    @pytest.mark.parametrize("model", ["reynolds analogy", "owen thomson"])
    def test_roughness_thermal_model_rejected_without_thermal_state(self, model):
        """Nothing reads it on an adiabatic or resolved wall, so writing it,
        even at its default, must not be silently inert."""
        message = _rejected(
            {**WALL_FUNCTION, "roughness": {**ROUGH, "thermal_model": model}}
        )
        assert (
            "roughness.thermal_model: 'roughness.thermal model' only applies when "
            "kind is 'wall function' and either temperature is set or heat flux is "
            "set" in message
        )

    def test_roughness_thermal_model_rejected_on_a_resolved_wall(self):
        message = _rejected(
            {
                "type": "wall",
                "kind": "no slip",
                "temperature": 300.0,
                "roughness": {**ROUGH, "thermal_model": "owen thomson"},
            }
        )
        assert "'roughness.thermal model' only applies when" in message

    def test_roughness_default_thermal_model_needs_no_thermal_state(self):
        """The default is not written, so a plain rough wall is still accepted."""
        bc = _accepted({**WALL_FUNCTION, "roughness": ROUGH})
        assert bc.roughness.thermal_model == "owen thomson"

    def test_wall_function_dict_is_not_a_key(self):
        with pytest.raises(ValidationError):
            WallBoundary(
                zones=[5], kind="wall function", wall_function={"thermal": True}
            )


class TestMostWall:
    """A 'most' block exists exactly on kind: 'most' walls."""

    def test_most_wall_with_block_accepted(self):
        bc = _accepted(
            {"type": "wall", "kind": "most", "most": {"roughness_length": 0.1}}
        )
        assert bc.most.roughness_length == 0.1

    def test_most_wall_without_block_rejected(self):
        message = _rejected({"type": "wall", "kind": "most"})
        assert "most: 'most' is required when kind is 'most'" in message

    @pytest.mark.parametrize("kind", ["slip", "no slip", "wall function"])
    def test_block_on_another_kind_rejected(self, kind):
        message = _rejected(
            {"type": "wall", "kind": kind, "most": {"roughness_length": 0.1}}
        )
        assert "most: 'most' only applies when kind is 'most'" in message


class TestCompressibleWallFunctionFlag:
    """'compressible wall function' selects the compressible law of the wall
    without a thermal state, i.e. on an adiabatic wall."""

    def test_default_is_off(self):
        bc = _accepted(WALL_FUNCTION)
        assert bc.compressible_wall_function is False
        assert bc.uses_compressible_wall_function is False

    def test_flag_alone_selects_it_adiabatically(self):
        bc = _accepted({**WALL_FUNCTION, "compressible_wall_function": True})
        assert bc.uses_compressible_wall_function is True
        assert bc.has_thermal_state is False

    @pytest.mark.parametrize("thermal", [{"temperature": 300.0}, {"heat_flux": 5.0e3}])
    def test_thermal_state_selects_it_without_the_flag(self, thermal):
        bc = _accepted({**WALL_FUNCTION, **thermal})
        assert bc.compressible_wall_function is False
        assert bc.uses_compressible_wall_function is True

    @pytest.mark.parametrize("kind", ["slip", "no slip"])
    def test_flag_needs_a_wall_function_wall(self, kind):
        message = _rejected(
            {"type": "wall", "kind": kind, "compressible_wall_function": True}
        )
        assert (
            "'compressible wall function': True only applies when kind is "
            "'wall function'" in message
        )

    @pytest.mark.parametrize("kind", ["slip", "no slip"])
    def test_flag_written_off_is_accepted_on_any_wall(self, kind):
        bc = _accepted(
            {"type": "wall", "kind": kind, "compressible_wall_function": False}
        )
        assert bc.compressible_wall_function is False

    def test_flag_accepts_roughness(self):
        """Roughness is supported under the compressible wall function; an adiabatic
        one transfers no heat, so only 'thermal model' is inert there."""
        bc = _accepted(
            {**WALL_FUNCTION, "compressible_wall_function": True, "roughness": ROUGH}
        )
        assert bc.uses_compressible_wall_function is True
        assert bc.roughness.scalar == 0.001

    def test_flag_alone_rejects_a_roughness_thermal_model(self):
        message = _rejected(
            {
                **WALL_FUNCTION,
                "compressible_wall_function": True,
                "roughness": {**ROUGH, "thermal_model": "reynolds analogy"},
            }
        )
        assert "'roughness.thermal model' only applies when" in message

    def test_roughness_still_allowed_on_a_plain_wall_function(self):
        bc = _accepted({**WALL_FUNCTION, "roughness": ROUGH})
        assert bc.uses_compressible_wall_function is False


class TestRuleReachesTheEngineer:
    """The structured validator routes a boundary-condition rule to the key
    the deck wrote, in the deck's spelling."""

    def test_heat_flux_on_a_resolved_wall(self):
        from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive

        deck = _deck({"type": "wall", "kind": "no slip", "heat flux": 5.0e3})
        model, nodes = validate_zcfd_comprehensive(deck)
        assert model is None
        assert [(n["nodes"], n["field_name"], n["display_message"]) for n in nodes] == [
            (
                ["boundary conditions", "under_test"],
                "heat_flux",
                "'heat flux' only applies when kind is 'wall function'",
            )
        ]


MODAL_MODEL = {
    "transform_type": "rbf_multiscale",
    "file_format": "nastran",
    "nastran_casename": "wing",
    "mode_mapping_max_distance": 1.0,
    "fluid_force_scaling": 1.0,
    "mode_list": [1],
    "modal_damping": [0.01],
    "base_point_fraction": 0.5,
}


class TestModalModelZones:
    """A modal model moves the zones of the wall that owns it; the solver
    aborts when that list is empty."""

    def test_empty_zones_rejected(self):
        deck = _deck({"type": "wall", "kind": "slip", "fsi": MODAL_MODEL})
        deck["model"]["base"]["boundary_conditions"]["under_test"]["zones"] = []
        with pytest.raises(ValidationError) as info:
            ZCFDInputModelBase.model_validate(deck)
        message = "\n".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in info.value.errors()
        )
        assert (
            "fsi.type: 'fsi.type': 'modal_model' only applies when zones is not []; "
            "a modal model moves the zones of its boundary condition, so list at "
            "least one under 'zones'"
        ) in message

    def test_zones_given_accepted(self):
        bc = _accepted({"type": "wall", "kind": "slip", "fsi": MODAL_MODEL})
        assert bc.fsi.type == "modal_model"
