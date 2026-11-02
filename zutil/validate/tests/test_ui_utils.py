"""Tests for the UI introspection utilities."""

from pydantic import BaseModel

from zutil.validate.schemas.ui_utils import (
    get_discriminator_field,
    get_union_variants,
    is_union_type,
)
from zutil.validate.schemas.boundary_conditions_schemas import (
    BoundaryCondition,
    BoundaryConditionBase,
    InterfaceBoundaryBase,
    WallBoundary,
    FarfieldBoundary,
    SymmetryBoundary,
    OversetInterface,
    SlidingInterface,
    PeriodicInterface,
    PressureInflowBoundary,
    MassFlowInflowBoundary,
    PressureOutflowBoundary,
    MassFlowOutflowBoundary,
    RadialPressureGradientOutflowBoundary,
)
from zutil.validate.schemas.fluid_zones_schemas import (
    Fluid_Zone_Base_Model,
    FluidZoneBase,
    MomentumSourceZone,
    RotatingZone,
    ActuatorDiscZone,
    PorousMediaZone,
    CanopyZone,
    ABLZone,
    IDWTransformZone,
    RBFTransformZone,
)
from zutil.validate.schemas.motion_schemas import (
    Motion,
    RotatingMotion,
    TranslatingMotion,
)
from zutil.validate.schemas.equations_schemas import EquationsBase


# ── BoundaryCondition ────────────────────────────────────────────────────


class TestBoundaryConditionDiscovery:
    """Tests for BoundaryCondition discriminated union introspection."""

    def test_discriminator_field(self):
        assert get_discriminator_field(BoundaryCondition) == "type"

    def test_base_model_fields(self):
        fields = list(BoundaryConditionBase.model_fields.keys())
        assert "type" in fields
        assert "zones" in fields

    def test_variant_keys(self):
        variants = get_union_variants(BoundaryCondition)
        expected = {
            "wall",
            "farfield",
            "symmetry",
            "inflow",
            "outflow",
            "interface",
        }
        assert set(variants.keys()) == expected

    def test_single_stage_variants_are_models(self):
        variants = get_union_variants(BoundaryCondition)
        assert variants["wall"] is WallBoundary
        assert variants["farfield"] is FarfieldBoundary
        assert variants["symmetry"] is SymmetryBoundary

    def test_two_stage_variants_are_unions(self):
        variants = get_union_variants(BoundaryCondition)
        assert is_union_type(variants["inflow"])
        assert is_union_type(variants["outflow"])
        assert is_union_type(variants["interface"])

    def test_single_stage_variants_not_unions(self):
        variants = get_union_variants(BoundaryCondition)
        assert not is_union_type(variants["wall"])
        assert not is_union_type(variants["farfield"])
        assert not is_union_type(variants["symmetry"])


# ── Inflow (second-level) ────────────────────────────────────────────────


class TestInflowBoundaryDiscovery:
    """Tests for InflowBoundary sub-union introspection."""

    def test_discriminator_field(self):
        variants = get_union_variants(BoundaryCondition)
        inflow_union = variants["inflow"]
        assert get_discriminator_field(inflow_union) == "kind"

    def test_variant_keys(self):
        variants = get_union_variants(BoundaryCondition)
        inflow_variants = get_union_variants(variants["inflow"])
        # Tag matches the model's discriminator value ("kind" == "pressure"),
        # not the free-text "pressure inlet".
        assert set(inflow_variants.keys()) == {"pressure", "massflow", "default"}

    def test_variant_classes(self):
        variants = get_union_variants(BoundaryCondition)
        inflow_variants = get_union_variants(variants["inflow"])
        assert inflow_variants["pressure"] is PressureInflowBoundary
        assert inflow_variants["massflow"] is MassFlowInflowBoundary


# ── Outflow (second-level) ───────────────────────────────────────────────


class TestOutflowBoundaryDiscovery:
    """Tests for OutflowBoundary sub-union introspection."""

    def test_discriminator_field(self):
        variants = get_union_variants(BoundaryCondition)
        outflow_union = variants["outflow"]
        assert get_discriminator_field(outflow_union) == "kind"

    def test_variant_keys(self):
        variants = get_union_variants(BoundaryCondition)
        outflow_variants = get_union_variants(variants["outflow"])
        assert set(outflow_variants.keys()) == {
            "pressure",
            "massflow",
            "radial pressure gradient",
        }

    def test_variant_classes(self):
        variants = get_union_variants(BoundaryCondition)
        outflow_variants = get_union_variants(variants["outflow"])
        assert outflow_variants["pressure"] is PressureOutflowBoundary
        assert outflow_variants["massflow"] is MassFlowOutflowBoundary
        assert (
            outflow_variants["radial pressure gradient"]
            is RadialPressureGradientOutflowBoundary
        )


# ── Interface (second-level) ─────────────────────────────────────────────


class TestInterfaceBoundaryDiscovery:
    def test_discriminator_field(self):
        variants = get_union_variants(BoundaryCondition)
        assert get_discriminator_field(variants["interface"]) == "kind"

    def test_variant_classes(self):
        variants = get_union_variants(BoundaryCondition)
        interface_variants = get_union_variants(variants["interface"])
        assert interface_variants == {
            "overset": OversetInterface,
            "sliding": SlidingInterface,
            "periodic": PeriodicInterface,
        }

    def test_base_model_fields(self):
        fields = list(InterfaceBoundaryBase.model_fields.keys())
        assert "type" in fields
        assert "zones" in fields
        assert "kind" in fields


# ── Fluid Zones ──────────────────────────────────────────────────────────


class TestFluidZoneDiscovery:
    """Tests for Fluid_Zone_Base_Model introspection (Discriminator-based pattern)."""

    def test_discriminator_field(self):
        assert get_discriminator_field(Fluid_Zone_Base_Model) == "type"

    def test_base_model_fields(self):
        fields = list(FluidZoneBase.model_fields.keys())
        assert fields == ["type"]

    def test_variant_keys(self):
        variants = get_union_variants(Fluid_Zone_Base_Model)
        expected = {
            "momentum",
            "rotating",
            "disc",
            "porous",
            "canopy",
            "abl",
            "idw transform",
            "rbf transform",
        }
        assert set(variants.keys()) == expected

    def test_variant_classes(self):
        variants = get_union_variants(Fluid_Zone_Base_Model)
        assert variants["momentum"] is MomentumSourceZone
        assert variants["rotating"] is RotatingZone
        assert variants["disc"] is ActuatorDiscZone
        assert variants["porous"] is PorousMediaZone
        assert variants["canopy"] is CanopyZone
        assert variants["abl"] is ABLZone
        assert variants["idw transform"] is IDWTransformZone
        assert variants["rbf transform"] is RBFTransformZone

    def test_all_variants_are_concrete(self):
        """Fluid zones have no two-stage discriminators."""
        variants = get_union_variants(Fluid_Zone_Base_Model)
        for name, cls in variants.items():
            assert not is_union_type(cls), f"{name} should be a concrete model"


class TestMotionDiscovery:
    """Tests for the Motion union introspection."""

    def test_discriminator_field(self):
        assert get_discriminator_field(Motion) == "type"

    def test_variants(self):
        assert get_union_variants(Motion) == {
            "rotating": RotatingMotion,
            "translating": TranslatingMotion,
        }

    def test_rotate_nodes_is_offered_only_for_rotation(self):
        assert "rotate_nodes" in RotatingMotion.model_fields
        assert "rotate_nodes" not in TranslatingMotion.model_fields

    def test_both_motions_carry_the_frame_settings(self):
        for motion in (RotatingMotion, TranslatingMotion):
            assert {"moving_mesh", "ramp_time"} <= set(motion.model_fields)


# ── Equations ────────────────────────────────────────────────────────────


class TestEquationsBase:
    def test_base_model_fields(self):
        fields = list(EquationsBase.model_fields.keys())
        assert "type" in fields
        assert "precondition" in fields


# ── is_union_type edge cases ────────────────────────────────────────────


class TestIsUnionType:
    """Edge case tests for is_union_type."""

    def test_concrete_model_class(self):
        assert not is_union_type(WallBoundary)

    def test_base_model(self):
        assert not is_union_type(BaseModel)

    def test_top_level_union(self):
        assert is_union_type(BoundaryCondition)

    def test_sub_union(self):
        variants = get_union_variants(BoundaryCondition)
        assert is_union_type(variants["inflow"])
