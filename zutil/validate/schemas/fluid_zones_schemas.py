from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
)
from typing import Literal, Union, Optional, List, Annotated
from pydantic.json_schema import SkipJsonSchema
from zutil.validate.schemas.base_types import (
    UNSET,
    ConditionalChoice,
    ConditionalRequired,
    Disjoint,
    Exclusive,
    NotIn,
    Rotation,
    When,
    case_file,
    zCFD_Field,
)
from .base_types import Vector3d, NormalisedVector3d
from .callables import (
    LEAF_AREA_DENSITY,
    MESH_DEFORMATION,
    MOMENTUM_SOURCE,
    LeafAreaDensityFunction,
    MeshDeformationFunction,
    MomentumSourceFunction,
)
from .transform_schemas import (
    RBFTransformConfig,
    IDWTransformConfig,
    MeshDeformationConfig,
)
from .union_tags import literal_of

# Zone kinds
zone_kind_types = Literal["fixed", "ramp", "func"]

CellZones = Annotated[List[int], Field(min_length=1)]


def _cell_zones(region: str, alternatives: str) -> str:
    return (
        f"Mesh cell zone numbers whose cells form the {region}, for example "
        f'"cell zones": [4]; give this, {alternatives}, or neither for the '
        "whole mesh"
    )


_ONE_FRAME = When(
    {
        "<mesh>.transforms.motion": (UNSET,),
        "<mesh>.boundary_conditions.*.kind": NotIn("overset", "sliding"),
    },
    text=("the model has no 'transforms > motion' and no overset or sliding interface"),
    hint=(
        "A model has one frame: rotating fluid zones on its cell zones, or "
        "'transforms > motion' for the whole mesh. A mesh with an overset or "
        "sliding interface turns as a whole, so it takes 'transforms > motion' "
        "only. Remove 'motion' from 'transforms', or remove the rotating fluid "
        'zones and turn the whole mesh with "transforms": {"motion": {"type": '
        '"rotating", "axis": [...], "origin": [...], "omega": ..., '
        '"moving mesh": False}}.'
    ),
)


class FluidZoneBase(BaseModel):
    """UI base model for fluid zones.

    Exposes only the common required field shared by every fluid zone
    variant — ``type``.  The UI renders it first when the user adds a new
    fluid zone, before a concrete type is selected.

    Once the user chooses a ``type`` value, use
    :func:`~.ui_utils.get_union_variants` on the ``Fluid_Zone_Base_Model``
    union to resolve the concrete model class and render its full field set.
    """

    model_config = ConfigDict(extra="forbid")
    type: "FluidZoneType" = zCFD_Field(..., title="Type", description="Fluid zone type")


class MomentumSourceZone(FluidZoneBase):
    """Momentum source fluid zone."""

    type: Literal["momentum"] = "momentum"

    cell_zones: Annotated[Optional[CellZones], Exclusive("definition")] = zCFD_Field(
        default=None,
        title="Cell Zones",
        description=_cell_zones("source region", "'definition'"),
    )
    definition: Optional[str] = zCFD_Field(
        default=None,
        title="Definition",
        description=(
            "Closed-surface VTP file selecting the source region cells; "
            "without it or 'cell zones' the source applies to the whole mesh"
        ),
    )

    kind: Optional[zone_kind_types] = zCFD_Field(
        default="fixed", title="Kind", description="Source kind"
    )
    frequency: Optional[int] = zCFD_Field(
        default=1, ge=1, title="Frequency", description="Update frequency"
    )
    func: SkipJsonSchema[Optional[MomentumSourceFunction]] = zCFD_Field(
        default=None,
        title="Func",
        description=MOMENTUM_SOURCE.description(
            "Function of the cell centres giving the source, in the units of "
            "'source', in every cell of the source"
        ),
    )

    # Source vector
    source: Optional[Vector3d] = zCFD_Field(
        default=None, title="Source", description="Momentum source vector"
    )


turbine_controller_types = Literal["tsr curve", "fixed", "schedule"]
turbine_model_types = Literal[
    "simple", "bet"
]  # normalised lowercase — replace_spaces_in_keys() lowercases any "type" value
turbine_model_kinds = Literal["propellor", "turbine"]
turbine_power_model_types = Literal["curve", "fixed"]
turbine_tip_loss_correction_types = Literal[
    "elliptic", "acos-fit", "acos shift-fit", "f-fit", "none", "rstar"
]


class TurbineController(BaseModel):
    """Turbine/BET controller configuration."""

    model_config = ConfigDict(extra="forbid")

    type: turbine_controller_types = zCFD_Field(
        ..., title="Type", description="Controller type"
    )
    pitch: Optional[float] = zCFD_Field(
        default=None, title="Pitch", description="Pitch angle"
    )
    omega: Optional[float] = zCFD_Field(
        default=None,
        title="Omega",
        description="Fixed rotational speed (rad/s), used with type='fixed'",
    )
    tip_speed_ratio: Optional[float] = zCFD_Field(
        default=None,
        title="Tip Speed Ratio",
        description="Target tip speed ratio, used with type='tsr curve'",
    )
    tip_speed_ratio_curve: Optional[List[List[float]]] = zCFD_Field(
        default=None,
        title="Tip Speed Ratio Curve",
        description="Tip speed ratio vs. time/velocity curve, as [x, y] pairs",
    )


class TurbineGeometry(BaseModel):
    """Turbine geometry configuration."""

    model_config = ConfigDict(extra="forbid")

    centre: Vector3d = zCFD_Field(..., title="Centre", description="Centre coordinates")
    inner_radius: Optional[float] = zCFD_Field(
        default=None, title="Inner Radius", description="Inner radius"
    )
    outer_radius: Optional[float] = zCFD_Field(
        default=None, title="Outer Radius", description="Outer radius"
    )
    normal: Vector3d = zCFD_Field(..., title="Normal", description="Normal vector")
    up: Vector3d = zCFD_Field(..., title="Up", description="Up vector")


class TurbineDiscretisation(BaseModel):
    """Turbine discretisation configuration."""

    model_config = ConfigDict(extra="forbid")

    type: str = zCFD_Field(..., title="Type", description="Discretisation type")
    number_of_elements: Optional[int] = zCFD_Field(
        default=None, title="Number of Elements", description="Number of elements"
    )


class TurbineModel(BaseModel):
    """Turbine/BET aerodynamic model configuration."""

    model_config = ConfigDict(extra="forbid")

    kind: Optional[turbine_model_kinds] = zCFD_Field(
        default=None, title="Kind", description="Model kind (propellor or turbine)"
    )
    type: Optional[turbine_model_types] = zCFD_Field(
        default=None,
        title="Type",
        description="Model type (simple actuator disc or blade-element BET)",
    )
    aerofoils: Optional[dict] = zCFD_Field(
        default=None,
        title="Aerofoils",
        description=(
            "Aerofoil polar data keyed by aerofoil name (genuinely "
            "open-ended — aerofoil names and curve shapes are user-defined)."
        ),
    )
    aerofoil_positions: Optional[List[list]] = zCFD_Field(
        default=None,
        title="Aerofoil Positions",
        description="Radial position to aerofoil-name mapping, as [position, name] pairs",
    )
    blade_chord: Optional[List[List[float]]] = zCFD_Field(
        default=None, title="Blade Chord", description="Blade chord distribution"
    )
    blade_twist: Optional[List[List[float]]] = zCFD_Field(
        default=None, title="Blade Twist", description="Blade twist distribution"
    )
    number_of_sections: Optional[int] = zCFD_Field(
        default=None, title="Number of Sections", description="Number of sections"
    )
    number_of_blades: Optional[int] = zCFD_Field(
        default=None, title="Number of Blades", description="Number of blades"
    )
    power: Optional[float] = zCFD_Field(
        default=None, title="Power", description="Fixed power (W)"
    )
    power_model: Optional[turbine_power_model_types] = zCFD_Field(
        default=None, title="Power Model", description="Power model kind"
    )
    power_curve: Optional[List[List[float]]] = zCFD_Field(
        default=None,
        title="Power Curve",
        description="Power vs. tip speed ratio curve",
    )
    power_coefficient: Optional[float] = zCFD_Field(
        default=None,
        gt=0,
        title="Power Coefficient",
        description=(
            "Fixed power coefficient for the simple propellor model — "
            "torque/power are derived from it (zutil.zWind.ForceModels."
            "zPropellorSimpleFixed)"
        ),
    )
    thrust_coefficient: Optional[float] = zCFD_Field(
        default=None,
        title="Thrust Coefficient",
        description="Fixed thrust coefficient",
    )
    thrust_coefficient_curve: Optional[List[List[float]]] = zCFD_Field(
        default=None,
        title="Thrust Coefficient Curve",
        description="Thrust coefficient vs. tip speed ratio curve",
    )
    tip_loss_correction: Optional[turbine_tip_loss_correction_types] = zCFD_Field(
        default=None,
        title="Tip Loss Correction",
        description="Tip loss correction method",
    )
    tip_loss_correction_radius: Optional[float] = zCFD_Field(
        default=None,
        title="Tip Loss Correction Radius",
        description="Tip loss correction radius",
    )


class ActuatorDiscZone(BaseModel):
    """Actuator disc fluid zone."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["disc"] = "disc"

    # Disc properties
    thrust_coefficient: Optional[float] = zCFD_Field(
        default=None, title="Thrust Coefficient", description="Thrust coefficient"
    )
    disc_area: Optional[float] = zCFD_Field(
        default=None, gt=0, title="Disc Area", description="Disc area"
    )
    tip_speed_ratio: Optional[float] = zCFD_Field(
        default=None, title="Tip Speed Ratio", description="Tip speed ratio"
    )

    # Reference conditions
    reference_density: Optional[float] = zCFD_Field(
        default=None, gt=0, title="Reference Density", description="Reference density"
    )
    # Read by zWind/zTurbine.py:62 (`control_dict.get("verbose", False)`) —
    # gates the turbine-model/controller load messages.
    verbose: Optional[bool] = zCFD_Field(
        default=False,
        title="Verbose",
        description="Print turbine model and controller details at load",
        advanced=True,
    )

    # Read by zWind/zTurbine.py:65 (`control_dict["u_ref"]`).
    u_ref: Optional[float] = zCFD_Field(
        default=None,
        gt=0,
        title="Reference Velocity",
        description=(
            "Reference velocity (m/s) override for the BET force "
            "calculation — used instead of the sampled freestream, e.g. "
            "for hover cases where the freestream is near zero"
        ),
    )

    # Turbine-specific fields (Turbine/BET models use 'disc' type). Required:
    # the C++ consumer does an unguarded .getDict() on each, so a bare
    # {"type": "disc"} would validate here but crash the solver at setup.
    controller: TurbineController = zCFD_Field(
        title="Controller", description="Turbine controller"
    )
    geometry: TurbineGeometry = zCFD_Field(
        title="Geometry", description="Turbine geometry"
    )
    discretisation: TurbineDiscretisation = zCFD_Field(
        title="Discretisation", description="Turbine discretisation"
    )
    model: TurbineModel = zCFD_Field(
        title="Model", description="Turbine model definition"
    )

    # Metadata/Other
    name: str = zCFD_Field(title="Name", description="Zone name")
    reference_plane: Optional[bool] = zCFD_Field(
        default=False, title="Reference Plane", description="Use reference plane"
    )
    reference_point: Optional[Vector3d] = zCFD_Field(
        default=None, title="Reference Point", description="Reference point"
    )
    definition: ConditionalRequired[
        str,
        When(
            {"<self>.type": ("disc",)},
            text="the fluid zone is an actuator disc",
            hint="A disc selects its cells with a closed-surface VTP file.",
        ),
    ] = zCFD_Field(
        default=None,
        title="Definition",
        description="Closed-surface VTP file that selects the disc's cells",
    )
    update_frequency: Optional[int] = zCFD_Field(
        default=1, title="Update Frequency", description="Update frequency"
    )
    rotation_direction: str = zCFD_Field(
        title="Rotation Direction", description="Rotation direction"
    )


class PorousMediaZone(FluidZoneBase):
    """Porous media fluid zone."""

    type: Literal["porous"] = "porous"

    cell_zones: Annotated[Optional[CellZones], Exclusive("definition")] = zCFD_Field(
        default=None,
        title="Cell Zones",
        description=_cell_zones("porous region", "'definition'"),
    )
    definition: Optional[str] = zCFD_Field(
        default=None,
        title="Definition",
        description=(
            "Closed-surface VTP file selecting the porous region cells; "
            "without it or 'cell zones' the porous region is the whole mesh"
        ),
    )

    # Porous media properties
    alpha: float = zCFD_Field(
        default=1.0e12,
        ge=0,
        title="Alpha",
        description="Resistance coefficient (Darcy term)",
    )
    # Model-dict keys are lowercased at load (replace_spaces_in_keys), so the
    # field is spelled c2; decks may still write 'C2'.
    c2: float = zCFD_Field(
        default=0.0,
        ge=0,
        title="C2",
        description="Inertial coefficient (Forchheimer term)",
    )


CanopyHeightMap = case_file("canopy height", point_array="Height")


class CanopyZone(BaseModel):
    """Canopy model fluid zone for atmospheric boundary layer simulations with vegetation."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["canopy"] = "canopy"

    cell_zones: Annotated[Optional[CellZones], Exclusive("definition", "field")] = (
        zCFD_Field(
            default=None,
            title="Cell Zones",
            description=_cell_zones("canopy", "'definition' or 'field'"),
        )
    )
    definition: Optional[str] = zCFD_Field(
        default=None,
        title="Definition",
        description=(
            "Closed-surface VTP file selecting the canopy cells; without it, "
            "'cell zones' or 'field' the canopy covers the whole mesh"
        ),
    )

    # Alternative field definition
    field: Optional[CanopyHeightMap] = zCFD_Field(
        default=None,
        title="Field",
        description="Field file for canopy definition (alternative to VTP)",
    )

    # Leaf area density function
    function: SkipJsonSchema[Optional[LeafAreaDensityFunction]] = zCFD_Field(
        default=None,
        title="Function",
        description=LEAF_AREA_DENSITY.description(
            "Function of the cell centres and their wall distances giving the "
            "leaf area density (1/m) in every canopy cell"
        ),
    )

    # Canopy turbulence model parameters
    cd: Optional[float] = zCFD_Field(
        default=0.25,
        ge=0,
        title="Cd",
        description="Drag coefficient for canopy momentum sink",
    )

    beta_p: Optional[float] = zCFD_Field(
        default=0.17,
        ge=0,
        title="Beta P",
        description="Canopy turbulence production parameter",
    )

    beta_d: Optional[float] = zCFD_Field(
        default=3.37,
        ge=0,
        title="Beta D",
        description="Canopy turbulence dissipation parameter",
    )

    ceps_4: Optional[float] = zCFD_Field(
        default=0.9,
        ge=0,
        title="Ceps 4",
        description="Canopy C_epsilon_4 turbulence parameter",
    )

    ceps_5: Optional[float] = zCFD_Field(
        default=0.9,
        ge=0,
        title="Ceps 5",
        description="Canopy C_epsilon_5 turbulence parameter",
    )


class RotatingZone(Rotation, FluidZoneBase):
    """A rotating frame on mesh cell zones (MRF); the mesh stays still."""

    type: ConditionalChoice[  # type: ignore[valid-type]
        Literal["rotating"], {"rotating": _ONE_FRAME}
    ] = "rotating"
    cell_zones: Annotated[CellZones, Disjoint("cell zone", "rotating fluid zone")] = (
        zCFD_Field(
            ...,
            title="Cell Zones",
            description=(
                "Mesh cell zone numbers whose cells are solved in the rotating "
                'frame, for example "cell zones": [17]'
            ),
        )
    )


class ABLZone(FluidZoneBase):
    """Atmospheric boundary layer fluid zone, applied to the whole mesh."""

    type: Literal["abl"] = "abl"

    # ABL properties
    roughness_length: Optional[float] = zCFD_Field(
        default=None,
        gt=0,
        title="Roughness Length",
        description="Surface roughness length",
    )
    friction_velocity: Optional[float] = zCFD_Field(
        default=None, gt=0, title="Friction Velocity", description="Friction velocity"
    )
    reference_height: Optional[float] = zCFD_Field(
        default=None, gt=0, title="Reference Height", description="Reference height"
    )

    # Wind direction
    wind_direction: Optional[NormalisedVector3d] = zCFD_Field(
        default=None, title="Wind Direction", description="Wind direction vector"
    )


# Transform zone inheritance hierarchy
class TransformZoneBase(BaseModel):
    """Base class for all transform zones."""

    model_config = ConfigDict(extra="forbid")

    zones: List[int] = zCFD_Field(
        ...,
        title="Zones",
        description="Boundary zone IDs of the moving surfaces",
    )
    func: SkipJsonSchema[MeshDeformationFunction] = zCFD_Field(
        title="Func",
        description=MESH_DEFORMATION.description(
            "Function of the node position giving the deformed position of "
            "every surface node"
        ),
    )


class MeshDeformationTransformBase(TransformZoneBase, MeshDeformationConfig):
    """Base class for mesh deformation transforms."""

    pass


class IDWTransformZone(TransformZoneBase, IDWTransformConfig):
    """Inverse Distance Weighting transform fluid zone."""

    type: Literal["idw transform"] = "idw transform"
    zones: List[int] = zCFD_Field(
        ...,
        min_length=1,
        title="Zones",
        description="Boundary zones the morph moves.",
    )


class RBFTransformZone(TransformZoneBase, RBFTransformConfig):
    """Radial Basis Function transform fluid zone.

    Full RBF when base_point_fraction=1.0 (default),
    Multiscale RBF when base_point_fraction < 1.0.
    """

    type: Literal["rbf transform"] = "rbf transform"
    zones: List[int] = zCFD_Field(
        ...,
        min_length=1,
        title="Zones",
        description="Boundary zones the morph moves.",
    )


Fluid_Zone_Base_Model = Annotated[
    Union[
        MomentumSourceZone,
        RotatingZone,
        ActuatorDiscZone,
        PorousMediaZone,
        CanopyZone,
        ABLZone,
        IDWTransformZone,
        RBFTransformZone,
    ],
    Discriminator("type"),
]

FluidZoneType = literal_of(Fluid_Zone_Base_Model)
FluidZoneBase.model_rebuild()
