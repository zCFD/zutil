from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Discriminator,
    Tag,
    field_validator,
)
import math
from typing import Literal, Union, Optional, List, Annotated
from zutil.validate.schemas.base_types import Connects, ModelWide, zCFD_Field
from .base_types import (
    SET,
    UNSET,
    Advises,
    AnyOf,
    case_file,
    ConditionalChoice,
    ConditionalDefault,
    ConditionalOptional,
    ConditionalRequired,
    Exclusive,
    LinearVelocity,
    NormalisedVector3d,
    NotIn,
    RequiredWhen,
    RotationalVelocity,
    Vector3d,
    When,
)
from . import fsi_schemas as fsi_sch
from .union_tags import literal_of

wall_kind_types = Literal["slip", "no slip", "wall function", "most"]

sliding_area_types = Literal["conservative", "free stream"]
sliding_flux_types = Literal["segment", "parent"]

# How roughness feeds the compressible wall function's thermal law of the wall
roughness_thermal_model_types = Literal["owen thomson", "reynolds analogy"]

# Farfield kinds
farfield_kind_types = Literal["riemann", "pressure", "supersonic", "preconditioned"]

overset_interpolation_methods = Literal[
    "nearest cell",
    "inverse distance",
    "green-gauss",
    "least-squares",
]


class BoundaryConditionBase(BaseModel):
    """UI base model for boundary conditions.

    Exposes only the common required fields shared by every boundary condition
    variant — ``zones`` and ``type``.  The UI renders these fields first when the
    user adds a new boundary condition, before a concrete type is selected.

    Once the user chooses a ``type`` value, use
    :func:`~.ui_utils.get_union_variants` on the ``BoundaryCondition`` union
    to resolve the concrete model class and render its full field set.
    """

    model_config = ConfigDict(extra="forbid")
    type: "BoundaryConditionType" = zCFD_Field(
        ...,
        title="Type",
        description="Boundary condition applied on these zones; it determines which further settings the block takes.",
        options={
            "symmetry": "A mirror plane: no flow crosses it, and the solution is reflected across it.",
            "farfield": "An open boundary held at a reference condition, which detects locally whether the flow enters or leaves.",
            "wall": "A solid surface; its 'kind' selects the wall model.",
            "inflow": "An inlet whose state is set from pressure ratios, a mass flow or a reference condition.",
            "outflow": "An outlet whose state is set from a pressure ratio, a mass flow or a radial pressure gradient.",
            "interface": "A boundary joined to another part of the domain: an overset boundary, a sliding interface or one side of a periodic pair.",
        },
    )
    zones: List[int] = zCFD_Field(
        ...,
        title="Zones",
        description="Mesh zone ids this boundary condition applies to.",
    )


class SpongeLayerDampingModel(BaseModel):
    """Sponge layer damping configuration."""

    model_config = ConfigDict(extra="forbid")

    distance: float = zCFD_Field(
        ...,
        title="Distance",
        description="Depth of the sponge layer, measured into the domain from the farfield boundaries.",
        units="m",
    )
    damping_factor: float = zCFD_Field(
        ...,
        title="Damping Factor",
        description="Peak damping rate, applied at the farfield boundary and ramped to zero at 'distance' along a cosine profile.",
        units="1/s",
    )
    condition: str = zCFD_Field(
        ...,
        title="Condition",
        description="Reference condition the solution is damped towards inside the layer.",
    )


RoughnessMap = case_file("wall roughness", point_array="Roughness")

WallScalarMap = case_file("wall temperature or heat flux")


class RoughnessModel(BaseModel):
    """Wall roughness model configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["height", "length"] = zCFD_Field(
        ...,
        title="Type",
        description="Which roughness measure the value gives.",
        options={
            "height": "The value is the equivalent sand-grain height of the roughness elements.",
            "length": "The value is the aerodynamic roughness length, converted internally to a sand-grain height of length / 0.03.",
        },
    )

    scalar: Annotated[Optional[float], Exclusive("field", required=True)] = zCFD_Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        title="Scalar",
        description="Uniform roughness over the wall, as the measure 'type' names.",
        units="m",
    )
    field: Optional[RoughnessMap] = zCFD_Field(
        default=None,
        title="Field",
        description="VTP file with a point array named 'Roughness'; each wall face takes the nearest point's value.",
    )

    thermal_model: ConditionalDefault[
        roughness_thermal_model_types,
        AnyOf(
            When({"<parent>.kind": "wall function", "<parent>.temperature": SET}),
            When({"<parent>.kind": "wall function", "<parent>.heat_flux": SET}),
        ),
    ] = zCFD_Field(
        default="owen thomson",
        title="Thermal Model",
        description="How roughness affects wall heat transfer under the compressible wall function.",
        options={
            "owen thomson": "Adds a thermal sublayer resistance between the surface and the roughness crest, so the Stanton number rises less than the skin friction.",
            "reynolds analogy": "Omits that resistance, so the Stanton number rises in proportion to the skin friction.",
        },
        advanced=True,
    )


class WallScalarField(BaseModel):
    """Wall value read per face from a VTP file."""

    model_config = ConfigDict(extra="forbid")

    field: WallScalarMap = zCFD_Field(
        ...,
        title="Field",
        description="VTP file with a point array named after the quantity ('Temperature' or 'HeatFlux'); each face takes the nearest point's value",
    )


def _wall_scalar_tag(v):
    return "map" if isinstance(v, (dict, WallScalarField)) else "number"


WallScalar = Annotated[
    Union[
        Annotated[float, Tag("number")],
        Annotated[WallScalarField, Tag("map")],
    ],
    Discriminator(_wall_scalar_tag),
]


WallTemperature = Annotated[
    Union[
        Annotated[float, Field(gt=0.0), Tag("number")],
        Annotated[WallScalarField, Tag("map")],
    ],
    Discriminator(_wall_scalar_tag),
]


def _reject_wall_scalar_shapes(v, key):
    if isinstance(v, dict) and "scalar" in v:
        raise ValueError(
            f"{{'scalar': {v['scalar']}}} is the v1 form: write the number {v['scalar']} directly"
        )
    if isinstance(v, str):
        if v == "adiabatic":
            raise ValueError(f"adiabatic is the default: delete the '{key}' key")
        raise ValueError(
            f"'{v}' is not a {key}: give a number or {{'field': '<map>.vtp'}}"
        )
    return v


class SymmetryBoundary(BoundaryConditionBase):
    """Symmetry boundary condition."""

    type: Literal["symmetry"] = "symmetry"


class MostWallOptions(BaseModel):
    """Monin-Obukhov surface-layer wall (kind: 'most') options.

    An atmospheric rough-wall model: the wall stress and surface heat flux come
    from similarity theory on a roughness length, not from resolving the wall.
    """

    model_config = ConfigDict(extra="forbid")

    roughness_length: float = zCFD_Field(
        ...,
        gt=0.0,
        title="Roughness Length",
        description="Aerodynamic roughness length z0",
        units="m",
    )
    heat_roughness_length: Optional[float] = zCFD_Field(
        default=None,
        gt=0.0,
        title="Heat Roughness Length",
        description="Thermal roughness length z0h; defaults to roughness_length",
        units="m",
        advanced=True,
    )
    surface_potential_temperature: Optional[float] = zCFD_Field(
        default=None,
        gt=0.0,
        title="Surface Potential Temperature",
        description="Prescribed surface theta_s; unset leaves the wall thermally passive",
        units="K",
        advanced=True,
    )
    surface_cooling_rate: Optional[float] = zCFD_Field(
        default=None,
        title="Surface Cooling Rate",
        description="d(theta_s)/dt applied to surface_potential_temperature (GABLS1 uses -0.25)",
        units="K/h",
        advanced=True,
    )
    stable_form: Literal["hogstrom", "beljaars-holtslag"] = zCFD_Field(
        default="hogstrom",
        title="Stable Form",
        description="Stability function family for stable stratification",
        options={
            "hogstrom": "Linear functions, phi = 1 + beta zeta, with coefficients 'beta m' and 'beta h' (Högström 1988); the surface flux collapses as stability grows.",
            "beljaars-holtslag": "Beljaars and Holtslag (1991) functions, which keep a finite surface flux under strong stability.",
        },
        advanced=True,
    )
    von_karman: float = zCFD_Field(
        default=0.4,
        gt=0.0,
        title="Von Karman",
        description="Von Karman constant (atmospheric convention 0.4)",
        advanced=True,
    )
    beta_m: float = zCFD_Field(
        default=4.8,
        gt=0.0,
        title="Beta M",
        description="Stable momentum coefficient, phi_m = 1 + beta_m zeta",
        advanced=True,
    )
    beta_h: float = zCFD_Field(
        default=7.8,
        gt=0.0,
        title="Beta H",
        description="Stable heat coefficient, phi_h = 1 + beta_h zeta",
        advanced=True,
    )


_WALL_FUNCTION = When({"<self>.kind": "wall function"})


class WallBoundary(BoundaryConditionBase):
    """Wall boundary condition."""

    type: Literal["wall"] = "wall"
    kind: wall_kind_types = zCFD_Field(
        default="no slip",
        title="Kind",
        description="Wall model applied on these zones.",
        options={
            "slip": "Zero velocity normal to the wall and no shear stress along it.",
            "no slip": "Applies the no-slip condition directly at the wall.",
            "wall function": "Takes the wall stress from a law of the wall instead of resolving the near-wall layer: Spalding's law with SST, the Allmaras fit with SA-neg.",
            "most": "Takes the wall stress and surface heat flux from Monin-Obukhov similarity theory on a roughness length.",
        },
    )

    most: ConditionalRequired[MostWallOptions, When({"<self>.kind": "most"})] = (
        zCFD_Field(
            default=None,
            title="Most",
            description="Monin-Obukhov surface-layer settings.",
            advanced=True,
        )
    )

    temperature: Annotated[Optional[WallTemperature], Exclusive("heat_flux")] = (
        zCFD_Field(
            default=None,
            title="Wall Temperature",
            description="Wall temperature, as a number or a {'field': '<map>.vtp'} map; unset, the wall is adiabatic.",
            units="K",
            advanced=True,
        )
    )
    heat_flux: ConditionalOptional[WallScalar, _WALL_FUNCTION] = zCFD_Field(
        default=None,
        title="Heat Flux",
        description="Wall heat flux, positive into the fluid, as a number or a {'field': '<map>.vtp'} map.",
        units="W/m^2",
        advanced=True,
    )
    compressible_wall_function: ConditionalChoice[bool, {True: _WALL_FUNCTION}] = (
        zCFD_Field(
            default=False,
            title="Compressible Wall Function",
            description="Applies the compressible law of the wall on an adiabatic wall-function wall; a 'temperature' or a 'heat flux' selects it anyway.",
            advanced=True,
        )
    )

    @field_validator("fsi", mode="before")
    @classmethod
    def _fill_default_fsi_type(cls, v):
        return fsi_sch.default_fsi_type(v)

    @field_validator("temperature", mode="before")
    @classmethod
    def temperature_shape(cls, v):
        return _reject_wall_scalar_shapes(v, "temperature")

    @field_validator("heat_flux", mode="before")
    @classmethod
    def heat_flux_shape(cls, v):
        return _reject_wall_scalar_shapes(v, "heat flux")

    # Wall velocity, for moving walls.
    velocity: Optional[
        Annotated[
            Union[LinearVelocity, RotationalVelocity], Field(discriminator="type")
        ]
    ] = zCFD_Field(
        default=None,
        title="Wall Velocity",
        description="Velocity of the wall surface, for surfaces in relative motion such as a moving belt or a rotating wheel.",
        advanced=True,
    )

    # Roughness parameters
    roughness: Optional[RoughnessModel] = zCFD_Field(
        default=None,
        title="Roughness",
        description="Surface roughness; unset, the wall is smooth.",
        advanced=True,
    )

    # FSI parameters (compatible with any wall kind)
    fsi: Optional[fsi_sch.FSIModel] = zCFD_Field(
        default=None,
        title="FSI",
        description="Fluid-structure interaction model that couples this wall's motion to a structural solve.",
        advanced=True,
    )

    @property
    def has_thermal_state(self) -> bool:
        return self.temperature is not None or self.heat_flux is not None

    @property
    def uses_compressible_wall_function(self) -> bool:
        """A thermal state implies the compressible law; the flag asks for it
        on its own, leaving the wall adiabatic."""
        return self.kind == "wall function" and (
            self.has_thermal_state or self.compressible_wall_function
        )


class OversetConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: List[str] = zCFD_Field(
        ...,
        title="Model",
        description="Model(s) whose mesh provides donor data at this interface.",
    )

    @field_validator("model", mode="before")
    @classmethod
    def normalise_model_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class OversetInterface(BoundaryConditionBase):
    """Overset interface: this mesh overlaps another and takes its fringe values from it."""

    type: Literal["interface"] = "interface"
    kind: Literal["overset"] = zCFD_Field(
        default="overset",
        title="Kind",
        description="How this interface joins its two sides.",
        options={
            "overset": "This mesh overlaps another and takes the values in its fringe from the mesh beneath it.",
        },
    )
    connection: Annotated[
        Optional[OversetConnection],
        Connects(boundary=False, other_model=True),
    ] = zCFD_Field(
        default=None,
        title="Connection",
        description=(
            "Model(s) whose mesh provides donor data at this interface. Without "
            "it, the solver maps against every preceding model."
        ),
    )
    interpolation_method: overset_interpolation_methods = zCFD_Field(
        default="least-squares",
        title="Interpolation Method",
        description="How the donor value is built once the containing donor cell is known.",
        options={
            "nearest cell": "Takes the containing donor cell's value directly: a piecewise-constant, first-order transfer.",
            "inverse distance": "A distance-weighted average over the containing donor cell and its neighbours.",
            "green-gauss": "A Green-Gauss linear reconstruction about the donor cell centre, evaluated at the receptor point.",
            "least-squares": "A least-squares linear reconstruction about the donor cell centre, evaluated at the receptor point.",
        },
    )
    additional_active_layers: int = zCFD_Field(
        default=0,
        ge=0,
        title="Additional Active Layers",
        description="Number of additional active cell layers near the blanking boundary",
        advanced=True,
    )
    blank_enclosed_cells: bool = zCFD_Field(
        default=True,
        title="Blank Enclosed Cells",
        description=(
            "Also blank the background cells the hole encloses but no overset cell "
            "covers: the ones inside this mesh's own solid bodies"
        ),
    )
    create_overset_halos: Annotated[bool, ModelWide()] = zCFD_Field(
        default=True,
        title="Create Overset Halos",
        description=(
            "Search for overset donors at the halo cell centre rather than at the "
            "boundary face centre, on the background-to-overset direction. The halo "
            "centre is the interior cell centre mirrored across the face, and that is "
            "where the solver uses the value: the interface flux extrapolates the "
            "halo from it and the boundary gradient averages it with the interior "
            "cell. Sampling at the face centre instead puts the value half a cell "
            "from where it is used; on the eulervortex static overset case that "
            "leaves a 416 Pa pressure overshoot at the interface, against 4 Pa with "
            "halo centres -- what the conformal mesh gives. A uniform free stream "
            "cannot show the difference. The solver applies one value to the whole "
            "model, so every overset interface on a model gives the same one."
        ),
    )
    moving_mesh_halo_layers: int = zCFD_Field(
        default=0,
        ge=0,
        title="Moving Mesh Halo Layers",
        description=(
            "Halo fringe layers around the hole when the overset mesh moves "
            "(0 = treat every blanked cell as a halo; N > 0 is sufficient when "
            "the hole is re-cut every real time step)"
        ),
        advanced=True,
    )


class SlidingConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = zCFD_Field(
        ..., title="Model", description="Model on the other side of this interface."
    )
    boundary: str = zCFD_Field(
        ...,
        title="Boundary",
        description="Boundary on that model that forms the other side of this interface.",
    )


class SlidingInterface(BoundaryConditionBase):
    """Sliding-plane interface between two models that abut on a surface of revolution."""

    type: Literal["interface"] = "interface"
    kind: Literal["sliding"] = zCFD_Field(
        default="sliding",
        title="Kind",
        description="How this interface joins its two sides.",
        options={
            "sliding": "Two meshes meet on a shared surface of revolution with non-matching faces and exchange a conservative flux across it.",
        },
    )
    connection: Annotated[
        SlidingConnection,
        Connects(boundary=True, other_model=True, reciprocal=True),
    ] = zCFD_Field(
        ...,
        title="Connection",
        description="Model and boundary on the other side of this interface.",
    )
    area: sliding_area_types = zCFD_Field(
        default="conservative",
        title="Area",
        description=(
            "Which area vector each side's segments use. The two options are "
            "identical on a planar interface and for matching meshes."
        ),
        options={
            "conservative": "Both sides use the mean of the two parent area maps, so the interface conserves to round-off; the per-face closure error is first order in cell width.",
            "free stream": "Each side uses its own parent area map, so closure and free-stream preservation are exact; the conservation deficit is second order.",
        },
    )
    flux: sliding_flux_types = zCFD_Field(
        default="segment",
        title="Flux",
        description="Where the interface's numerical flux is evaluated.",
        options={
            "segment": "One flux per intersection polygon, applied with a shared area vector to the cell on each side, so the two sides cancel exactly and the interface conserves.",
            "parent": "One flux per parent face, against the area-weighted average of that face's donors; quieter as cells pass one another, but not conservative.",
        },
    )
    segment_capacity_factor: int = zCFD_Field(
        default=8,
        ge=1,
        title="Segment Capacity Factor",
        description=(
            "Segment storage reserved per interface face. The supermesh is "
            "rebuilt every real time step and its segment count varies with "
            "the rotation angle, so storage is reserved once at this multiple "
            "of the local interface face count. Setup sweeps a full revolution "
            "and aborts with the factor actually needed if this is too small."
        ),
        advanced=True,
    )
    sliver_tolerance: float = zCFD_Field(
        default=1.0e-10,
        ge=0.0,
        title="Sliver Tolerance",
        description=(
            "Overlaps smaller than this fraction of the smaller parent face "
            "are discarded. Slivers carry no flux worth having but do cost a "
            "segment slot each."
        ),
        advanced=True,
    )
    closure_tolerance: float = zCFD_Field(
        default=1.0e-2,
        ge=0.0,
        title="Closure Tolerance",
        description=(
            "Largest fraction of a face's area that may be left uncovered by "
            "its segments before setup aborts. A real gap between the two "
            "sides -- mismatched radii, a partial annulus, a wrong axis -- "
            "shows up here, and would otherwise be a silent leak."
        ),
        advanced=True,
    )


signed_axis_types = Literal["+x", "-x", "+y", "-y", "+z", "-z"]


class TargetAxes(BaseModel):
    """The body axes a target's angles and forces are measured in."""

    model_config = ConfigDict(extra="forbid")

    forward: signed_axis_types = zCFD_Field(
        default="+x",
        title="Forward",
        description="Body axis pointing forward, into the oncoming flow.",
    )
    up: signed_axis_types = zCFD_Field(
        default="+z",
        title="Up",
        description="Body axis pointing up. Lift is positive towards it.",
    )

    @field_validator("up")
    @classmethod
    def _at_right_angles(cls, up, info):
        forward = info.data.get("forward")
        if forward is not None and forward[1] == up[1]:
            raise ValueError(
                f"'up' must be at right angles to 'forward': '{forward}' and "
                f"'{up}' both lie along {up[1]}."
            )
        return up


class TargetAdvanced(BaseModel):
    """Tuning of the law that turns the angle of attack."""

    model_config = ConfigDict(extra="forbid")

    max_step: Optional[float] = zCFD_Field(
        default=None,
        gt=0,
        le=6,
        title="Max Step",
        description=(
            "Largest change of the angle of attack in one step, in degrees, up "
            "to 6: a larger jump can make the flow solve diverge. Unset, it is 4."
        ),
    )
    relaxation: Optional[float] = zCFD_Field(
        default=None,
        gt=0,
        le=2,
        title="Relaxation",
        description=(
            "Fraction of the computed step that is taken, up to 2. Unset, it is 1."
        ),
    )
    average_over: Optional[int] = zCFD_Field(
        default=None,
        ge=1,
        title="Average Over",
        description=(
            "Number of report rows averaged into each measurement of the "
            "coefficient. Unset, it is 10."
        ),
    )
    max_wait: Optional[int] = zCFD_Field(
        default=None,
        gt=0,
        title="Max Wait",
        description=(
            "Cycles to wait after a step for the coefficient to settle, after "
            "which the next step is taken regardless. Unset, it is 1500."
        ),
    )
    slope: Optional[float] = zCFD_Field(
        default=None,
        title="Slope",
        description=(
            "First guess of the change of the coefficient per degree of angle "
            "of attack. Unset, it is 0.11 for CL, the thin-aerofoil slope of "
            "2π per radian; for CD a first step of half the max step is "
            "taken instead."
        ),
    )


class AlphaTarget(BaseModel):
    """A lift or drag coefficient to reach by turning the angle of attack."""

    model_config = ConfigDict(extra="forbid")

    cl: Annotated[Optional[float], Exclusive("cd", required=True)] = zCFD_Field(
        default=None,
        title="CL",
        description="Lift coefficient to trim to.",
    )
    cd: Optional[float] = zCFD_Field(
        default=None,
        title="CD",
        description="Drag coefficient to trim to.",
    )
    hold: Optional[int] = zCFD_Field(
        default=None,
        gt=0,
        title="Hold",
        description=(
            "Cycles to run on once the target is met, after which the run "
            "stops. Unset, the run ends in the ordinary way."
        ),
    )
    tolerance: Optional[float] = zCFD_Field(
        default=None,
        gt=0,
        title="Tolerance",
        description=(
            "How close the coefficient must come to its target to count as met. "
            "Unset, it is 0.001 for CL and 0.0001 for CD."
        ),
    )
    of: Optional[str] = zCFD_Field(
        default=None,
        title="Force report",
        description=(
            "Force report the coefficient is read from. Needed only when "
            "there is more than one force report."
        ),
    )
    advanced: Optional[TargetAdvanced] = zCFD_Field(
        default=None,
        title="Advanced",
        description=(
            "Tuning of how the angle is stepped and how the coefficient is "
            "measured. Unset, every setting takes its default."
        ),
        advanced=True,
    )


class FarfieldTarget(BaseModel):
    """A force the farfield is trimmed to reach."""

    model_config = ConfigDict(extra="forbid")

    alpha: Optional[AlphaTarget] = zCFD_Field(
        default=None,
        title="Alpha",
        description=(
            "Turn the angle of attack, starting from the condition's own "
            "velocity, until the coefficient is met."
        ),
    )
    axes: TargetAxes = zCFD_Field(
        default_factory=TargetAxes,
        title="Axes",
        description=(
            "Body axes the angle of attack and the forces are measured in. "
            "Alpha turns from forward towards up; drag lies along the flow "
            "and lift at right angles to it, towards up."
        ),
    )


_TARGET_NEEDS_STEADY = Advises(
    When({"solver.time_settings.type": NotIn(UNSET, "steady")}),
    "Farfield '{owner}' has a 'target', but this run is unsteady: a target is "
    "trimmed only in a steady compressible run, so it is ignored and the run "
    "keeps the condition's own angle.",
)

_TARGET_NEEDS_COMPRESSIBLE = Advises(
    When({"solver.solver_settings.type": ("incompressible",)}),
    "Farfield '{owner}' has a 'target', but this run is incompressible: a target "
    "is trimmed only in a steady compressible run, so it is ignored and the run "
    "keeps the condition's own angle.",
)


class FarfieldBoundary(BoundaryConditionBase):
    """Farfield boundary condition."""

    type: ConditionalChoice[  # type: ignore[valid-type]
        Literal["farfield"],
        {
            "farfield": (
                When({"solver.solver_settings.type": ("compressible",)}),
                "an incompressible domain is bounded with inflow, outflow, symmetry or wall",
            )
        },
    ] = "farfield"
    kind: farfield_kind_types = zCFD_Field(
        default="riemann",
        title="Kind",
        description="How the farfield state is imposed.",
        options={
            "riemann": "Uses Riemann invariants for a non-reflecting inflow with a pressure outflow.",
            "pressure": "Uses a pressure boundary condition for both inflow (downstream pressure) and outflow (user-specified pressure).",
            "supersonic": "Uses upstream conditions for both inflow and outflow.",
            "preconditioned": "Imposes the reference condition as a supersonic inflow on every face, whatever the local flow direction.",
        },
    )
    condition: str = zCFD_Field(
        ...,
        title="Condition",
        description="Reference condition that sets the farfield state.",
    )
    sponge_layer_damping: Optional[SpongeLayerDampingModel] = zCFD_Field(
        default=None,
        title="Sponge Layer Damping",
        description="Sponge layer that damps the solution towards a reference condition near the farfield boundaries.",
        advanced=True,
    )
    target: Annotated[
        Optional[FarfieldTarget], _TARGET_NEEDS_STEADY, _TARGET_NEEDS_COMPRESSIBLE
    ] = zCFD_Field(
        default=None,
        title="Target",
        description=(
            "A lift or drag coefficient the farfield is trimmed to reach, by "
            "turning its angle of attack."
        ),
    )


_WITH_REFERENCE = When({"<self>.reference": SET})
_INCOMPRESSIBLE = When({"solver.solver_settings.type": ("incompressible",)})

RatioToReference = ConditionalOptional[float, _WITH_REFERENCE]


MassFlowRate = Annotated[
    Optional[float],
    Exclusive("mass_flow_ratio"),
    RequiredWhen(
        When(
            {
                "solver.solver_settings.type": ("compressible",),
                "<self>.mass_flow_ratio": UNSET,
            }
        )
    ),
]


class InflowBoundaryBase(BoundaryConditionBase):
    """Base class for inflow boundary conditions."""

    type: Literal["inflow"] = "inflow"

    reference: Optional[str] = zCFD_Field(
        default=None,
        title="Reference",
        description="Reference condition the inflow state and its ratios are taken from.",
    )

    total_temperature_ratio: RatioToReference = zCFD_Field(
        default=None,
        title="Total Temperature Ratio",
        description="Inlet total temperature as a ratio of the reference static temperature.",
        advanced=True,
    )


class PressureInflowBoundary(InflowBoundaryBase):
    """Pressure-based inflow boundary condition."""

    kind: Literal["pressure"] = zCFD_Field(
        default="pressure",
        title="Kind",
        description="How the inflow state is set.",
        options={
            "pressure": "Velocity and density are derived from a total pressure ratio and a total temperature ratio relative to 'reference'.",
        },
    )

    static_pressure_ratio: RatioToReference = zCFD_Field(
        default=None,
        title="Static Pressure Ratio",
        description="Inlet static pressure as a ratio of the reference static pressure.",
        advanced=True,
    )
    total_pressure_ratio: float = zCFD_Field(
        default=1.0,
        title="Total Pressure Ratio",
        description="Inlet total pressure as a ratio of the reference static pressure.",
    )
    total_temperature_ratio: ConditionalDefault[float, _WITH_REFERENCE] = zCFD_Field(
        default=1.0,
        title="Total Temperature Ratio",
        description="Inlet total temperature as a ratio of the reference static temperature.",
        advanced=True,
    )

    direction: Optional[NormalisedVector3d] = zCFD_Field(
        default=None,
        title="Direction",
        description="Direction the flow enters along; unset, it enters normal to the boundary.",
        advanced=True,
    )
    mach: Optional[float] = zCFD_Field(
        default=None,
        gt=0,
        title="Mach",
        description="Inlet Mach number; the inlet static pressure then follows from the total pressure ratio and this Mach number rather than from the interior flow.",
        advanced=True,
    )


class DefaultInflowBoundary(InflowBoundaryBase):
    """Condition-driven inflow: the inflow state (velocity, temperature,
    turbulence) is taken directly from the named reference condition.

    The incompressible solver's velocity inlet; the v1 schema called this
    kind "default"."""

    kind: ConditionalChoice[
        Literal["default"],
        {
            "default": (
                _INCOMPRESSIBLE,
                "a compressible inflow is kind 'pressure' or 'massflow'",
            )
        },
    ] = zCFD_Field(
        default="default",
        title="Kind",
        description="How the inflow state is set.",
        options={
            "default": "Incompressible solver only: the inflow state is taken directly from the reference condition.",
        },
    )


class MassFlowInflowBoundary(InflowBoundaryBase):
    """Mass flow-based inflow boundary condition."""

    kind: Literal["massflow"] = zCFD_Field(
        default="massflow",
        title="Kind",
        description="How the inflow state is set.",
        options={
            "massflow": "The inlet velocity is set from a mass flow rate or a mass flow ratio.",
        },
    )

    mass_flow_rate: MassFlowRate = zCFD_Field(
        default=None,
        gt=0,
        title="Mass Flow Rate",
        description="Mass flow rate through the inlet zones, as a magnitude: the inflow or outflow type sets the direction. The mesh must be in metres.",
        units="kg/s",
    )

    mass_flow_ratio: RatioToReference = zCFD_Field(
        default=None,
        gt=0,
        title="Mass Flow Ratio",
        description="Inlet mass flow divided by the reference mass flux, rho_in U_in A_in / (rho_ref U_ref), with A_in in mesh units; a magnitude, since the inflow or outflow type sets the direction.",
        advanced=True,
    )
    total_temperature_ratio: Annotated[
        RatioToReference,
        RequiredWhen(When({"solver.solver_settings.type": ("compressible",)})),
    ] = zCFD_Field(
        default=None,
        title="Total Temperature Ratio",
        description="Inlet total temperature as a ratio of the reference static temperature.",
        advanced=True,
    )

    direction: Optional[NormalisedVector3d] = zCFD_Field(
        default=None,
        title="Direction",
        description="Direction the flow enters along; unset, it enters normal to the boundary.",
        advanced=True,
    )


class OutflowBoundaryBase(BoundaryConditionBase):
    """Base class for outflow boundary conditions."""

    type: Literal["outflow"] = "outflow"

    reference: Optional[str] = zCFD_Field(
        default=None,
        title="Reference",
        description="Reference condition the outflow ratios are taken relative to.",
    )


class PressureOutflowBoundary(OutflowBoundaryBase):
    """Pressure-based outflow boundary condition."""

    kind: Literal["pressure"] = zCFD_Field(
        default="pressure",
        title="Kind",
        description="How the outflow state is set.",
        options={
            "pressure": "The exit pressure is set from a static pressure ratio relative to 'reference'.",
        },
    )

    static_pressure_ratio: float = zCFD_Field(
        default=1.0,
        title="Static Pressure Ratio",
        description="Exit static pressure as a ratio of the reference static pressure.",
        advanced=True,
    )


class MassFlowOutflowBoundary(OutflowBoundaryBase):
    """Mass flow-based outflow boundary condition."""

    kind: Literal["massflow"] = zCFD_Field(
        default="massflow",
        title="Kind",
        description="How the outflow state is set.",
        options={
            "massflow": "Scales the local velocity so the integrated mass flow across all zones matches the requested value.",
        },
    )

    mass_flow_rate: MassFlowRate = zCFD_Field(
        default=None,
        gt=0,
        title="Mass Flow Rate",
        description="Mass flow rate through the outlet zones, as a magnitude: the inflow or outflow type sets the direction. The mesh must be in metres.",
        units="kg/s",
    )

    mass_flow_ratio: RatioToReference = zCFD_Field(
        default=None,
        gt=0,
        title="Mass Flow Ratio",
        description="Outlet mass flow divided by the reference mass flux, rho_out U_out A_out / (rho_ref U_ref), with A_out in mesh units; a magnitude, since the inflow or outflow type sets the direction.",
        advanced=True,
    )


class RadialPressureGradientOutflowBoundary(OutflowBoundaryBase):
    """Radial pressure gradient outflow boundary condition."""

    kind: Literal["radial pressure gradient"] = zCFD_Field(
        default="radial pressure gradient",
        title="Kind",
        description="How the outflow state is set.",
        options={
            "radial pressure gradient": "As 'pressure', but the static pressure ratio holds at 'reference radius' rather than uniformly.",
        },
    )

    reference_radius: float = zCFD_Field(
        ...,
        title="Reference Radius",
        description="Radius at which the static pressure ratio is defined",
        units="m",
    )

    static_pressure_ratio: float = zCFD_Field(
        default=1.0,
        title="Static Pressure Ratio",
        description="Exit static pressure at 'reference radius' as a ratio of the reference static pressure.",
        advanced=True,
    )


def _format_vector(v) -> str:
    return "(" + ", ".join(f"{float(x):g}" for x in v) + ")"


def _unit(v):
    length = math.sqrt(sum(float(x) * float(x) for x in v))
    return [float(x) / length for x in v]


def _close(a, b, tol=1.0e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _not_inverse(own: str, other: str, detail: str) -> str:
    return (
        f"the transforms on '{own}' and '{other}' are not inverses of each "
        f"other: {detail}"
    )


class RotatedPeriodicTransform(BaseModel):
    """Rotated periodic transformation parameters."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rotated"] = zCFD_Field(
        default="rotated",
        title="Type",
        description="How this boundary maps onto its partner.",
        options={
            "rotated": "A rotation about an axis maps this boundary onto its partner.",
        },
    )
    theta: float = zCFD_Field(
        ...,
        title="Theta",
        description="Angle of the rotation that maps this boundary onto its partner.",
        units="rad",
    )
    axis: NormalisedVector3d = zCFD_Field(
        ..., title="Axis", description="Direction of the rotation axis."
    )
    origin: Vector3d = zCFD_Field(
        ..., title="Origin", description="A point on the rotation axis.", units="m"
    )

    def inverse_mismatch(self, other, own: str, other_name: str) -> Optional[str]:
        a, b = _unit(self.axis), _unit(other.axis)
        alignment = sum(x * y for x, y in zip(a, b))
        offset = [float(q) - float(p) for p, q in zip(self.origin, other.origin)]
        across = [
            offset[1] * a[2] - offset[2] * a[1],
            offset[2] * a[0] - offset[0] * a[2],
            offset[0] * a[1] - offset[1] * a[0],
        ]
        scale = max(
            1.0,
            math.sqrt(sum(float(x) ** 2 for x in self.origin)),
            math.sqrt(sum(float(x) ** 2 for x in other.origin)),
        )
        same_line = (
            _close(abs(alignment), 1.0)
            and math.sqrt(sum(x * x for x in across)) <= 1.0e-6 * scale
        )
        sign = 1.0 if alignment > 0 else -1.0
        if same_line and _close(other.theta, -sign * self.theta):
            return None
        return _not_inverse(
            own,
            other_name,
            f"a rotation of {self.theta:g} rad about {_format_vector(self.axis)} "
            f"through {_format_vector(self.origin)} on '{own}' needs "
            f"{0.0 - self.theta:g} rad about the same axis on '{other_name}', which "
            f"gives {other.theta:g} rad about {_format_vector(other.axis)} "
            f"through {_format_vector(other.origin)}",
        )


class LinearPeriodicTransform(BaseModel):
    """Linear periodic transformation parameters."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["linear"] = zCFD_Field(
        default="linear",
        title="Type",
        description="How this boundary maps onto its partner.",
        options={
            "linear": "A translation maps this boundary onto its partner.",
        },
    )
    vector: Vector3d = zCFD_Field(
        ...,
        title="Vector",
        description="Translation that maps this boundary onto its partner.",
        units="m",
    )

    def inverse_mismatch(self, other, own: str, other_name: str) -> Optional[str]:
        if all(_close(float(p), -float(q)) for p, q in zip(self.vector, other.vector)):
            return None
        return _not_inverse(
            own,
            other_name,
            f"a translation of {_format_vector(self.vector)} on '{own}' needs "
            f"{_format_vector([0.0 - float(x) for x in self.vector])} on "
            f"'{other_name}', which gives {_format_vector(other.vector)}",
        )


PeriodicTransform = Annotated[
    Union[RotatedPeriodicTransform, LinearPeriodicTransform],
    Field(discriminator="type"),
]


class PeriodicConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boundary: str = zCFD_Field(
        ...,
        title="Boundary",
        description="Boundary on this model that forms the other side of the periodic pair.",
    )


class PeriodicInterface(BoundaryConditionBase):
    """Periodic interface: this boundary and the one it connects to are one surface."""

    type: Literal["interface"] = "interface"
    kind: Literal["periodic"] = zCFD_Field(
        default="periodic",
        title="Kind",
        description="How this interface joins its two sides.",
        options={
            "periodic": "Two boundaries of the same mesh are treated as one surface, related by a rotation or a translation.",
        },
    )
    connection: Annotated[
        PeriodicConnection,
        Connects(boundary=True, other_model=False, reciprocal=True),
    ] = zCFD_Field(
        ...,
        title="Connection",
        description="Boundary on this model that forms the other side of the periodic pair.",
    )
    transform: PeriodicTransform = zCFD_Field(
        ...,
        title="Transform",
        description="Transformation that maps this boundary onto the one it connects to.",
    )

    def connection_mismatch(self, other, own: str, other_name: str) -> Optional[str]:
        if self.transform.type != other.transform.type:
            return _not_inverse(
                own,
                other_name,
                f"'{own}' is {self.transform.type} and '{other_name}' is "
                f"{other.transform.type}",
            )
        return self.transform.inverse_mismatch(other.transform, own, other_name)


InterfaceBoundary = Annotated[
    Union[OversetInterface, SlidingInterface, PeriodicInterface],
    Field(discriminator="kind"),
]


class InterfaceBoundaryBase(BoundaryConditionBase):
    """UI base model for interface boundaries."""

    type: Literal["interface"] = "interface"
    kind: literal_of(InterfaceBoundary) = zCFD_Field(
        ..., title="Kind", description="Interface kind"
    )


InflowBoundary = Annotated[
    Union[PressureInflowBoundary, MassFlowInflowBoundary, DefaultInflowBoundary],
    Field(discriminator="kind"),
]

OutflowBoundary = Annotated[
    Union[
        PressureOutflowBoundary,
        MassFlowOutflowBoundary,
        RadialPressureGradientOutflowBoundary,
    ],
    Field(discriminator="kind"),
]

BoundaryCondition = Annotated[
    Union[
        WallBoundary,
        FarfieldBoundary,
        SymmetryBoundary,
        Annotated[InflowBoundary, Tag("inflow")],
        Annotated[OutflowBoundary, Tag("outflow")],
        Annotated[InterfaceBoundary, Tag("interface")],
    ],
    Field(discriminator="type"),
]

BoundaryConditionType = literal_of(BoundaryCondition)
BoundaryConditionBase.model_rebuild()
