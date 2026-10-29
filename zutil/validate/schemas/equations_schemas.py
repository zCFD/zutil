from pydantic import BaseModel, ConfigDict, Field, Tag, field_validator
from typing import Literal, Union, Optional, Annotated, List
from zutil.validate.schemas.base_types import (
    SET,
    ConditionalChoice,
    ConditionalOptional,
    When,
    zCFD_Field,
)
from zutil.validate.schemas.fluid_model_schemas import WITH_GRAVITY, WITH_LATITUDE
from zutil.validate.schemas.time_settings_schemas import APPLIES_WITH_PSEUDO_TIME

equation_types = Literal["euler", "viscous", "rans", "les"]


def _lowercase_str(v):
    """BeforeValidator helper: lowercase string values for case-insensitive enum matching."""
    return v.lower() if isinstance(v, str) else v


# Turbulence model types
turbulence_model_types = Literal[
    "sst",
    "sas",
    "sa-neg",
    "sst-transition",
    "spalart-allmaras",
]

# QCR model types
qcr_types = Literal[2000, 2020, 2021]

hybrid_types = Literal["DES", "DDES", "IDDES", "SAS"]

# LES model types
les_model_types = Literal["none", "wale", "deardorff"]


class TurbulenceModel(BaseModel):
    """Turbulence model configuration for RANS equations."""

    model_config = ConfigDict(extra="forbid")

    model: turbulence_model_types = zCFD_Field(
        ...,
        title="Model",
        description="RANS turbulence model (sst = Menter SST k-ω, sas = Scale-Adaptive Simulation, sa-neg = Spalart-Allmaras Negative, sst-transition = γ-Reθ transition)",
    )

    # DES / Hybrid RANS-LES support
    hybrid: Optional[hybrid_types] = zCFD_Field(
        default=None,
        title="Hybrid RANS-LES",
        description="Hybrid RANS-LES approach (DES = Detached Eddy Simulation, DDES = Delayed DES, IDDES = Improved Delayed DES, SAS = Scale-Adaptive Simulation)",
    )

    # Eddy viscosity limiter (SST model)
    limit_mut: Optional[bool] = zCFD_Field(
        default=True,
        title="Limit Eddy Viscosity",
        description="Limit eddy viscosity in SST model",
        advanced=True,
    )

    limit_gradient_k: Optional[float] = zCFD_Field(
        default=None,
        ge=0.01,
        le=1.0,
        title="Gradient Limiter k",
        description="Gradient limiter coefficient for turbulence kinetic energy k",
        advanced=True,
    )

    qcr: Optional[qcr_types] = zCFD_Field(
        default=None,
        title="QCR Variant",
        description="QCR variant year (2000, 2020, or 2021)",
    )

    rotation_correction: Optional[bool] = zCFD_Field(
        default=False,
        title="Rotation Correction",
        description=(
            "Enable the rotation correction. Read by both turbulence models, "
            "in their own forms: SST applies Hellsten's f4 = 1/(1 + Crc*Ri) to "
            "the omega destruction term, and SA adds Crot*min(0, S - Omega) to "
            "production (Dacles-Mariani / Spalart-Shur)."
        ),
    )

    production: Optional[
        Literal[
            "vorticity", "strain-incompressible", "strain-compressible", "kato-launder"
        ]
    ] = zCFD_Field(
        default=None,
        title="Production Term",
        description=(
            "Production term model for SST (vorticity = SST-V, "
            "strain-incompressible = strain-based for incompressible, "
            "strain-compressible = strain-based for compressible, "
            "kato-launder = Kato-Launder). When omitted the solver build "
            "chooses: kato-launder for sst-transition, vorticity otherwise "
            "(the solver's Menter SST defaults) — a schema default here would "
            "silently override the transition build's choice."
        ),
        advanced=True,
    )

    @field_validator("production", mode="before")
    @classmethod
    def normalise_production(cls, v):
        """Accept legacy integer production term values, map to string."""
        if isinstance(v, int):
            mapping = {
                0: "vorticity",
                1: "strain-incompressible",
                2: "strain-compressible",
                3: "kato-launder",
            }
            if v in mapping:
                return mapping[v]
            raise ValueError(
                f"Unknown production term '{v}'. Valid integers: {list(mapping.keys())}"
            )
        return v

    ## Advanced constants
    betastar: Optional[float] = zCFD_Field(
        default=0.09,
        title="β*",
        description="Constant in the dissipation term on the k equation in the SST model",
        advanced=True,
    )
    cdes_kw: Optional[float] = zCFD_Field(
        default=0.78,
        title="C_DES (k-ω)",
        description="The k-omega part of the blended C_DES in SST",
        advanced=True,
    )
    cdes_keps: Optional[float] = zCFD_Field(
        default=0.61,
        title="C_DES (k-ε)",
        description="The k-epsilon part of the blended C_DES in SST",
        advanced=True,
    )
    cd1: Optional[float] = zCFD_Field(
        default=20.0,
        title="C_d1",
        description="Constant in the f_dt blending function in DDES and IDDES",
        advanced=True,
    )
    cd2: Optional[int] = zCFD_Field(
        default=3,
        title="C_d2",
        description="Constant in the f_dt blending function in DDES and IDDES",
        advanced=True,
    )
    cw: Optional[float] = zCFD_Field(
        default=0.15,
        title="C_w (IDDES)",
        description="Constant in the length scale calculation in IDDES",
        advanced=True,
    )
    a1: Optional[float] = zCFD_Field(
        default=0.31,
        title="a₁ (SST)",
        description="Constant in the eddy viscosity limiter in SST",
        advanced=True,
    )
    cdes: Optional[float] = zCFD_Field(
        default=0.65,
        title="C_DES (SA-neg)",
        description="C_DES constant used for SA-neg based hybrid RANS/LES models",
        advanced=True,
    )
    ca1: Optional[float] = zCFD_Field(
        default=2.0,
        title="C_a1 (transition)",
        description="Constant in the production of the gamma equation in the transition model",
        advanced=True,
    )
    ca2: Optional[float] = zCFD_Field(
        default=0.06,
        title="C_a2 (transition)",
        description="Constant in the destruction term for gamma equation in the transition model",
        advanced=True,
    )
    ce1: Optional[float] = zCFD_Field(
        default=1.0,
        title="C_e1 (transition)",
        description="Constant in the production term of the gamma equation in the transition model",
        advanced=True,
    )
    ce2: Optional[float] = zCFD_Field(
        default=50.0,
        title="C_e2 (transition)",
        description="Constant in the destruction term for gamma equation in the transition model",
        advanced=True,
    )
    cthetat: Optional[float] = zCFD_Field(
        default=0.03,
        title="C_θt (transition)",
        description="Constant in the Re-theta transition correlation in the transition model",
        advanced=True,
    )
    sigmagamma: Optional[float] = zCFD_Field(
        default=1.0,
        title="σ_γ (transition)",
        description="Constant in the transition model in the compressible solver",
        advanced=True,
    )
    sigmathetat: Optional[float] = zCFD_Field(
        default=2.0,
        title="σ_θt (transition)",
        description="Constant in the transition model in the compressible solver",
        advanced=True,
    )
    separation_correction: Optional[bool] = zCFD_Field(
        default=True,
        title="Separation Correction",
        description="Activates the separation correction for the transition model",
        advanced=True,
    )

    # ── CFL shield for the hybrid RANS-LES branch (DES/DDES/IDDES only) ──────
    # The shield is inactive unless the run is time accurate, so both fields
    # carry the unsteady clause.
    les_cfl_limit: ConditionalOptional[  # type: ignore[valid-type]
        float, When({"solver.time_settings.type": ("unsteady",)})
    ] = zCFD_Field(
        default=None,
        ge=0.0,
        title="LES CFL Limit",
        description=(
            "Restrict the LES branch of the DES/DDES/IDDES length scale to cells whose "
            "nominal CFL is at or below this limit. 0 (the default) disables the shield, "
            "which is also always inactive unless the run is time accurate"
        ),
        advanced=True,
    )
    les_cfl_width: ConditionalOptional[  # type: ignore[valid-type]
        float, When({"solver.time_settings.type": ("unsteady",)})
    ] = zCFD_Field(
        default=None,
        gt=0.0,
        title="LES CFL Width",
        description=(
            "Width of the blend above les_cfl_limit. The shield ramps from LES at "
            "les_cfl_limit to full RANS at les_cfl_limit plus this width."
        ),
        advanced=True,
    )
    les_turnover_coefficient: ConditionalOptional[  # type: ignore[valid-type]
        float, When({"solver.time_settings.type": ("unsteady",)})
    ] = zCFD_Field(
        default=None,
        ge=0.0,
        title="LES Turnover Coefficient",
        description=(
            "Weight on the eddy turnover contribution to the shielded CFL. The time step "
            "has to resolve both the transport of an eddy across the cell and the eddy's "
            "own turnover, and the shield takes whichever is larger. 0 leaves the transport "
            "only criterion, which cannot see a cell whose speed relative to the mesh is "
            "near zero. Defaults to 1, and has no effect unless les_cfl_limit is set"
        ),
        advanced=True,
    )


class EquationsBase(BaseModel):
    """Base equations model."""

    model_config = ConfigDict(extra="forbid")

    type: equation_types = zCFD_Field(
        ...,
        title="Type",
        description=(
            "The form of the flow equations to solve: inviscid Euler, viscous "
            "Navier–Stokes with no turbulence model, Reynolds-averaged "
            "Navier–Stokes (RANS) or large-eddy simulation (LES)."
        ),
    )

    # Preconditioning modifies the pseudo-time system, so it needs the inner
    # pseudo-time loop to exist (steady/DTS) — under GTS there is nothing to
    # precondition. False (the default) is inert and never a violation.
    precondition: ConditionalOptional[  # type: ignore[valid-type]
        bool, When(APPLIES_WITH_PSEUDO_TIME)
    ] = zCFD_Field(
        default=False,
        title="Precondition",
        description="Enable low mach number preconditioning",
    )


class EulerEquations(EquationsBase):
    """Euler equations for inviscid flow."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["euler"] = "euler"


class ViscousEquations(EquationsBase):
    """Viscous Navier-Stokes equations."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["viscous"] = "viscous"


class RANSEquations(EquationsBase):
    """RANS equations with turbulence modeling."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rans"] = "rans"
    # Turbulence model configuration
    turbulence: TurbulenceModel = zCFD_Field(
        ..., title="Turbulence", description="Turbulence model configuration"
    )


class LESModel(BaseModel):
    """LES subgrid-scale model configuration."""

    model_config = ConfigDict(extra="forbid")

    les: les_model_types = zCFD_Field(
        ...,
        title="LES",
        description=(
            "LES subgrid-scale model (wale = Wall-Adapting Local Eddy-viscosity, "
            "deardorff = 1.5-order prognostic SGS-TKE with a stratification-limited length scale, "
            "none = no SGS model)"
        ),
    )
    wall_damping: Optional[bool] = zCFD_Field(
        default=None,
        title="Wall Damping",
        description="Enable wall damping for LES model",
    )
    relax: float = zCFD_Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        title="Relax",
        description=(
            "Under-relaxation factor for the SGS turbulence equation. Read UNCONDITIONALLY by the "
            "C++ (IncompSolverData.h initKernels, turbulence.relax), so it must always be present "
            "in the emitted dict -- it carried a default of 0.5 in the legacy schema and was lost "
            "when the schemas moved, leaving the deardorff path with no way to supply it."
        ),
    )

    # WALE model constants
    eps: Optional[float] = zCFD_Field(
        default=0.41,
        title="Eps",
        description="von Kármán constant for WALE near-wall scaling",
    )
    cw: Optional[float] = zCFD_Field(
        default=0.325,
        title="Cw",
        description="WALE model constant Cw (subgrid viscosity coefficient)",
    )


class LESEquations(EquationsBase):
    """Large Eddy Simulation equations."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["les"] = "les"

    # LES configuration — users set this as 'turbulence' in the input dict
    turbulence: Optional[LESModel] = zCFD_Field(
        default=None, title="Turbulence", description="LES subgrid model configuration"
    )


# ── Incompressible-specific equation fields ──────────────────────────────────
# These are read by the C++ incompressible solver from parameters.equations().

correction_types = Literal["none", "orthogonal", "minimum", "over_relaxed"]
gradient_limiter_types = Literal["cellmd", "cell"]
energy_variable_types = Literal["temperature", "potential temperature"]


class BaseStateCommon(BaseModel):
    """Hydrostatic base state of a stratified (anelastic) atmosphere.

    Dimensional SI throughout; the solver non-dimensionalises. Height is
    measured along -gravity from the reference condition's location. The
    surface state comes from the named reference condition
    (resolve_base_state_surface fills surface_temperature/surface_pressure
    before the dict reaches the C++).
    """

    model_config = ConfigDict(extra="forbid")

    reference: Optional[str] = zCFD_Field(
        default=None,
        title="Reference",
        description=(
            "Reference condition supplying the surface temperature and pressure "
            "(e.g. 'IC_1'); defaults to the initialisation reference"
        ),
    )
    surface_temperature: Optional[float] = zCFD_Field(
        default=None,
        gt=0.0,
        title="Surface Temperature",
        description="Temperature at the reference location; overrides the reference condition's value",
        units="K",
        advanced=True,
    )
    surface_pressure: Optional[float] = zCFD_Field(
        default=None,
        gt=0.0,
        title="Surface Pressure",
        description="Pressure at the reference location; overrides the reference condition's value",
        units="Pa",
        advanced=True,
    )


class IsothermalBaseState(BaseStateCommon):
    kind: Literal["isothermal"] = zCFD_Field(
        ...,
        title="Kind",
        description="Uniform temperature; theta grows as exp(g z / (cp T0))",
    )


class ConstantLapseRateBaseState(BaseStateCommon):
    kind: Literal["constant lapse rate"] = zCFD_Field(
        ...,
        title="Kind",
        description="Linear temperature profile T = T0 - lapse_rate z",
    )
    lapse_rate: float = zCFD_Field(
        default=0.0065,
        title="Lapse Rate",
        description="-dT/dz; g/cp (about 0.0098) is the neutral dry adiabat",
        units="K/m",
    )


class ConstantBruntVaisalaBaseState(BaseStateCommon):
    kind: Literal["constant brunt vaisala"] = zCFD_Field(
        ...,
        title="Kind",
        description="Uniform stratification, theta = theta0 exp(N^2 z/g)",
    )
    brunt_vaisala_frequency: float = zCFD_Field(
        default=0.01,
        ge=0.0,
        title="Brunt-Vaisala Frequency",
        description="N",
        units="1/s",
    )


class TwoLayerBaseState(BaseStateCommon):
    kind: Literal["two layer"] = zCFD_Field(
        ...,
        title="Kind",
        description=(
            "Neutral mixed layer capped by uniform stable stratification: theta = theta0 "
            "below mixed_layer_depth, theta0 + upper_lapse_rate (z - h) above (GABLS1)"
        ),
    )
    mixed_layer_depth: float = zCFD_Field(
        default=100.0,
        gt=0.0,
        title="Mixed Layer Depth",
        description="Depth h of the neutral mixed layer",
        units="m",
    )
    upper_lapse_rate: float = zCFD_Field(
        default=0.01,
        ge=0.0,
        title="Upper Lapse Rate",
        description="dtheta/dz above the mixed layer",
        units="K/m",
    )


BaseState = Annotated[
    Union[
        Annotated[IsothermalBaseState, Tag("isothermal")],
        Annotated[ConstantLapseRateBaseState, Tag("constant lapse rate")],
        Annotated[ConstantBruntVaisalaBaseState, Tag("constant brunt vaisala")],
        Annotated[TwoLayerBaseState, Tag("two layer")],
    ],
    Field(discriminator="kind"),
]


theta_perturbation_kinds = Literal["inertia gravity wave", "bubble", "random"]


class ThetaPerturbation(BaseModel):
    """Initial potential-temperature perturbation theta'(x, z) added to the base state.

    Dimensional SI. x is the mesh x coordinate and z the height, both measured
    from the reference condition's location. Seeds the initial condition only;
    the base state itself is unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    kind: theta_perturbation_kinds = zCFD_Field(
        ...,
        title="Kind",
        description=(
            "inertia gravity wave: A sin(pi z/H) / (1 + ((x - xc)/a)^2) (Skamarock & Klemp 1994); "
            "bubble: A cos^2(pi r/2) inside the ellipse r = |((x - xc)/rx, (z - zc)/rz)| <= 1; "
            "random: position-hashed +/-A turbulence seed below 'depth' (GABLS1)"
        ),
    )
    amplitude: float = zCFD_Field(
        ...,
        title="Amplitude",
        description="A; negative for a cold bubble",
        units="K",
    )
    centre: List[float] = zCFD_Field(
        default=[0.0, 0.0],
        min_length=2,
        max_length=2,
        title="Centre",
        description="[xc, zc]; zc is used by the bubble only",
        units="m",
    )
    half_width: float = zCFD_Field(
        default=5000.0,
        gt=0.0,
        title="Half Width",
        description="a for the gravity wave, or the bubble x radius rx",
        units="m",
    )
    half_height: float = zCFD_Field(
        default=2000.0,
        gt=0.0,
        title="Half Height",
        description="Bubble z radius rz",
        units="m",
    )
    depth: float = zCFD_Field(
        default=10000.0,
        gt=0.0,
        title="Depth",
        description="H in sin(pi z/H) for the gravity wave",
        units="m",
    )


class VelocityPerturbation(BaseModel):
    """Initial random velocity perturbation for seeding shear-driven turbulence.

    GABLS1-style LES needs it: a theta-only seed decays in a neutral mixed
    layer instead of growing. Position-hashed, so runs are reproducible and
    invariant under repartitioning.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["random"] = zCFD_Field(
        ...,
        title="Kind",
        description="Position-hashed +/-amplitude seed on all three components",
    )
    amplitude: float = zCFD_Field(
        ...,
        gt=0.0,
        title="Amplitude",
        description="Peak perturbation on each velocity component",
        units="m/s",
    )
    depth: float = zCFD_Field(
        default=50.0,
        gt=0.0,
        title="Depth",
        description="Height below which the seed is applied; zero above",
        units="m",
    )


class SpongeLayer(BaseModel):
    """Height-based sponge damping toward a target state near the domain lid,
    so upward-propagating gravity waves are absorbed rather than reflected.
    sigma(z) ramps as sin^2 from start_height to the mesh top."""

    model_config = ConfigDict(extra="forbid")

    sigma_max: float = zCFD_Field(
        ...,
        gt=0.0,
        title="Sigma Max",
        description="Peak damping rate at the domain top",
        units="1/s",
    )
    start_height: float = zCFD_Field(
        ...,
        ge=0.0,
        title="Start Height",
        description="Height where the damping ramp begins",
        units="m",
    )
    relax_theta: bool = zCFD_Field(
        default=True,
        title="Relax Theta",
        description=(
            "Relax potential temperature toward the base state inside the sponge, as well as "
            "velocity. TRUE reproduces the existing behaviour: EnergyEqn.cu applies the sponge to "
            "theta whenever a sponge and a base state are present, which the deck's 'target "
            "velocity' wording does not convey. Set FALSE for a velocity-only sponge -- the upper "
            "boundary then stops being a heat sink, which is what isolates its effect on a "
            "stably-stratified case without also letting gravity waves reflect off the lid."
        ),
    )
    target_velocity: Optional[List[float]] = zCFD_Field(
        default=None,
        min_length=3,
        max_length=3,
        title="Target Velocity",
        description="Velocity the sponge damps toward; unset damps toward rest",
        units="m/s",
    )


_WITH_BASE_STATE = When({"<self>.base_state": SET})


class IncompressibleEquationsOptions(BaseModel):
    """Fields consumed by the C++ incompressible solver."""

    correction: correction_types = zCFD_Field(
        default="over_relaxed",
        title="Correction",
        description="Non-orthogonal correction method",
    )
    correction_limiter: float = zCFD_Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        title="Correction Limiter",
        description="Non-orthogonal correction limiter coefficient",
    )
    gradient_limiter_type: gradient_limiter_types = zCFD_Field(
        default="cellmd",
        title="Gradient Limiter Type",
        description="Gradient limiter type",
    )
    gradient_limiter: float = zCFD_Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        title="Gradient Limiter",
        description="Gradient limiter coefficient",
    )
    max_second_order_non_orthogonal_angle: float = zCFD_Field(
        default=90.0,
        ge=0.0,
        le=90.0,
        title="Max Second Order Non Orthogonal Angle",
        description="Maximum angle for second order non-orthogonal correction",
    )
    central_blend: float = zCFD_Field(
        default=0.8,
        title="Central Blend",
        description="Central differencing blend factor",
    )
    solve_energy: bool = zCFD_Field(
        default=False,
        title="Solve Energy",
        description="Solve the energy (temperature) transport equation",
    )
    energy_variable: energy_variable_types = zCFD_Field(
        default="temperature",
        title="Energy Variable",
        description=(
            "Variable transported by the energy equation: temperature, or potential "
            "temperature theta = T (p0/p)^(R/cp) for atmospheric cases"
        ),
        advanced=True,
    )
    base_state: ConditionalOptional[  # type: ignore[valid-type]
        BaseState,
        When(
            {
                **WITH_GRAVITY,
                "<self>.solve_energy": True,
                "<self>.energy_variable": "potential temperature",
            }
        ),
    ] = zCFD_Field(
        default=None,
        title="Base State",
        description=(
            "Hydrostatic base state for a stratified (anelastic) atmosphere. Its presence "
            "turns stratification on: density becomes the base-state rho(z) and buoyancy is "
            "g (theta - theta_base)/theta_base. Needs a non-zero gravity"
        ),
        advanced=True,
    )
    theta_perturbation: ConditionalOptional[  # type: ignore[valid-type]
        ThetaPerturbation, _WITH_BASE_STATE
    ] = zCFD_Field(
        default=None,
        title="Theta Perturbation",
        description=(
            "Initial potential-temperature perturbation added to the base state "
            "(gravity-wave and bubble benchmarks)"
        ),
        advanced=True,
    )
    buoyancy_diagnostics: ConditionalChoice[  # type: ignore[valid-type]
        bool, {True: _WITH_BASE_STATE}
    ] = zCFD_Field(
        default=False,
        title="Buoyancy Diagnostics",
        description=(
            "Log the buoyancy source against the momentum diagonal every cycle. Four "
            "host syncs per cycle: debugging only"
        ),
        advanced=True,
    )
    velocity_perturbation: Optional[VelocityPerturbation] = zCFD_Field(
        default=None,
        title="Velocity Perturbation",
        description="Initial random velocity seed for shear-driven turbulence (LES spin-up)",
        advanced=True,
    )
    geostrophic_wind: ConditionalOptional[  # type: ignore[valid-type]
        List[float], When(WITH_LATITUDE)
    ] = zCFD_Field(
        default=None,
        min_length=3,
        max_length=3,
        title="Geostrophic Wind",
        description=(
            "Large-scale forcing u_g: the Coriolis source becomes -f k x (u - u_g), "
            "standing in for the synoptic pressure gradient. Needs a non-zero latitude"
        ),
        units="m/s",
        advanced=True,
    )
    sponge_layer: Optional[SpongeLayer] = zCFD_Field(
        default=None,
        title="Sponge Layer",
        description="Height-based wave-absorbing layer below the domain lid. Requires base_state",
        advanced=True,
    )


class IncompViscousEquations(IncompressibleEquationsOptions, ViscousEquations):
    """Incompressible viscous equations with solver-specific parameters."""

    model_config = ConfigDict(extra="forbid")


class IncompRANSEquations(IncompressibleEquationsOptions, RANSEquations):
    """Incompressible RANS equations with solver-specific parameters."""

    model_config = ConfigDict(extra="forbid")


class IncompLESEquations(IncompressibleEquationsOptions, LESEquations):
    """Incompressible LES equations with solver-specific parameters."""

    model_config = ConfigDict(extra="forbid")


Equations = Union[
    EulerEquations,
    ViscousEquations,
    RANSEquations,
    LESEquations,
    IncompViscousEquations,
    IncompRANSEquations,
    IncompLESEquations,
]
