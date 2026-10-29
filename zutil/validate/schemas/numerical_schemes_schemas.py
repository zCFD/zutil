from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing import Union, Literal, Optional
from zutil.validate.schemas.base_types import (
    UNSET,
    ConditionalOptional,
    When,
    zCFD_Field,
)
from zutil.validate.schemas.convergence_schemas import EXPLICIT_SCHEME_NAMES
from zutil.validate.schemas.time_settings_schemas import APPLIES_WITH_PSEUDO_TIME

# Order types for different solvers. "central" is incompressible-only:
# FVCompressibleNumericalSchemeBase narrows to compressible_order_types below.
order_types = Literal["first", "second", "euler_second", "central"]
compressible_order_types = Literal["first", "second", "euler_second"]

# "symmetric ratio" is the 1D function formerly named "barth jespersen";
# that name now selects the real Barth & Jespersen cell-based limiter.
sweby_limiter_types = ("van albada", "van leer", "symmetric ratio", "none")
cell_limiter_types = ("barth jespersen", "michalak gooch", "venkatakrishnan")

limiter_types = Literal[
    "van albada",
    "van leer",
    "symmetric ratio",
    "none",
    "barth jespersen",
    "michalak gooch",
    "venkatakrishnan",
]

# Turbulence limiter types. NOT a reconstruction limiter
turbulence_limiter_types = Literal["zero", "no update"]


def normalise_limiter(v):
    """BeforeValidator: accept the legacy 'vanalbada' spelling."""
    if isinstance(v, str) and v.lower() == "vanalbada":
        return "van albada"
    return v


# Inviscid flux scheme types — lowercase canonical, with BeforeValidator normalisation
inviscid_flux_scheme_types = Literal[
    "roe", "hllc", "roe low-diss", "hllc low-diss", "rusanov"
]

# Precondition types
precondition_types = Literal[True, False, "low speed", "matrix"]

# Roe low-dissipation sensor types
roe_low_dissipation_sensor_types = Literal["FD", "NTS", "DES", "NONE"]

# Roe low-dissipation sensor parameter (field used to compute the sensor)
roe_low_dissipation_sensor_parameter_types = Literal[
    "density", "velocity", "velocity divergence/vorticity"
]


class MultigridConfig(BaseModel):
    """Multigrid configuration for explicit FV schemes."""

    model_config = ConfigDict(extra="forbid")

    levels: int = zCFD_Field(
        ...,
        ge=1,
        le=10,
        title="Levels",
        description="Number of multigrid levels",
        updatable=True,
    )
    cycles: Optional[int] = zCFD_Field(
        default=int(1e8),
        ge=1,
        title="Cycles",
        description="Number of multigrid cycles",
        updatable=True,
    )
    multigrid_ramp: Optional[int] = zCFD_Field(
        default=None,
        ge=0,
        title="Multigrid Ramp",
        description="Solve cycle after which the coarse multigrid levels are used",
        updatable=True,
    )
    prolong_factor: Optional[float] = zCFD_Field(
        default=None,
        title="Prolongation Factor",
        description="Prolongation factor",
        updatable=True,
    )
    prolong_transport_factor: Optional[float] = zCFD_Field(
        default=None,
        title="Prolongation Transport Factor",
        description="Prolongation factor for the turbulence/transport equations",
        updatable=True,
    )


class PreconditionerConfig(BaseModel):
    """Low-Mach preconditioner cut-off.

    Read at src/solver/physics/Reference.cxx:200-206 out of
    ``parameters.numericalScheme()`` — the sub-dict is the *only* way to
    override ``Reference::precondK``, which otherwise takes the Mach-dependent
    default ``M * max(-5M + 3, min(M, 0.5))`` set on the line above the read.
    ``precondK`` is the floor applied to the preconditioning reference
    velocity in ``PreconditionHelper::refVel`` (Precondition.h, LUSGS.h), so
    it only bites once ``equations.precondition`` is on.
    """

    model_config = ConfigDict(extra="forbid")

    minimum_mach_number: float = zCFD_Field(
        ...,
        gt=0.0,
        title="Minimum Mach Number",
        description=(
            "Cut-off Mach number for low-Mach preconditioning: floors the "
            "preconditioning reference velocity so it cannot collapse at a "
            "stagnation point. Overrides the Mach-dependent default."
        ),
        advanced=True,
    )


class FVNumericalSchemeBase(BaseModel):
    """
    Base numerical scheme model for all finite volume methods.
    Contains common fields shared by both compressible and incompressible schemes.
    """

    model_config = ConfigDict(extra="forbid")

    order: order_types = zCFD_Field(
        default="second",
        title="Spatial Order",
        description="Spatial order of accuracy (central is incompressible only)",
        updatable=True,
    )

    @field_validator("order", mode="before")
    @classmethod
    def normalise_order(cls, v):
        """Accept legacy integer order values."""
        if isinstance(v, int):
            mapping = {1: "first", 2: "second"}
            if v in mapping:
                return mapping[v]
            raise ValueError(
                f"Unknown integer order '{v}'. Use 1 (first) or 2 (second)."
            )
        return v

    limiter: limiter_types = zCFD_Field(
        default="van albada",
        title="Slope Limiter",
        description="Slope limiter for higher-order schemes",
    )

    @field_validator("limiter", mode="before")
    @classmethod
    def normalise_limiter_field(cls, v):
        """Accept the legacy 'vanalbada' spelling; canonical is 'van albada'."""
        return normalise_limiter(v)

    limiter_coefficient: float = zCFD_Field(
        default=5.0,
        ge=0.0,
        title="Limiter Coefficient",
        description=(
            "K in the grid-scaled smoothness threshold eps^2 = (K h / reference_length)^3 "
            "used by the 'michalak gooch' and 'venkatakrishnan' cell-based limiters to "
            "deactivate limiting in smooth flow. Larger values treat more of the flow as "
            "smooth, so limit less. "
            "Ignored by the other limiters."
        ),
        advanced=True,
    )

    first_order_cycles: int = zCFD_Field(
        default=0,
        ge=0,
        title="First Order Cycles",
        description="Number of first-order cycles before switching to higher order",
        updatable=True,
    )

    freeze_limiter_cycle: Optional[int] = zCFD_Field(
        default=-1,
        title="Freeze Limiter Cycle",
        description="Cycle at which to freeze the limiter (-1 = never freeze)",
        advanced=True,
    )

    gradient: Optional[Literal["green-gauss", "least-squares"]] = zCFD_Field(
        default="green-gauss",
        title="Gradient Method",
        description="Gradient calculation method (green-gauss or least-squares)",
        advanced=True,
    )
    reconstruction: Optional[Literal["muscl", "umuscl", "lpumuscl"]] = zCFD_Field(
        default="muscl",
        title="Reconstruction Method",
        description="Reconstruction method for MUSCL-type schemes",
        advanced=True,
    )
    permit_backward_difference_during_reconstruction: bool = zCFD_Field(
        default=False,
        title="Permit Backward Difference",
        description="Allow backward difference stencils during MUSCL reconstruction",
        advanced=True,
    )

    inviscid_flux_scheme: Optional[inviscid_flux_scheme_types] = zCFD_Field(
        default="roe",
        title="Inviscid Flux Scheme",
        description="Inviscid flux calculation scheme",
    )

    @field_validator("inviscid_flux_scheme", mode="before")
    @classmethod
    def normalise_inviscid_flux_scheme(cls, v):
        """Normalise legacy mixed-case flux scheme names to lowercase."""
        if isinstance(v, str):
            return v.lower()
        return v


class FVCompressibleNumericalSchemeBase(FVNumericalSchemeBase):
    """
    Base numerical scheme for compressible finite volume methods.

    Includes parameters specific to compressible flow such as flux schemes,
    entropy fixes, and Roe low-dissipation sensors.
    """

    # Narrower than FVNumericalSchemeBase.order: "central" is read
    # only by the incompressible solver path — reject it here rather than
    # silently ignoring it on a compressible deck.
    order: compressible_order_types = zCFD_Field(
        default="second",
        title="Spatial Order",
        description="Spatial order of accuracy",
        updatable=True,
    )

    # Compressible-only
    linear_gradients: bool = zCFD_Field(
        default=False,
        title="Linear Gradients",
        description=(
            "Use an unweighted (linear) Green-Gauss gradient instead of the "
            "default inverse-distance-weighted one. Ignored when 'gradient' is "
            "'least-squares' — the least-squares branch never reads the flag."
        ),
        advanced=True,
    )

    # Multigrid V-cycles run only on the explicit pseudo-time advance path
    # (an RK sweep per level), so the clause needs an inner pseudo-time loop
    # (steady/DTS, not GTS) AND an explicit scheme — or unset, which the
    # solver treats as runge kutta.
    multigrid: ConditionalOptional[  # type: ignore[valid-type]
        MultigridConfig,
        When(
            {
                **APPLIES_WITH_PSEUDO_TIME,
                "solver.convergence_control.scheme.name": (
                    UNSET,
                    *EXPLICIT_SCHEME_NAMES,
                ),
            }
        ),
    ] = zCFD_Field(
        default=None,
        title="Multigrid",
        description="Multigrid acceleration for explicit schemes",
        advanced=True,
    )

    # Reference.cxx:198-206 reads this sub-dict out of numericalScheme()
    preconditioner: ConditionalOptional[  # type: ignore[valid-type]
        PreconditionerConfig,
        When({"solver.equations.precondition": (True,)}),
    ] = zCFD_Field(
        default=None,
        title="Preconditioner",
        description="Low Mach number preconditioner settings",
        advanced=True,
    )

    entropy_fix: Optional[float] = zCFD_Field(
        default=0.0,
        ge=0,
        le=0.2,
        title="Entropy Fix",
        description="Entropy fix coefficient for Roe scheme (limits maximum dissipation)",
        advanced=True,
    )

    roe_low_dissipation_sensor: roe_low_dissipation_sensor_types = zCFD_Field(
        default="NONE",
        title="Roe Low-Dissipation Sensor",
        description="Sensor type for Roe low-dissipation scheme",
        advanced=True,
    )

    roe_entropy_fix_coefficient: Optional[float] = zCFD_Field(
        default=0.001,
        ge=0,
        title="Roe Entropy Fix Coefficient",
        description="Roe entropy fix coefficient for low-dissipation scheme",
        advanced=True,
    )
    roe_dissipation_sensor_minimum: Optional[float] = zCFD_Field(
        default=None,
        ge=0.05,
        le=1.0,
        title="Roe Dissipation Sensor Min",
        description="Minimum value for Roe dissipation sensor (prevents excessive dissipation reduction)",
        advanced=True,
    )

    roe_low_dissipation_sensor_parameter: Optional[
        roe_low_dissipation_sensor_parameter_types
    ] = zCFD_Field(
        default=None,
        title="Roe Low-Dissipation Sensor Parameter",
        description="Field used to compute the Roe low-dissipation sensor",
        advanced=True,
    )


class FVIncompressibleNumericalSchemeBase(FVNumericalSchemeBase):
    """
    Base numerical scheme for incompressible finite volume methods.
    """

    @model_validator(mode="before")
    @classmethod
    def reject_multigrid(cls, data):
        """Multigrid is compressible-only: the advance loop returns into the
        incompressible path before the multigrid branch is ever reached, and
        coarse-mesh generation is skipped entirely for incompressible, so
        there are no coarse levels for a V-cycle to run on."""
        if isinstance(data, dict) and data.get("multigrid") is not None:
            raise ValueError(
                "'multigrid' is compressible-only: the incompressible solver "
                "never builds coarse mesh levels and never reaches the "
                "multigrid branch — remove it from numerical_scheme"
            )
        return data


# Compressible Equation Schemes
class FVCompressibleEulerNumericalScheme(FVCompressibleNumericalSchemeBase):
    """Numerical scheme for compressible Euler equations."""

    pass


class FVCompressibleViscousNumericalScheme(FVCompressibleNumericalSchemeBase):
    """Numerical scheme for compressible viscous equations."""

    viscous_time_step_factor: Optional[float] = zCFD_Field(
        default=1.0,
        gt=0,
        title="Viscous Time Step Factor",
        description="Time step scaling for viscous terms",
        advanced=True,
    )


class FVCompressibleRANSNumericalScheme(FVCompressibleViscousNumericalScheme):
    """Numerical scheme for compressible RANS equations."""

    turbulence_limiter: Optional[turbulence_limiter_types] = zCFD_Field(
        default=None,
        title="Turbulence Limiter",
        description=(
            "Treatment of invalid turbulence variables after the solution "
            "update: 'zero' clamps them to their floor, 'no update' reverts "
            "the cell's turbulence state (default when unset)"
        ),
        advanced=True,
    )


class FVCompressibleLESNumericalScheme(FVCompressibleViscousNumericalScheme):
    """Numerical scheme for compressible LES equations."""

    pass


# Incompressible Equation Schemes
class FVIncompressibleViscousNumericalScheme(FVIncompressibleNumericalSchemeBase):
    """Numerical scheme for incompressible viscous equations."""

    viscous_time_step_factor: Optional[float] = zCFD_Field(
        default=1.0,
        gt=0,
        title="Viscous Time Step Factor",
        description="Time step scaling for viscous terms",
    )


class FVIncompressibleRANSNumericalScheme(FVIncompressibleViscousNumericalScheme):
    """Numerical scheme for incompressible RANS equations."""

    turbulence_limiter: Optional[turbulence_limiter_types] = zCFD_Field(
        default=None,
        title="Turbulence Limiter",
        description=(
            "Treatment of invalid turbulence variables after the solution "
            "update: 'zero' clamps them to their floor, 'no update' reverts "
            "the cell's turbulence state (default when unset)"
        ),
    )


class FVIncompressibleLESNumericalScheme(FVIncompressibleViscousNumericalScheme):
    """Numerical scheme for incompressible LES equations."""

    pass


FVIncompressibleNumericalScheme = Union[
    FVIncompressibleViscousNumericalScheme,
    FVIncompressibleLESNumericalScheme,
    FVIncompressibleRANSNumericalScheme,
]

FVCompressibleNumericalScheme = Union[
    FVCompressibleEulerNumericalScheme,
    FVCompressibleViscousNumericalScheme,
    FVCompressibleRANSNumericalScheme,
    FVCompressibleLESNumericalScheme,
]

# Union type for all numerical schemes
NumericalScheme = Union[FVIncompressibleNumericalScheme, FVCompressibleNumericalScheme]
