import warnings

from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
    ConfigDict,
    ValidationError,
)
from typing import Annotated, Union, Optional
from zutil.validate.schemas.base_types import (
    UNSET,
    case_file,
    ConditionalOptional,
    RequiredWhen,
    When,
    zCFD_Field,
)
from .base_types import Vector3d

# The incompressible inflow BC reads only roughness_length and
# friction_velocity from the ABL dict; the remaining profile fields are
# consumed only by the compressible initial-condition and farfield read sites.
_ABL_COMPRESSIBLE_ONLY = When({"solver.solver_settings.type": (UNSET, "compressible")})


class ABLProfile(BaseModel):
    """
    Atmospheric Boundary Layer (ABL) profile parameters.
    Used for atmospheric flow simulations with canopy models.

    Consumed by three solver read sites, each reading a different subset of
    these keys: the initial condition, the compressible farfield BC, and the
    incompressible inflow BC. The incompressible path reads only
    ``roughness_length``/``friction_velocity``, so ``ground_level``,
    stability and ``tke`` are silently inert there.
    """

    model_config = ConfigDict(extra="forbid")

    roughness_length: float = zCFD_Field(
        gt=0,
        title="Roughness Length",
        description=(
            "Surface roughness length in meters. Must be positive. "
            "This is the aerodynamic roughness z₀ in the log-law "
            "profile (not to be confused with 'ground_level', which is a "
            "ground datum elevation, not a roughness)."
        ),
    )
    friction_velocity: Annotated[
        Optional[float],
        RequiredWhen(
            When(
                {
                    "solver.solver_settings.type": ("compressible",),
                    "<self>.surface_layer_height": UNSET,
                }
            )
        ),
    ] = zCFD_Field(
        default=None,
        gt=0,
        title="Friction Velocity",
        description=(
            "Friction velocity in m/s. Must be positive when set. The "
            "compressible solver needs friction_velocity, "
            "surface_layer_height or both."
        ),
    )
    surface_layer_height: ConditionalOptional[  # type: ignore[valid-type]
        float, _ABL_COMPRESSIBLE_ONLY
    ] = zCFD_Field(
        default=None,
        gt=0,
        title="Surface Layer Height",
        description=(
            "Height of the surface layer (m) that the log-law profile applies "
            "within; above this height the profile is clamped. C++ sentinel "
            "default is -1.0 (unset): the compressible read sites then derive "
            "it from roughness_length and friction_velocity when both are set."
        ),
    )
    monin_obukhov_length: ConditionalOptional[  # type: ignore[valid-type]
        float, _ABL_COMPRESSIBLE_ONLY
    ] = zCFD_Field(
        default=None,
        title="Monin-Obukhov Length",
        description=(
            "Monin-Obukhov stability length L (m). Negative values are "
            "read as unstable and add a stability correction to the "
            "log-law velocity/TKE/epsilon (only when L < 0 — the current "
            "C++ never applies a correction for L > 0, i.e. stable "
            "stratification is not represented here). Zero is not a valid "
            "value. The solver's local default is +1.0 (a positive, "
            "'no correction' value) when the key is absent — this is the "
            "effective neutral default. Leave unset for neutral stability."
        ),
    )
    tke: ConditionalOptional[  # type: ignore[valid-type]
        float, _ABL_COMPRESSIBLE_ONLY
    ] = zCFD_Field(
        default=None,
        ge=0,
        title="Reference TKE",
        description=(
            "Reference turbulent kinetic energy (m^2/s^2), scaled by "
            "friction_velocity**2 to give the profile's TKE. Only applied "
            "when friction_velocity is also set — if friction_velocity is "
            "absent, the solver discards "
            "this value with a warning and derives TKE from the wall "
            "distance instead."
        ),
    )
    up: ConditionalOptional[  # type: ignore[valid-type]
        Vector3d, _ABL_COMPRESSIBLE_ONLY
    ] = zCFD_Field(
        default=None,
        title="Up",
        description=(
            "Unit vector [x, y, z] defining the vertical ('up') direction "
            "used with ground_level to compute height above ground as "
            "dot(position, up) - ground_level. "
            "Default when unset is (0, 0, 1)."
        ),
    )
    ground_level: ConditionalOptional[  # type: ignore[valid-type]
        float, _ABL_COMPRESSIBLE_ONLY
    ] = zCFD_Field(
        default=0.0,
        ge=0,
        title="Ground Level",
        description=(
            "Datum elevation of the ground (m) in the mesh's own coordinate "
            "system, measured along 'up'. Subtracted from dot(position, up) "
            "before the log-law height is computed, as "
            "height = min(dot(p, up) - ground_level, surface_layer_height) "
            "— it shifts where the "
            "profile's height origin sits, it is NOT a measurement or "
            "reference height. This is unrelated to the aerodynamic "
            "roughness length z₀ (that's 'roughness_length' above) and "
            "unrelated to ABLZone.reference_height. NOTE: the initial-condition "
            "profile reads this key but never applies it "
            "to the initial field — only the farfield BC profile honours it "
            "— so a mesh with a non-zero ground datum gets inconsistent "
            "IC/BC profiles."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise_legacy_abl_keys(cls, data):
        """Accept legacy key spellings and emit the canonical key.

        ``z0`` is the legacy spelling of ``ground_level``.
        ``"Monin-Obukhov length"`` is the raw C++/v1 dict key; every v2 deck
        already runs through ``replace_spaces_in_keys()`` before reaching
        this model, which lowercases it and replaces spaces (but not
        hyphens) with underscores, producing ``"monin-obukhov_length"`` —
        still not a valid snake_case field name. Both legacy spellings (and
        the untouched raw spelling, for callers constructing ABLProfile
        directly) are normalised here to the canonical field name; this
        normalises *input*, it does not create a second output key.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        if "z0" in data:
            if "ground_level" in data:
                raise ValueError(
                    "ABLProfile: specify either the legacy 'z0' or the "
                    "canonical 'ground_level', not both."
                )
            data["ground_level"] = data.pop("z0")

        for legacy_key in (
            "monin-obukhov_length",
            "monin-obukhov length",
            "Monin-Obukhov length",
        ):
            if legacy_key in data:
                if "monin_obukhov_length" in data:
                    raise ValueError(
                        "ABLProfile: specify either the legacy Monin-Obukhov "
                        "length spelling or the canonical "
                        "'monin_obukhov_length', not both."
                    )
                data["monin_obukhov_length"] = data.pop(legacy_key)

        return data

    @field_validator("monin_obukhov_length")
    @classmethod
    def _reject_zero_monin_obukhov_length(cls, v):
        if v is not None and v == 0:
            raise ValueError(
                "monin_obukhov_length must be nonzero — L=0 is not a "
                "physically valid Monin-Obukhov length. Omit the field "
                "for neutral stability instead."
            )
        return v

    @model_validator(mode="after")
    def _warn_tke_without_friction_velocity(self):
        # Mirrors the solver's runtime warning: 'tke' is scaled by
        # friction_velocity**2 only when friction_velocity is also set;
        # otherwise the value is silently discarded (not an error), so this
        # warns rather than rejects.
        if self.tke is not None and self.friction_velocity is None:
            warnings.warn(
                "ABLProfile: 'tke' is set without 'friction_velocity'. The "
                "profile initialiser only uses 'tke' scaled by "
                "friction_velocity**2 when friction_velocity is also set; "
                "with friction_velocity unset, "
                "'tke' is silently ignored and TKE is derived from the wall "
                "distance instead.",
                UserWarning,
                stacklevel=2,
            )
        return self


ProfileMap = case_file("reference condition profile")


class ProfileModel(BaseModel):
    """
    Container for different profile types.
    """

    model_config = ConfigDict(extra="forbid")

    abl: Optional[ABLProfile] = zCFD_Field(
        default=None,
        title="ABL",
        description="Atmospheric Boundary Layer profile parameters",
    )

    # File-based profile
    field: Optional[ProfileMap] = zCFD_Field(
        default=None, title="Field", description="Profile file name (e.g. .vtp)"
    )
    use_wall_distance: Optional[bool] = zCFD_Field(
        default=None,
        title="Use Wall Distance",
        description="Use wall distance for profile",
    )


class InflowVectorModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector: Vector3d = zCFD_Field(
        title="Vector",
        description=(
            "Velocity [Vx, Vy, Vz] in m/s. When mach is set, the compressible "
            "solver keeps only its direction and takes the speed from the Mach "
            "number; the incompressible solver ignores mach."
        ),
    )
    mach: Optional[float] = zCFD_Field(
        None,
        ge=0,
        title="Mach",
        description="Mach number to scale magnitude of inflow vector by",
    )  # Mach number must be a positive float


class ReferenceConditionsBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = zCFD_Field(
        ge=0, title="Temperature", description="Static temperature (K)"
    )
    pressure: float = zCFD_Field(
        ge=0, title="Pressure", description="Static pressure (Pa)"
    )
    v: Union[Vector3d, InflowVectorModel] = zCFD_Field(
        default={"vector": [0.0, 0.0, 0.0]},
        title="V",
        description="Velocity vector [Vx, Vy, Vz] (m/s) or inflow vector specification",
    )

    @field_validator("v", mode="before")
    @classmethod
    def _wrap_bare_velocity_vector(cls, v):
        """Canonicalise a bare [Vx, Vy, Vz] to {"vector": [...]}, the one shape the solver reads."""
        if isinstance(v, (list, tuple)):
            return {"vector": list(v)}
        return v

    reference_length: float = zCFD_Field(
        default=1.0,
        ge=0,
        title="Reference Length",
        description="Reference length (m) used in Reynolds number calculation",
    )
    profile: Optional[ProfileModel] = zCFD_Field(
        default=None,
        title="Profile",
        description="Atmospheric profile parameters for specialised flow conditions",
    )
    # Pressure and temperature ratios — applicable to all equation types.
    total_pressure_ratio: Optional[float] = zCFD_Field(
        default=None,
        ge=0,
        title="Total Pressure Ratio",
        description="Total pressure ratio for inflow boundary conditions",
        group="Ratios",
    )
    total_temperature_ratio: Optional[float] = zCFD_Field(
        default=None,
        ge=0,
        title="Total Temperature Ratio",
        description="Total temperature ratio for inflow boundary conditions",
        group="Ratios",
    )
    static_pressure_ratio: Optional[float] = zCFD_Field(
        default=None,
        ge=0,
        title="Static Pressure Ratio",
        description="Static pressure ratio for outflow boundary conditions",
        group="Ratios",
    )


class EulerReferenceConditions(ReferenceConditionsBase):
    """
    Reference conditions for Euler equations (inviscid flow).
    Use with the compressible solver.
    Do not include viscosity-related fields - use viscous or RANS equations if viscosity is needed.
    """

    pass


class ViscousReferenceConditions(ReferenceConditionsBase):
    """
    Reference conditions for viscous equations (laminar flow).
    Use with compressible or incompressible solvers.
    Do not include turbulence-related fields - use RANS equations if turbulence modeling is needed.
    """

    reynolds_no: Optional[float] = zCFD_Field(
        default=None,
        ge=0,
        title="Reynolds no",
        description="Reynolds number. Only valid for viscous or RANS equations.",
    )
    viscosity: Optional[float] = zCFD_Field(
        default=None,
        ge=0,
        title="Viscosity",
        description="Viscosity. Only valid for viscous or RANS equations.",
    )

    @model_validator(mode="wrap")
    @classmethod
    def validate_reynolds_viscosity_mutual_exclusivity(cls, data, handler):
        """Ensure either reynolds_no or viscosity is set, but not both."""
        errors = []
        try:
            validated = handler(data)
        except ValidationError as e:
            errors.extend(e.errors(include_url=False))
            validated = None

        if isinstance(data, dict):
            reynolds_no = data.get("reynolds_no")
            viscosity = data.get("viscosity")
            have_data = True
        elif isinstance(data, cls):
            # model_validate() called on an already-constructed instance
            # (e.g. re-validating a model built elsewhere) bypasses the
            # dict branch entirely; read the fields off the instance so
            # the mutual-exclusivity check still applies.
            reynolds_no = data.reynolds_no
            viscosity = data.viscosity
            have_data = True
        else:
            have_data = False

        if have_data:
            from pydantic_core import PydanticCustomError

            # Check that only one of reynolds_no or viscosity is set (not both, not neither)
            if (reynolds_no is not None) and (viscosity is not None):
                msg = f"ViscousReferenceConditions: Cannot specify both 'reynolds_no' ({reynolds_no}) and 'viscosity' ({viscosity}). These are mutually exclusive - specify only one to define the flow viscosity. Use 'reynolds_no' for non-dimensional specification or 'viscosity' for direct physical viscosity specification."
                errors.append(
                    {
                        "type": PydanticCustomError("value_error", msg),
                        "loc": ("reynolds_no",),
                        "input": data,
                    }
                )
            elif (reynolds_no is None) and (viscosity is None):
                msg = "ViscousReferenceConditions: Must specify either 'reynolds_no' or 'viscosity' to define the flow viscosity. For viscous flow equations, the viscosity must be defined either non-dimensionally (reynolds_no) or dimensionally (viscosity)."
                errors.append(
                    {
                        "type": PydanticCustomError("value_error", msg),
                        "loc": ("reynolds_no",),
                        "input": data,
                    }
                )

        if errors:
            from pydantic_core import ValidationError as PydanticCoreValidationError

            raise PydanticCoreValidationError.from_exception_data(cls.__name__, errors)

        return validated


class RANSReferenceConditions(ViscousReferenceConditions):
    """
    Reference conditions for RANS equations (turbulent flow).
    Use with compressible or incompressible solvers.
    Includes both viscosity and turbulence modeling fields.
    """

    turbulence_intensity: float = zCFD_Field(
        default=0.01,
        ge=0,
        title="Turbulence Intensity",
        description="Freestream turbulence intensity (fraction, e.g. 0.01 = 1%)",
        group="Turbulence",
    )
    eddy_viscosity_ratio: float = zCFD_Field(
        default=0.1,
        ge=0,
        title="Eddy Viscosity Ratio",
        description="Ratio of eddy viscosity to molecular viscosity (μt/μ)",
        group="Turbulence",
    )

    # Ambient turbulence parameters
    ambient_turbulence_intensity: float = zCFD_Field(
        default=1.0e-20,
        ge=0,
        title="Ambient Turbulence Intensity",
        description="Ambient turbulence intensity for far-field conditions",
        group="Turbulence",
    )
    ambient_eddy_viscosity_ratio: float = zCFD_Field(
        default=1.0e-20,
        ge=0,
        title="Ambient Eddy Viscosity Ratio",
        description="Ambient eddy viscosity ratio for far-field conditions",
        group="Turbulence",
    )


# Reference conditions union - route to appropriate model based on equations type
ReferenceConditions = Union[
    EulerReferenceConditions,
    ViscousReferenceConditions,
    RANSReferenceConditions,
]
