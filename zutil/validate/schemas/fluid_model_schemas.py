from pydantic import BaseModel, ConfigDict
from typing import Union
from zutil.validate.schemas.base_types import (
    NONZERO,
    ConditionalOptional,
    When,
    zCFD_Field,
)
from .base_types import Vector3d

_INCOMPRESSIBLE = When({"solver.solver_settings.type": ("incompressible",)})
WITH_GRAVITY = {"solver.fluid_properties.gravity": NONZERO}
WITH_LATITUDE = {"solver.fluid_properties.latitude": NONZERO}


class FluidModelBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material: str = zCFD_Field(
        default="ideal gas", title="Material", description="Fluid material model"
    )
    gamma: float = zCFD_Field(
        default=1.4, gt=1.0, title="Gamma", description="Ratio of specific heats Cp/Cv"
    )
    gas_constant: float = zCFD_Field(
        default=287.0,
        gt=0,
        title="Gas Constant",
        description="Specific gas constant (J/(kg·K))",
    )
    sutherlands_const: float = zCFD_Field(
        default=110.4,
        gt=0,
        title="Sutherlands Const",
        description="Sutherland's reference temperature (K)",
    )
    prandtl_no: float = zCFD_Field(
        default=0.72, gt=0, title="Prandtl no", description="Molecular Prandtl number"
    )

    gravity: ConditionalOptional[  # type: ignore[valid-type]
        Vector3d, _INCOMPRESSIBLE
    ] = zCFD_Field(
        default=None,
        title="Gravity",
        description=(
            "Gravitational acceleration vector, for example [0, 0, -9.81]. Drives "
            "Boussinesq and anelastic buoyancy, sets the vertical for the Coriolis "
            "force and the stratified turbulence terms, and enters the Monin-Obukhov "
            "wall law. Omitted means no gravity."
        ),
        units="m/s^2",
        advanced=True,
    )
    latitude: ConditionalOptional[  # type: ignore[valid-type]
        float,
        When({**_INCOMPRESSIBLE.conditions, **WITH_GRAVITY}),
    ] = zCFD_Field(
        default=None,
        ge=-90,
        le=90,
        title="Latitude",
        description=(
            "Latitude for the f-plane Coriolis force, f = 2 Omega sin(latitude), "
            "positive north. Needs a non-zero gravity, which sets the local vertical. "
            "Omitted means no Coriolis force."
        ),
        units="deg",
        advanced=True,
    )
    planet_rotation: ConditionalOptional[  # type: ignore[valid-type]
        float,
        When({**_INCOMPRESSIBLE.conditions, **WITH_LATITUDE}),
    ] = zCFD_Field(
        default=None,
        gt=0,
        title="Planet Rotation",
        description=(
            "Planetary rotation rate for the Coriolis parameter f = 2 Omega sin(latitude). "
            "Needs a non-zero latitude. Defaults to Earth's sidereal rate, 7.292115e-5"
        ),
        units="rad/s",
        advanced=True,
    )
    thermal_expansion_coeff: ConditionalOptional[  # type: ignore[valid-type]
        float,
        When(
            {
                **_INCOMPRESSIBLE.conditions,
                **WITH_GRAVITY,
                "solver.equations.solve_energy": (True,),
            }
        ),
    ] = zCFD_Field(
        default=None,
        gt=0,
        title="Thermal Expansion Coefficient",
        description=(
            "Thermal expansion coefficient beta for Boussinesq buoyancy. The solver "
            "stores the dimensionless group beta*Tref, so the buoyancy source becomes "
            "-g_nondim * (beta*Tref) * (theta - 1). Needs a non-zero gravity. Ignored "
            "when base_state is set."
        ),
        units="1/K",
        advanced=True,
    )


class ViscousFluidModel(FluidModelBase):
    """Fluid model for viscous flows."""

    model_config = ConfigDict(extra="forbid")

    turbulent_prandtl_no: float = zCFD_Field(
        default=0.9,
        gt=0,
        title="Turbulent Prandtl no",
        description="Turbulent Prandtl number",
    )


class IncompressibleFluidModel(ViscousFluidModel):
    """Fluid model for the incompressible solver."""

    model_config = ConfigDict(extra="forbid")


# Union type for different fluid models
FluidModel = Union[FluidModelBase, ViscousFluidModel, IncompressibleFluidModel]


def get_fluid_model_schema(equations: str):
    """Return the correct fluid model class for the given equation type."""
    if equations == "euler":
        return FluidModelBase
    return ViscousFluidModel
