"""
Solver block: how to solve, and the defaults every mesh inherits.

Blocks whose version depends on the solver or equation type take it from
their table in block_versions.py.
"""

from typing import Annotated, Dict, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from zutil.validate.schemas.base_types import (
    UNSET,
    ConditionalOptional,
    RequiredWhen,
    When,
    zCFD_Field,
)

from . import block_versions as versions
from . import convergence_schemas as conv
from . import equations_schemas as eqn
from . import fluid_model_schemas as fld
from . import model_schemas as mod
from . import numerical_schemes_schemas as num
from . import output_schemas as out
from . import reference_conditions_schemas as refs
from . import solver_settings_schemas as sset
from . import time_settings_schemas as ts


class SolverBlockModel(BaseModel):
    """
    Core solver block for zCFD simulations.
    This model defines global configuration "HOW" options and sets the
    overrideable defaults.
    """

    model_config = ConfigDict(extra="forbid")

    solver_settings: Union[
        sset.CompressibleSolverSettingsPropertiesModel,
        sset.IncompressibleSolverSettingsPropertiesModel,
    ] = zCFD_Field(
        title="Solver Settings",
        description="Which solver runs the case, and how it is partitioned and parallelised",
        variants=versions.SOLVER_SETTINGS,
    )
    equations: eqn.Equations = zCFD_Field(
        title="Equations",
        description="The equations solved, and their turbulence and transport models",
        variants=versions.EQUATIONS_BLOCK,
    )
    fluid_properties: fld.FluidModel = zCFD_Field(
        title="Fluid Properties",
        description="Fluid model properties and definition",
        variants=versions.FLUID_PROPERTIES,
    )
    reference_conditions: Dict[str, refs.ReferenceConditions] = zCFD_Field(
        default_factory=dict,
        title="Reference Conditions",
        description="Dictionary of reference conditions used throughout the simulation",
        variants=versions.REFERENCE_CONDITION,
    )
    initialisation: mod.InitialisationModel = zCFD_Field(
        title="Initialisation",
        description="Default initialisation and restart settings for all meshes.",
    )
    time_settings: ts.TimeSettings = zCFD_Field(
        title="Time Settings",
        description="Physical time layout (steady, global or dual timestepping)",
        variants=versions.TIME_SETTINGS,
    )
    convergence_control: Annotated[
        ConditionalOptional[  # type: ignore[valid-type]
            Union[
                conv.CompressibleConvergenceControl,
                conv.IncompressibleConvergenceControl,
            ],
            When(ts.APPLIES_WITH_PSEUDO_TIME),
        ],
        RequiredWhen(
            When({**ts.APPLIES_WITH_PSEUDO_TIME, "model.*.convergence_control": UNSET})
        ),
    ] = zCFD_Field(
        default=None,
        title="Convergence Control",
        description="Convergence pacing for the inner pseudo-time loop (required for steady/DTS)",
        variants=versions.CONVERGENCE_CONTROL,
    )

    numerical_scheme: num.NumericalScheme = zCFD_Field(
        title="Numerical Scheme",
        description="Default spatial numerical scheme for all meshes.",
        variants=versions.NUMERICAL_SCHEME,
    )
    output_settings: out.OutputModel = zCFD_Field(
        title="Output Settings",
        description="Default output variables and monitors for all meshes.",
    )

    @field_validator("reference_conditions", mode="before")
    @classmethod
    def expand_inherited_conditions(cls, data):
        """A condition may name another with ``reference`` and inherit its settings."""
        from ...validate.validation_utils import expand_reference_inheritance

        if not isinstance(data, dict):
            return data
        return expand_reference_inheritance(data)

    @model_validator(mode="after")
    def resolve_base_state_surface(self):
        """Fill the base state's surface temperature/pressure from its named
        reference condition (default: the initialisation reference), so the
        surface state is written once per deck. Explicit values win."""
        base_state = getattr(self.equations, "base_state", None)
        if base_state is None:
            return self
        if (
            base_state.surface_temperature is None
            or base_state.surface_pressure is None
        ):
            name = base_state.reference or self.initialisation.reference
            if name is None:
                initial = self.initialisation.initial_conditions
                if isinstance(initial, str) and initial in self.reference_conditions:
                    name = initial
            if name is None:
                raise ValueError(
                    "'base_state' needs a surface state: name a reference "
                    "condition ('base_state.reference' or "
                    "'initialisation.reference'), or set 'surface_temperature' "
                    "and 'surface_pressure' explicitly"
                )
            if name not in self.reference_conditions:
                raise ValueError(
                    f"'base_state' reference '{name}' is not a reference condition"
                )
            condition = self.reference_conditions[name]
            if base_state.surface_temperature is None:
                base_state.surface_temperature = condition.temperature
            if base_state.surface_pressure is None:
                base_state.surface_pressure = condition.pressure
        return self


MESH_OVERRIDES = tuple(
    name
    for name in mod.ModelSettings.model_fields
    if name in SolverBlockModel.model_fields
)
