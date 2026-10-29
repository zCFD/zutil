"""
Initialisation settings for zCFD simulations.

Defines the InitialisationModel for configuring simulation startup,
restart, and initial condition assignment.
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    WrapValidator,
    field_validator,
    model_validator,
)
from typing import Annotated, Any, Callable, Literal, Optional, Union
from zutil.validate.schemas.base_types import (
    UNSET,
    ConditionalOptional,
    RequiredWhen,
    When,
    zCFD_Field,
)
from zutil.validate.schemas.callables import INITIAL_CONDITIONS


_RESTARTING = When({"<self>.restart": (True,)})


class InitialisationModel(BaseModel):
    """Settings for mesh initialisation and restart."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    reference: Annotated[
        Optional[str],
        RequiredWhen(
            When({"<self>.restart": (True,), "<self>.initial_conditions": UNSET})
        ),
    ] = zCFD_Field(
        default=None,
        title="Reference",
        description=(
            "Reference condition used for non-dimensionalization (e.g. 'IC_1'). "
            "When the initial_conditions IC has zero velocity (e.g. a quiescent "
            "cavity), set this to a reference condition with the physical "
            "velocity/Reynolds number so the solver can non-dimensionalize correctly."
        ),
    )
    initial_conditions: Annotated[
        Optional[Union[str, Callable[..., dict[str, Any]]]],
        WrapValidator(INITIAL_CONDITIONS.validate),
    ] = zCFD_Field(
        default=None,
        title="Initial Conditions",
        description=INITIAL_CONDITIONS.description(
            "Initial conditions for the simulation: a named condition from "
            "reference_conditions, or a function of each cell's dimensional "
            "state (m, m, Pa, K, m/s)"
        )
        + ". Required unless restart=True.",
    )
    restart: bool = zCFD_Field(
        default=False,
        title="Restart",
        description="Restart from a previous solution. Default is False.",
    )
    restart_casename: ConditionalOptional[  # type: ignore[valid-type]
        str, _RESTARTING
    ] = zCFD_Field(
        default=None,
        title="Restart Casename",
        description="Name of the case to restart from. Defaults to the current casename.",
    )
    restart_meshname: ConditionalOptional[  # type: ignore[valid-type]
        str, _RESTARTING
    ] = zCFD_Field(
        default=None,
        title="Restart Meshname",
        description=(
            "Mesh the restart case ran on, for an interpolated restart from a results file that has no "
            "case record beside it. A case run by this version of zCFD records its own mesh."
        ),
        advanced=True,
    )
    restart_ignore_history: ConditionalOptional[  # type: ignore[valid-type]
        bool, _RESTARTING
    ] = zCFD_Field(
        default=False,
        title="Restart Ignore History",
        description="Ignore cycle history from restart file. Default is False.",
    )
    interpolate_restart: ConditionalOptional[  # type: ignore[valid-type]
        bool, _RESTARTING
    ] = zCFD_Field(
        default=False,
        title="Interpolate Restart",
        description=(
            "Start from the restart case's solution interpolated onto this case's mesh, which may differ in "
            "resolution, cell type or geometry. Variables the restart case does not solve come from the "
            "initial conditions."
        ),
    )
    interpolation: ConditionalOptional[  # type: ignore[valid-type]
        Literal["linear", "injection"], _RESTARTING
    ] = zCFD_Field(
        default="linear",
        title="Interpolation",
        description=(
            "How an interpolated restart forms each new cell's value from the old cell containing its centre: "
            "'linear' adds the old cell's least-squares gradient times the offset between the centres, held "
            "within the values of that cell and its face neighbours; 'injection' copies the old cell's value."
        ),
        options={
            "linear": "Linear, bounded by the neighbouring cells",
            "injection": "Copy the containing cell",
        },
        advanced=True,
    )
    restart_models: ConditionalOptional[  # type: ignore[valid-type]
        dict[str, str], _RESTARTING
    ] = zCFD_Field(
        default=None,
        title="Restart Models",
        description=(
            "For an interpolated restart, the restart case's model each of this case's models starts from, "
            "where the names differ, as {this case's model: restart case's model}. Models not listed start "
            "from the model of the same name."
        ),
        advanced=True,
    )
    interpolate_outside_limit: ConditionalOptional[  # type: ignore[valid-type]
        float, _RESTARTING
    ] = zCFD_Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        title="Interpolate Outside Limit",
        description=(
            "Largest fraction of this case's cells that may lie outside the restart case's mesh, so 0.001 is "
            "0.1%. Those cells take the nearest restart cell's value; above the limit the run stops."
        ),
        advanced=True,
    )

    @model_validator(mode="after")
    def check_initial_conditions_required(self):
        """initial_conditions is required unless restart=True."""
        if self.initial_conditions is None and not self.restart:
            raise ValueError(
                "'initial_conditions' is required when 'restart' is False. "
                "Set it to a named IC (e.g. 'IC_1') or a callable function, "
                "or set 'restart: True' to resume from a previous solution."
            )
        if self.initial_conditions is None and self.interpolate_restart:
            raise ValueError(
                "'initial_conditions' is required with 'interpolate_restart': it sets any variable the "
                "restart case does not solve."
            )
        return self

    @model_validator(mode="after")
    def check_reference_required_for_callable(self):
        """reference is required when initial_conditions is a callable."""
        if callable(self.initial_conditions) and self.reference is None:
            raise ValueError(
                "'reference' is required when 'initial_conditions' is a callable function. "
                "Set it to the name of the reference condition used for non-dimensionalization "
                "(e.g. 'IC_1')."
            )
        return self

    @field_validator("restart_meshname")
    @classmethod
    def strip_h5_suffix(cls, v):
        """Strip a trailing '.h5' from restart_meshname (legacy __init__.py:1673-1683)."""
        if v and v.endswith(".h5"):
            return v[:-3]
        return v

    @model_validator(mode="after")
    def check_interpolate_restart_requirements(self):
        """An interpolated restart starts from a named restart case."""
        if self.interpolate_restart and not (self.restart and self.restart_casename):
            raise ValueError(
                "'interpolate_restart' needs 'restart': True and 'restart_casename' naming the case to "
                "start from."
            )
        return self
