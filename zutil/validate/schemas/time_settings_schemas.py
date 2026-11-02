from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing import Union, Literal, Optional
from zutil.validate.schemas.base_types import UNSET, zCFD_Field

from .convergence_schemas import (
    GTS_SCHEME_STAGES,
    GTSTimeScheme,
    _default_scheme_name,
)

# The inner pseudo-time convergence loop exists only for steady and DTS runs —
# GTS marches physical time directly with no inner loop.  Settings that pace
# or accelerate that loop (convergence_control, multigrid, preconditioning)
# carry this clause in their When(): steady dumps no 'kind' (UNSET), DTS is
# explicit, GTS ('global timestepping') is excluded.
APPLIES_WITH_PSEUDO_TIME = {
    "solver.time_settings.kind": (UNSET, "dual time stepping"),
}

order_types = Literal["first", "second", 1, 2]

GTS_PSEUDO_TIME_ONLY_KEYS = {
    "order": (
        "'order' is the backward-difference order of the physical time "
        "derivative, which global time stepping does not assemble — the "
        "scheme marches physical time directly and carries the order of "
        "accuracy on its own. Remove 'order' (the scheme is third "
        "order), or use kind: 'dual time stepping' if a BDF march is "
        "what you want."
    ),
    "start": (
        "'start' is a count of steady pseudo-time cycles to run before "
        "time-accurate stepping begins, and global time stepping has no "
        "pseudo-time loop to run them in — the value was silently "
        "ignored. Remove it; to start from a converged field, restart a "
        "GTS run from a steady run's results instead."
    ),
}


_KIND = (
    "How each physical time step is taken: `global timestepping` advances in "
    "physical time alone, `dual time stepping` converges a pseudo-time loop "
    "within every step."
)


class GTSTimeSettings(BaseModel):
    """Global Timestepping — explicit physical time march.

    GTS advances the solution in physical time only, with no pseudo-time
    loop: convergence_control, 'order' and 'start' may not be set.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["unsteady"] = zCFD_Field(
        default="unsteady",
        title="Type",
        description=(
            "Whether the run is time-accurate. 'unsteady' marches in physical "
            "time over total_time in steps of time_step, and kind selects how "
            "each step is taken; 'steady' converges a single steady state in "
            "pseudo-time."
        ),
    )
    kind: Literal["global timestepping"] = zCFD_Field(..., description=_KIND)
    total_time: float = zCFD_Field(
        ...,
        title="Total Time",
        gt=0,
        description="Total physical simulation time",
        updatable=True,
    )
    time_step: float = zCFD_Field(
        ...,
        title="Time Step",
        gt=0,
        description="Physical time step size",
        updatable=True,
    )
    scheme: GTSTimeScheme = zCFD_Field(
        ...,
        title="Scheme",
        description=(
            "Time integration scheme. GTS marches physical time with this "
            "scheme itself, so the scheme sets the run's order of accuracy — "
            "only the three-stage TVD Runge-Kutta (SSP-RK3, third order) is "
            "admissible."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_pseudo_time_only_keys(cls, data):
        """Reject 'order' and 'start', which a GTS run cannot act on."""
        if isinstance(data, dict):
            for key, message in GTS_PSEUDO_TIME_ONLY_KEYS.items():
                if key in data:
                    raise ValueError(message)
        return data

    @field_validator("scheme", mode="before")
    @classmethod
    def _fill_default_scheme_name(cls, v):
        """Fill in the omitted scheme name, then reject anything but the three-stage TVD Runge-Kutta."""
        v = _default_scheme_name(v)
        if not isinstance(v, dict):
            return v
        stage = v.get("stage")
        if v.get("name") != "runge kutta" or (
            stage is not None and stage not in GTS_SCHEME_STAGES
        ):
            raise ValueError(
                "global time stepping marches physical time with the "
                "scheme itself, so only the three-stage TVD "
                "Runge-Kutta is admissible — "
                "{'name': 'runge kutta', 'stage': 3} (equivalently "
                "'rk third order tvd'). Every other scheme is first order "
                "in physical time (euler, stage 1, and the stage 4/5 "
                "steady-state smoothers) or cannot drive an explicit march "
                "at all (implicit euler, lu-sgs); use "
                "kind: 'dual time stepping' to run those."
            )
        return v


class DTSTimeSettings(BaseModel):
    """Dual Time Stepping — physical time pacing.

    DTS advances the solution in physical time, with a pseudo-time
    convergence loop at each physical step.  The pseudo-time scheme,
    CFL, and cycle count live in convergence_control, not here.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["unsteady"] = zCFD_Field(
        default="unsteady",
        title="Type",
        description=(
            "Whether the run is time-accurate. 'unsteady' marches in physical "
            "time over total_time in steps of time_step, and kind selects how "
            "each step is taken; 'steady' converges a single steady state in "
            "pseudo-time."
        ),
    )
    kind: Literal["dual time stepping"] = zCFD_Field(..., description=_KIND)
    total_time: float = zCFD_Field(
        ...,
        title="Total Time",
        gt=0,
        description="Total physical simulation time",
        updatable=True,
    )
    time_step: float = zCFD_Field(
        ...,
        title="Time Step",
        gt=0,
        description="Physical time step size",
        updatable=True,
    )
    order: Optional[order_types] = zCFD_Field(
        default="second",
        title="Order",
        description="Time integration order",
        updatable=True,
    )
    start: Optional[int] = zCFD_Field(
        default=0,
        ge=0,
        title="Start",
        description=(
            "Steady pseudo cycles run at real-time cycle 0 before time-accurate "
            "stepping begins; the same solve as a steady run of that length. "
            "A settling phase for an impulsive start. 0 (the default) runs none, "
            "so real-time cycle 0 is itself time-accurate and a supplied initial "
            "condition is marched from as given"
        ),
        updatable=True,
    )

    @model_validator(mode="after")
    def reject_v1_steady_idiom(self):
        """total_time == time_step is the v1 steady idiom, not a 1-step DTS run.

        v1 (ExplicitSolver.py) explicitly detected this and ran STEADY
        (num_real_time_cycle = 0, dual time stepping off), so no v1 deck
        with this shape ever time-marched. Accepting it here would run a
        genuine dual-time step (doubling the cycle budget) on decks that
        meant steady.
        """
        if self.total_time == self.time_step:
            raise ValueError(
                "time_settings: total_time == time_step is the v1 steady "
                "idiom — declare {'type': 'steady'} instead (or set "
                "total_time to a genuine multiple of time_step for a real "
                "dual-time run)"
            )
        return self


class SteadyTimeSettings(BaseModel):
    """Timeline parameters for steady simulations.

    Steady simulations have no physical time — convergence is driven
    entirely by convergence_control.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["steady"] = zCFD_Field(
        default="steady",
        title="Type",
        description=(
            "Whether the run is time-accurate. 'steady' converges a single "
            "steady state in pseudo-time; 'unsteady' marches in physical time "
            "over total_time in steps of time_step."
        ),
    )


# Type alias for annotations
TimeSettings = Union[SteadyTimeSettings, GTSTimeSettings, DTSTimeSettings]
