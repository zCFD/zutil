from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    WrapValidator,
    field_serializer,
    field_validator,
)
from pydantic_core.core_schema import (
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
)
from typing import Union, List, Literal, Optional, Callable, Any, Annotated
from zutil.validate.schemas.base_types import (
    SET,
    UNSET,
    ConditionalChoice,
    ConditionalOptional,
    NotIn,
    When,
    zCFD_Field,
)
from zutil.validate.schemas.callables import CFL_RAMP
from zutil.validate.schemas.union_tags import tags_of


_RAMP_TYPE = (
    "How the ramp changes the CFL in its window: `exponential` multiplies it, "
    "`linear` and `stepped_linear` add to it."
)


class _RampBase(BaseModel):
    """Fields shared by every ramp shape."""

    model_config = ConfigDict(extra="forbid")

    initial: Optional[float] = zCFD_Field(
        default=None,
        gt=0,
        title="Initial",
        description=(
            "CFL this ramp starts from, returned at its own start cycle. "
            "Leave unset to carry on from wherever the previous ramp left "
            "off."
        ),
    )
    start_cycle: int = zCFD_Field(
        default=1,
        ge=1,
        title="Start Cycle",
        description="First solve cycle this ramp acts on.",
    )
    end_cycle: Optional[int] = zCFD_Field(
        default=None,
        ge=1,
        title="End Cycle",
        description=(
            "Last solve cycle this ramp acts on. Unset runs to the end of the run."
        ),
    )


class ExponentialRamp(_RampBase):
    """Multiplies the CFL by ``factor`` each cycle in its window."""

    type: Literal["exponential"] = zCFD_Field(..., description=_RAMP_TYPE)
    factor: float = zCFD_Field(
        ...,
        ge=1.0,
        title="Factor",
        description="Multiplied into the CFL every cycle. Must be at least 1.",
    )
    max_allowed: float = zCFD_Field(
        default=1e6,
        gt=0,
        title="Max Allowed",
        description="Ceiling on the ramp itself, guarding against overflow.",
        advanced=True,
    )


class LinearRamp(_RampBase):
    """Adds ``increment`` to the CFL every cycle in its window."""

    type: Literal["linear"] = zCFD_Field(..., description=_RAMP_TYPE)
    increment: float = zCFD_Field(
        ...,
        title="Increment",
        description=(
            "Added to the CFL every cycle in the window. An increment of 0 "
            "holds the CFL steady."
        ),
    )


class SteppedLinearRamp(_RampBase):
    """Adds ``increment`` once every ``step_cycles`` cycles in its window."""

    type: Literal["stepped_linear"] = zCFD_Field(..., description=_RAMP_TYPE)
    increment: float = zCFD_Field(
        ...,
        title="Increment",
        description="Added to the CFL once per step.",
    )
    step_cycles: int = zCFD_Field(
        ...,
        gt=0,
        title="Step Cycles",
        description="Cycles between steps.",
    )


RampSpec = Annotated[
    Union[ExponentialRamp, LinearRamp, SteppedLinearRamp],
    Field(discriminator="type"),
]

RampCallable = Callable[..., Any]


def _accept_ramp_list_or_callable(v: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    """Pass a callable through; validate anything else as a list.

    Wrap, not before: skipping ``handler`` keeps the callable out of the
    list annotation, so a bad list entry does not also report "input should
    be callable".
    """
    if callable(v):
        return CFL_RAMP.check(v)
    if v is None or isinstance(v, (list, tuple)):
        return handler(v)
    raise ValueError(
        "cfl_ramp must be a list of ramp specifications, or a function "
        f"{CFL_RAMP.call()}. A single ramp is a list of one: "
        "[{'type': 'exponential', 'factor': 1.05, 'initial': 0.1}]."
    )


def build_cfl_ramp(spec: Any) -> Optional[RampCallable]:
    """Build the solver's ramp callable from a resolved ``cfl_ramp``.

    Takes the resolved dict form. A shape's non-``type`` fields are its
    builder's keyword arguments.
    """
    if spec is None or callable(spec):
        return spec

    from zutil.control.ramp import (
        exponential_ramp,
        linear_ramp,
        linear_stepped_ramp,
        sequence_ramps,
    )

    builders = {
        "exponential": exponential_ramp,
        "linear": linear_ramp,
        "stepped_linear": linear_stepped_ramp,
    }
    ramps = [
        builders[entry["type"]](
            **{key: value for key, value in entry.items() if key != "type"}
        )
        for entry in spec
    ]
    if not ramps:
        return None
    # Chained even for one: a ramp behaves the same however many it runs with.
    return sequence_ramps(*ramps)


class CFLModel(BaseModel):
    """CFL configuration model."""

    model_config = ConfigDict(extra="forbid")

    cfl: float = zCFD_Field(
        ..., title="CFL", gt=0, description="CFL number", updatable=True
    )
    cfl_turbulence: Optional[float] = zCFD_Field(
        default=None,
        gt=0,
        title="CFL Turbulence",
        description=(
            "Working CFL for the turbulence transport equations (the "
            "conserved-variable slots beyond the 5 mean-flow equations — "
            "SA: 1, SST: 2). The solver uses it directly as the pseudo-time "
            "CFL for those equations, not as a cap. "
            "Has no effect when no turbulence model is active (euler, "
            "laminar viscous, non-hybrid LES)."
        ),
        advanced=True,
        updatable=True,
    )

    # Consumed only on the coarse-level sweeps of the multigrid V-cycle
    # (solver.py:_advance_rk_on_level mesh_level>0, reached only from
    # _advance_multigrid) — without multigrid it is stored and never read.
    cfl_coarse: ConditionalOptional[  # type: ignore[valid-type]
        float, When({"solver.numerical_scheme.multigrid": (SET,)})
    ] = zCFD_Field(
        default=None,
        gt=0,
        title="CFL Coarse",
        description="CFL number for multigrid coarse levels",
        advanced=True,
        updatable=True,
    )
    # The list is the editable form and the only thing the JSON schema
    # describes. A callable is the escape hatch: the wrap validator returns
    # it without ever offering it to the list annotation, so it stays out of
    # both the schema and the error messages. `to_cpp_dict` serialises
    # either into the callable the solver invokes.
    cfl_ramp: Annotated[
        Optional[List[RampSpec]], WrapValidator(_accept_ramp_list_or_callable)
    ] = zCFD_Field(
        default=None,
        title="CFL Ramp",
        description=CFL_RAMP.description(
            "How the CFL climbs toward 'cfl'. A list of ramps, each naming its "
            "shape — one entry for a single ramp, several to run in order. "
            "Alternatively a function"
        )
        + ". Keys a returned dict leaves out keep their static value.",
        group="Ramp",
    )
    cfl_viscous_factor: Optional[float] = zCFD_Field(
        default=None,
        gt=0,
        title="CFL Viscous Factor",
        description="CFL viscous factor",
        advanced=True,
    )

    @field_serializer("cfl_ramp", mode="wrap")
    def _serialize_cfl_ramp(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        """Pass a ramp callable straight through, unchanged.

        The annotation describes only the list form, so the serializer built
        from it treats the callable escape hatch as a type mismatch: every
        model_dump of a deck with a callable ramp emits
        PydanticSerializationUnexpectedValue, and the enclosing solver union
        then reports a mismatch too. The callable is deliberately outside the
        annotation, so serialising it is this method's job. Lists still go
        through the annotation's own serializer.
        """
        if callable(value):
            return value
        return handler(value)


class EulerScheme(BaseModel):
    """First-order explicit pseudo-time scheme."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["euler"] = zCFD_Field(
        default="euler",
        title="Name",
        description="Pseudo-time integration scheme: euler = first-order explicit",
    )
    # Inert under this scheme: get_time_integrator() in runge_kutta.py reads
    # scheme['stage'] only when name == 'runge kutta'; every other name
    # short-circuits to SINGLE_STAGE.
    stage: Optional[Literal[1, 3, 4, 5, "rk third order tvd"]] = zCFD_Field(
        default=None,
        title="Stage",
        description=(
            "Has no effect under 'euler' (single-stage by construction). "
            "Accepted for backward compatibility with decks that carry this "
            "key as boilerplate; prefer omitting it."
        ),
        advanced=True,
    )


class RungeKuttaScheme(BaseModel):
    """Multi-stage explicit pseudo-time scheme."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["runge kutta"] = zCFD_Field(
        default="runge kutta",
        title="Name",
        description="Pseudo-time integration scheme: runge kutta = multi-stage explicit",
    )
    stage: Optional[Literal[1, 3, 4, 5, "rk third order tvd"]] = zCFD_Field(
        default=None,
        title="Stage",
        description=(
            "Runge-Kutta stage configuration. Constrained to the schemes the "
            "runtime actually supports — 1 (single stage), 3 or 'rk third "
            "order tvd' (three-stage TVD), 4 (four stage), or 5 (five stage)."
        ),
    )


class ImplicitEulerScheme(BaseModel):
    """First-order implicit pseudo-time scheme."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["implicit euler"] = zCFD_Field(
        default="implicit euler",
        title="Name",
        description="Pseudo-time integration scheme: implicit euler = first-order implicit",
    )
    # Inert under this scheme — see EulerScheme.stage.
    stage: Optional[Literal[1, 3, 4, 5, "rk third order tvd"]] = zCFD_Field(
        default=None,
        title="Stage",
        description=(
            "Has no effect under 'implicit euler' (single-stage by "
            "construction). Accepted for backward compatibility with decks "
            "that carry this key as boilerplate; prefer omitting it."
        ),
        advanced=True,
    )


class LUSGSScheme(BaseModel):
    """Matrix-free point-implicit LU-SGS/DP-LUR pseudo-time scheme."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["lu-sgs"] = zCFD_Field(
        default="lu-sgs",
        title="Name",
        description=(
            "Pseudo-time integration scheme: lu-sgs = matrix-free point-implicit "
            "LU-SGS/DP-LUR"
        ),
    )
    sub_iterations: Optional[int] = zCFD_Field(
        default=None,
        gt=0,
        title="Sub Iterations",
        description="Number of LU-SGS/DP-LUR sweeps per pseudo-time step",
        advanced=True,
    )
    over_relaxation: Optional[float] = zCFD_Field(
        default=None,
        gt=0.0,
        title="Over Relaxation",
        description="Diagonal over-relaxation factor for the LU-SGS/DP-LUR sweeps",
        advanced=True,
    )
    turbulence_solver: Optional[Literal["amg", "point-implicit"]] = zCFD_Field(
        default=None,
        title="Turbulence Solver",
        description=(
            "Linear solver for the turbulence equations under LU-SGS. 'amg' keeps the "
            "assembled AMG solve; 'point-implicit' uses the matrix-free sweep. The C++ "
            "default when unset is 'point-implicit'."
        ),
        advanced=True,
    )


class MFGMRESScheme(BaseModel):
    """Matrix-free GMRES on the exact analytic Jacobian (pseudo-time)."""

    model_config = ConfigDict(extra="forbid")

    name: ConditionalChoice[  # type: ignore[valid-type]
        Literal["mf-gmres"],
        {
            "mf-gmres": When(
                {
                    "solver.equations.turbulence.model": NotIn("sas"),
                    "solver.equations.turbulence.hybrid": NotIn("SAS"),
                }
            )
        },
    ] = zCFD_Field(
        default="mf-gmres",
        title="Name",
        description=(
            "Pseudo-time integration scheme: mf-gmres = restarted GMRES applying the "
            "exact analytic flux Jacobian matrix-free (no assembled mean-flow matrix)"
        ),
    )
    krylov_subspace: Optional[int] = zCFD_Field(
        default=None,
        gt=0,
        title="Krylov Subspace",
        description=(
            "Restart length m. The solver stores m+1 Krylov vectors, or 2m+1 with flexible, "
            "so this is the dominant storage cost; size it from the observed iteration count "
            "rather than padding it."
        ),
        advanced=True,
    )
    linear_tolerance: Optional[float] = zCFD_Field(
        default=None,
        gt=0.0,
        title="Linear Tolerance",
        description="Relative residual drop required of the linear solve each implicit step",
        advanced=True,
    )
    max_iterations: Optional[int] = zCFD_Field(
        default=None,
        gt=0,
        title="Max Iterations",
        description="Hard cap on total Krylov iterations (across restarts) per implicit step",
        advanced=True,
    )
    preconditioner: ConditionalChoice[  # type: ignore[valid-type]
        Optional[Literal["block jacobi", "lu-sgs", "lu-sgs nonlinear", "none"]],
        {"lu-sgs nonlinear": When({"<self>.flexible": (True,)})},
    ] = zCFD_Field(
        default=None,
        title="Preconditioner",
        description=(
            "'block jacobi' inverts the exact analytic 5x5 block diagonal. 'lu-sgs' uses "
            "linearised symmetric Gauss-Seidel sweeps. 'lu-sgs nonlinear' uses the stock "
            "unlinearised sweep and REQUIRES flexible=true, since it is not a linear "
            "operator and plain GMRES cannot converge with it."
        ),
        advanced=True,
    )
    preconditioner_sweeps: Optional[int] = zCFD_Field(
        default=None,
        gt=0,
        title="Preconditioner Sweeps",
        description="Symmetric Gauss-Seidel sweeps per preconditioner application (lu-sgs only)",
        advanced=True,
    )
    orthogonalisation: Optional[Literal["mgs", "cgs2"]] = zCFD_Field(
        default=None,
        title="Arnoldi Orthogonalisation",
        description=(
            "How the Krylov basis is orthogonalised. 'mgs' (default) is modified Gram-Schmidt, "
            "matching AMGX's gmres/fgmres and so directly comparable with the implicit euler "
            "benchmark. 'cgs2' is classical Gram-Schmidt with one reorthogonalisation pass and "
            "batched reductions: it reduces the reduction round trips per restart cycle from "
            "O(m^2) to O(m) at the cost of twice the vector work, which is only expected to pay "
            "at high rank counts where each reduction is a real collective."
        ),
        advanced=True,
    )
    preconditioner_over_relaxation: Optional[float] = zCFD_Field(
        default=None,
        gt=0.0,
        title="Preconditioner Over Relaxation",
        description=(
            "Diagonal over-relaxation for the lu-sgs preconditioner sweeps. Named separately "
            "from the lu-sgs solver's own 'over relaxation' because here it tunes a "
            "PRECONDITIONER, not the solve."
        ),
        advanced=True,
    )
    turbulence_solver: Optional[Literal["amg", "point-implicit"]] = zCFD_Field(
        default=None,
        title="Turbulence Solver",
        description=(
            "Linear solver for the turbulence equations. mf-gmres is MEAN-FLOW ONLY: the "
            "turbulence matrix is always assembled, and this chooses how it is solved. "
            "'point-implicit' uses matrix-free sweeps on it, 'amg' the assembled AMG solve. "
            "The C++ default when unset is 'point-implicit', except where the backend forces "
            "amg (a CUDA build running host-only, or HIP)."
        ),
        advanced=True,
    )
    turbulence_sweeps: Optional[int] = zCFD_Field(
        default=None,
        gt=0,
        title="Turbulence Sweeps",
        description=(
            "Sweeps for the point-implicit turbulence solve. Deliberately NOT called "
            "'sub iterations': that name belongs to the lu-sgs solver, and confusing it with "
            "'preconditioner sweeps' (which tunes the MEAN-FLOW preconditioner) would be easy."
        ),
        advanced=True,
    )
    flexible: Optional[bool] = zCFD_Field(
        default=None,
        title="Flexible (FGMRES)",
        description=(
            "Use FGMRES instead of right-preconditioned GMRES. Permits a preconditioner "
            "that varies between iterations, at the cost of a second set of m Krylov vectors."
        ),
        advanced=True,
    )


def _default_scheme_name(v: Any, default: str = "runge kutta") -> Any:
    """BeforeValidator: fill in the discriminator when a scheme dict omits 'name'.

    Preserves today's behaviour, where a bare ``scheme: {}`` (or a dict that
    only sets e.g. 'stage') implicitly meant 'runge kutta' — the flat
    ``TimeScheme.name`` field defaulted to it.  A discriminated union can't
    default a missing discriminator on its own, so this fills it in before
    the union picks a variant.
    """
    if isinstance(v, dict) and "name" not in v:
        return {**v, "name": default}
    return v


ExplicitTimeScheme = Union[EulerScheme, RungeKuttaScheme]
ImplicitTimeScheme = Union[ImplicitEulerScheme, LUSGSScheme, MFGMRESScheme]

TimeScheme = Annotated[
    Union[ExplicitTimeScheme, ImplicitTimeScheme],
    Field(discriminator="name"),
]

EXPLICIT_SCHEME_NAMES = tags_of(ExplicitTimeScheme, "name")
IMPLICIT_SCHEME_NAMES = tags_of(ImplicitTimeScheme, "name")


class GTSRungeKuttaScheme(BaseModel):
    """The three-stage TVD Runge-Kutta (SSP-RK3) physical time integrator for global time stepping."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["runge kutta"] = zCFD_Field(
        default="runge kutta",
        title="Name",
        description=(
            "Physical time integration scheme. Global time stepping admits "
            "only the three-stage TVD Runge-Kutta"
        ),
    )
    stage: Literal[3, "rk third order tvd"] = zCFD_Field(
        default=3,
        title="Stage",
        description=(
            "Runge-Kutta stage configuration: the three-stage TVD scheme "
            "(SSP-RK3), third order in physical time. Spelled either 3 or "
            "'rk third order tvd' — the same scheme either way"
        ),
    )


GTSTimeScheme = GTSRungeKuttaScheme

GTS_SCHEME_STAGES = (3, "rk third order tvd")


APPLIES_WITH_DUAL_TIME = {
    "solver.time_settings.kind": ("dual time stepping",),
}


class InnerConvergence(BaseModel):
    """Ends each real time step once its inner iteration has converged."""

    model_config = ConfigDict(extra="forbid")

    target_orders: float = zCFD_Field(
        default=1.0,
        updatable=True,
        title="Target Orders",
        description=(
            "Orders of magnitude the worst selected residual measure must fall "
            "within the current real time step before it is called converged: "
            "the InnerRatio diagnostic for the compressible solver, the "
            "reported residual itself (InnerDrop) for the incompressible "
            "solver. Scale free in both, so one value carries between cases. "
            "0 or less disables this test."
        ),
    )
    target_ratio: ConditionalOptional[  # type: ignore[valid-type]
        float, When({"solver.solver_settings.type": (UNSET, "compressible")})
    ] = zCFD_Field(
        default=None,
        ge=0.0,
        updatable=True,
        title="Target Ratio",
        description=(
            "Absolute floor on the InnerRatio diagnostic itself, ORed with "
            "'target_orders'. Compressible solver only. Not portable between "
            "cases; measure it from a run first. Unset or 0 disables this test."
        ),
        advanced=True,
    )
    min_cycles: int = zCFD_Field(
        default=5,
        ge=1,
        updatable=True,
        title="Min Cycles",
        description="Inner cycles that always run before the criterion is tested.",
    )
    check_frequency: int = zCFD_Field(
        default=5,
        ge=1,
        updatable=True,
        title="Check Frequency",
        description=(
            "Inner cycles between evaluations of the criterion. A check cycle "
            "is also a reporting cycle: the criterion is judged on the "
            "reported residuals."
        ),
    )
    variables: Union[Literal["mean flow", "all"], List[str]] = zCFD_Field(
        default="mean flow",
        updatable=True,
        title="Variables",
        description=(
            "Residuals the criterion is judged on: 'mean flow', 'all' (mean "
            "flow plus the turbulence transport equations), or an explicit "
            "list of residual names. What 'mean flow' selects follows the "
            "solver: compressible takes continuity, momentum and energy (rho, "
            "rhoV[0..2], rhoE); the incompressible coupled scheme takes "
            "momentum and pressure (rhoV[0..2], p) — pressure being the "
            "continuity constraint that governs its outer iteration; SIMPLE "
            "takes momentum alone, its reported pressure being the "
            "pressure-correction solve's own residual rather than a flow one. "
            "The incompressible solver reports no energy residual, so an "
            "energy-solving case cannot select one."
        ),
        advanced=True,
    )
    stall_window: int = zCFD_Field(
        default=10,
        ge=0,
        updatable=True,
        title="Stall Window",
        description=(
            "Give up when the reduction stops improving over this many checks: "
            "limiter switching can floor a residual norm no number of cycles "
            "will get below. 0 disables stall detection."
        ),
        advanced=True,
    )
    stall_tolerance: float = zCFD_Field(
        default=0.05,
        updatable=True,
        title="Stall Tolerance",
        description=(
            "Orders that must be gained across 'stall_window' checks for the "
            "inner iteration to count as still making progress."
        ),
        advanced=True,
    )
    start_real_time_step: int = zCFD_Field(
        default=0,
        ge=0,
        updatable=True,
        title="Start Real Time Step",
        description=(
            "Real time step from which the criterion may fire. A solution "
            "restarted from a steady state and left to develop — RANS into "
            "DES — looks converged per real time step while the unsteadiness "
            "is still small, so an early exit through that phase would cut the "
            "inner iteration short exactly while the instability that has to "
            "grow is being resolved."
        ),
        advanced=True,
    )


class ConvergenceControlBase(BaseModel):
    """Pacing elements shared by both solvers.

    ``scheme`` is deliberately not here: the two solvers march a pseudo-step
    towards convergence by different means, so each carries its own union
    (``TimeScheme`` for compressible, ``IncompressibleScheme`` for
    incompressible) and neither variant's names are reachable from the other.
    """

    model_config = ConfigDict(extra="forbid")

    cycles: int = zCFD_Field(
        ...,
        title="Cycles",
        gt=0,
        description="Number of convergence cycles (steady) or max pseudo-time cycles per step (unsteady)",
        updatable=True,
    )
    inner_convergence: ConditionalOptional[  # type: ignore[valid-type]
        InnerConvergence, When(APPLIES_WITH_DUAL_TIME)
    ] = zCFD_Field(
        default=None,
        title="Inner Convergence",
        description=(
            "Let the solver decide how many inner (pseudo-time) cycles each "
            "real time step needs, with 'cycles' acting as an upper bound. "
            "Unset runs the full 'cycles' every real time step."
        ),
    )


class StartupRamp(BaseModel):
    """Log-linear / linear startup ramp for the incompressible solver."""

    model_config = ConfigDict(extra="forbid")

    initial: float = zCFD_Field(
        ...,
        gt=0,
        title="Initial",
        description="Ramp value on the first pseudo cycle; the ramp eases to 1 from here",
        updatable=True,
    )
    cycles: float = zCFD_Field(
        default=300.0,
        gt=0,
        title="Cycles",
        description="Number of pseudo cycles over which the ramp reaches 1",
        updatable=True,
    )


class _IncompressibleSchemeBase(BaseModel):
    """Relaxation shared by every incompressible coupling scheme.

    Both factors are targets: ``relaxation_ramp``, its sibling on the
    convergence-control block, scales them on the way up through the first
    real-time step.
    """

    model_config = ConfigDict(extra="forbid")

    implicit_relax: float = zCFD_Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        title="Implicit Relaxation",
        description=(
            "Under-relaxation factor (alpha_u) on the implicit momentum, pressure "
            "and energy updates"
        ),
    )
    turbulence_relax: float = zCFD_Field(
        default=0.5,
        gt=0,
        le=1.0,
        title="Turbulence Relaxation",
        description=(
            "Under-relaxation factor on the turbulence transport update. Has no "
            "effect without a turbulence model (laminar viscous, non-hybrid LES)"
        ),
    )
    # Both schemes read it: a closed domain leaves pressure singular for the coupled
    # block-4 system as much as for the SIMPLE p' Poisson.
    pin_pressure: Optional[bool] = zCFD_Field(
        default=None,
        title="Pin Pressure",
        description=(
            "Pin the pressure at a reference cell with a large diagonal instead of the "
            "default constant-null-space + mean-removal gauge fix. Needed on a fully "
            "closed domain (no fixed-pressure boundary)"
        ),
        advanced=True,
    )


class CoupledScheme(_IncompressibleSchemeBase):
    """Single block-4 [u,v,w,p] system per pseudo-time step."""

    name: Literal["coupled"] = zCFD_Field(
        default="coupled",
        title="Name",
        description=(
            "Pressure-velocity coupling scheme: coupled = one block-4 [u,v,w,p] "
            "system solved per pseudo-time step"
        ),
    )


class SimpleScheme(_IncompressibleSchemeBase):
    """Segregated SIMPLE momentum predictor / pressure correction.

    The pressure knobs live here rather than beside ``implicit_relax``: they
    describe the pressure-correction solve, which the coupled scheme does not
    perform at all.
    """

    name: Literal["simple"] = zCFD_Field(
        default="simple",
        title="Name",
        description=(
            "Pressure-velocity coupling scheme: simple = segregated SIMPLE "
            "momentum predictor / pressure correction"
        ),
    )
    pressure_relax: Optional[float] = zCFD_Field(
        default=None,
        gt=0.0,
        le=1.0,
        title="Pressure Relax",
        description=(
            "Explicit pressure under-relaxation factor (alpha_p) for "
            "p += alpha_p * p'. The solver default is 0.3"
        ),
        advanced=True,
    )
    pressure_regularisation: Optional[float] = zCFD_Field(
        default=None,
        ge=0.0,
        title="Pressure Regularisation",
        description=(
            "Diagonal regularisation (diag += beta*|diag|) of the pressure-correction "
            "Poisson operator. Vanishes at convergence, so the steady solution is "
            "unbiased"
        ),
        advanced=True,
    )
    pressure_correction_limit: Optional[float] = zCFD_Field(
        default=None,
        gt=0.0,
        title="Pressure Correction Limit",
        description="Limit factor applied to the pressure correction",
        advanced=True,
    )


IncompressibleScheme = Annotated[
    Union[CoupledScheme, SimpleScheme],
    Field(discriminator="name"),
]


class IncompressibleMeshConvergenceControl(ConvergenceControlBase):
    """Per-mesh incompressible convergence override.

    Carries everything the incompressible solver reads per mesh, which is
    everything except ``scheme``: the coupling scheme selects the C++ solver
    class for the whole run (``storage.useSimple`` and the SimpleSolver /
    CoupledSolver choice are both made once), so it cannot vary by mesh and
    is absent here rather than accepted and ignored.
    """

    # Startup ramps that make an impulsive start gentler. Both are active only during the
    # first real-time step. Defaults reproduce the historical behaviour: a 100x viscosity
    # ramp over 300 cycles, and no under-relaxation ramp.
    viscosity_ramp: StartupRamp = zCFD_Field(
        default_factory=lambda: StartupRamp(initial=100.0, cycles=300.0),
        title="Viscosity Ramp",
        description=(
            "Scales the effective viscosity, decaying log-linearly from initial to 1, "
            "lowering the effective Reynolds number early so the start is more diffusive. "
            "initial <= 1 disables it"
        ),
        advanced=True,
    )
    relaxation_ramp: StartupRamp = zCFD_Field(
        default_factory=lambda: StartupRamp(initial=1.0, cycles=300.0),
        title="Relaxation Ramp",
        description=(
            "Scales the under-relaxation factors (alpha_u, alpha_p, turbulence relax), "
            "rising linearly from initial to 1, so the solver starts more relaxed. "
            "initial >= 1 (the default) disables it"
        ),
        advanced=True,
    )


class IncompressibleConvergenceControl(IncompressibleMeshConvergenceControl):
    """Solver-level incompressible convergence control."""

    # Always present in the dump: the C++ reads implicit_relax/turbulence_relax
    # from this dict on every incompressible run, coupled or SIMPLE.
    scheme: IncompressibleScheme = zCFD_Field(
        default_factory=CoupledScheme,
        title="Scheme",
        description=(
            "Pressure-velocity coupling scheme and the relaxation driving it. A "
            "scheme dict that omits name defaults to coupled"
        ),
    )

    @field_validator("scheme", mode="before")
    @classmethod
    def _fill_default_scheme_name(cls, v):
        return _default_scheme_name(v, "coupled")


def _as_cfl_model(v: Any) -> Any:
    """Accept `cfl: 2.0` as shorthand for `cfl: {cfl: 2.0}`."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return {"cfl": v}
    return v


class CompressibleConvergenceControl(ConvergenceControlBase):
    scheme: Optional[TimeScheme] = zCFD_Field(
        default=None,
        title="Scheme",
        description="Pseudo-time/Convergence integration scheme",
    )

    @field_validator("scheme", mode="before")
    @classmethod
    def _fill_default_scheme_name(cls, v):
        return _default_scheme_name(v)

    # A bare number is normalised to a CFLModel on the way in.
    cfl: Annotated[CFLModel, BeforeValidator(_as_cfl_model)] = zCFD_Field(
        default_factory=lambda: CFLModel(cfl=1.0),
        title="CFL",
        description=(
            "CFL configuration. A bare number is shorthand for the working "
            "CFL: `cfl: 2.0` means `cfl: {cfl: 2.0}`."
        ),
    )
