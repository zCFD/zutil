"""
Solver Settings — structural, partition, and system configuration.

Contains the properties model for solver system-level settings like
partitioner, linear solver options, and safe mode. These are distinct
from the full SolverBlockModel which composes the complete solver block.
"""

from pydantic import BaseModel, ConfigDict
from pydantic.json_schema import SkipJsonSchema
from typing import Annotated, Literal, Dict
from zutil.validate.schemas.base_types import (
    Advises,
    ConditionalOptional,
    When,
    zCFD_Field,
)


solver_types = Literal["compressible", "incompressible"]

_METIS_IS_DEVELOPMENT = Advises(
    When({"<self>.partitioner": ("metis",)}),
    "'partitioner' is 'metis', a development setting available only in research "
    "versions of zCFD. Release versions partition with 'scotch' instead: remove "
    "'partitioner' to use it.",
)


class LinearSolverOptions(BaseModel):
    """PETSc / AMGX linear solver options passed to C++."""

    model_config = ConfigDict(extra="forbid")

    flow: Dict[str, str] = zCFD_Field(
        default_factory=lambda: {
            "-ksp_min_it": "2",
            "-ksp_type": "fgmres",
            "-ksp_rtol": "1.0e-3",
            "-ksp_converged_reason": "",
        },
        title="Flow",
        description="PETSc options for the mean flow linear solver",
    )
    turbulence: Dict[str, str] = zCFD_Field(
        default_factory=lambda: {
            "-ksp_min_it": "2",
            "-ksp_type": "fgmres",
            "-ksp_rtol": "1.0e-5",
            "-ksp_converged_reason": "",
        },
        title="Turbulence",
        description="PETSc options for the turbulence linear solver",
    )
    rbf: Dict[str, str] = zCFD_Field(
        default_factory=lambda: {
            "-ksp_type": "fgmres",
            "-ksp_rtol": "1.0e-5",
        },
        title="RBF",
        description="PETSc options for the RBF linear solver",
    )
    double_precision: bool = zCFD_Field(
        default=False,
        title="Double Precision",
        description="Enable double precision for linear solver",
    )


class IncompressibleLinearSolverOptions(LinearSolverOptions):
    """Incompressible solver defaults, following simpleFoam's fvSolution."""

    flow: Dict[str, str] = zCFD_Field(
        default_factory=lambda: {
            "-ksp_min_it": "2",
            "-ksp_type": "fgmres",
            "-ksp_rtol": "1.0e-7",
            "-ksp_atol": "1.0e-16",
            "-ksp_max_it": "50",
            "-pc_type": "bjacobi",
            "-sub_pc_type": "ilu",
            "-ksp_converged_reason": "",
        },
        title="Flow",
        description=(
            "PETSc options for the coupled (block-4) velocity-pressure solver. "
            "Unlike the segregated equations this solve IS the step, so it keeps "
            "a tight tolerance; simpleFoam has no counterpart to it"
        ),
    )
    turbulence: Dict[str, str] = zCFD_Field(
        default_factory=lambda: {
            "-ksp_min_it": "1",
            "-ksp_type": "bcgs",
            "-ksp_rtol": "0.1",
            "-ksp_atol": "1.0e-16",
            "-ksp_max_it": "50",
            "-pc_type": "bjacobi",
            "-sub_pc_type": "ilu",
            "-ksp_converged_reason": "",
        },
        title="Turbulence",
        description=(
            "PETSc options for the turbulence (and energy) transport solvers. "
            "simpleFoam's k/omega/epsilon/nuTilda entry: a cheap preconditioned "
            "Krylov solve to relTol 0.1, since the outer iteration does the rest"
        ),
    )
    rbf: Dict[str, str] = zCFD_Field(
        default_factory=lambda: {
            "-ksp_type": "fgmres",
            "-ksp_rtol": "1.0e-5",
        },
        title="RBF",
    )
    double_precision: bool = zCFD_Field(
        default=True,
        title="Double Precision",
    )
    pressure_correction: ConditionalOptional[  # type: ignore[valid-type]
        Dict[str, str], When({"solver.convergence_control.scheme.name": ("simple",)})
    ] = zCFD_Field(
        default_factory=lambda: {
            "-ksp_min_it": "1",
            "-ksp_type": "cg",
            "-ksp_rtol": "0.01",
            "-ksp_atol": "1.0e-12",
            "-ksp_max_it": "100",
            "-pc_type": "hypre",
            "-pc_hypre_type": "boomeramg",
            "-zcfd_pc_reuse_cycles": "20",
            "-zcfd_constant_nullspace": "1",
            "-ksp_converged_reason": "",
        },
        title="Pressure Correction",
        description=(
            "PETSc options for the SIMPLE pressure-correction (block-1 Poisson) solver. "
            "simpleFoam's p entry: algebraic multigrid, solved to one or two orders"
        ),
        advanced=True,
    )
    momentum: ConditionalOptional[  # type: ignore[valid-type]
        Dict[str, str], When({"solver.convergence_control.scheme.name": ("simple",)})
    ] = zCFD_Field(
        default_factory=lambda: {
            "-ksp_min_it": "1",
            "-ksp_type": "bcgs",
            "-ksp_rtol": "0.1",
            "-ksp_atol": "1.0e-12",
            "-ksp_max_it": "50",
            "-pc_type": "bjacobi",
            "-sub_pc_type": "ilu",
            "-ksp_converged_reason": "",
        },
        title="Momentum",
        description=(
            "PETSc options for the SIMPLE momentum predictor (block-3) solver. "
            "simpleFoam's U entry: BiCGStab + ILU to relTol 0.1. The under-relaxed "
            "momentum matrix is strongly diagonally dominant, so this is a few iterations"
        ),
        advanced=True,
    )


class CellReorderingOptions(BaseModel):
    """Cell/face reordering for cache locality and colouring balance.

    ``sfc`` and ``face_sort`` are on by default, as they have been since
    #5568 enabled them. The C++ reads (Mesh.cxx) fall back to off when a key
    is missing, but no deck reaches the solver without these defaults
    materialised, so the defaults here are what every case actually runs.
    """

    model_config = ConfigDict(extra="forbid")

    rcm: bool = zCFD_Field(
        default=False,
        title="RCM",
        description="Reverse Cuthill-McKee bandwidth-reducing cell reordering",
        advanced=True,
    )
    sloan: bool = zCFD_Field(
        default=False,
        title="Sloan",
        description="Sloan profile-reducing cell reordering",
        advanced=True,
    )
    sfc: bool = zCFD_Field(
        default=True,
        title="SFC",
        description="Space-filling-curve cell reordering",
        advanced=True,
    )
    sfc_curve: Literal["hilbert", "morton"] = zCFD_Field(
        default="hilbert",
        title="SFC Curve",
        description="Space-filling curve used when sfc is enabled",
        advanced=True,
    )
    face_sort: bool = zCFD_Field(
        default=True,
        title="Face Sort",
        description="Sort faces within each colour by their lowest cell index",
        advanced=True,
    )


class SolverSettingsPropertiesModel(BaseModel):
    """
    Settings specifically dealing with structural, partition, and system definitions.

    Deliberately excluded: the mesh-quality check in Mesh::checkMesh still reads
    an optional "reset_face_area" value from solver_settings that let a deck
    either force degenerate (near-zero-area) faces to be reset outright (bool
    form) or reset only faces below a numeric area threshold (numeric form).
    It was evaluated for this schema and left out on purpose: no current
    control deck sets it, and mesh-quality problems in practice are handled
    upstream (mesh generation/cleanup) rather than by patching face areas at
    load time. The C++ read is guarded (skips the reset entirely when the key
    is absent), so omitting the field here does not change behaviour for any
    existing deck.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    type: solver_types = zCFD_Field(
        title="Type", description="The zCFD solver to be used"
    )
    partitioner: Annotated[
        SkipJsonSchema[Literal["metis", "scotch"]], _METIS_IS_DEVELOPMENT
    ] = zCFD_Field(
        default="scotch",
        title="Partitioner",
        description="Mesh partitioner for parallel decomposition",
        hidden=True,
    )
    linear_solver_options: LinearSolverOptions = zCFD_Field(
        default_factory=LinearSolverOptions,
        title="Linear Solver Options",
        description="Linear solver options passed to C++.",
        advanced=True,
    )
    solution_smooth_cycles: int = zCFD_Field(
        default=0,
        title="Solution Smooth Cycles",
        description="Number of Laplace smoothing cycles for mapped solutions.",
        advanced=True,
    )
    safe: bool = zCFD_Field(
        default=False,
        title="Safe",
        description="Enable safe mode for extra checks and diagnostics.",
        updatable=True,
    )
    cell_reordering: CellReorderingOptions = zCFD_Field(
        default_factory=CellReorderingOptions,
        title="Cell Reordering",
        description="Cell and face reordering options for cache locality and colouring balance.",
        advanced=True,
    )
    mesh_quality_remediation_angle_threshold: ConditionalOptional[  # type: ignore[valid-type]
        float, When({"solver.solver_settings.type": ("compressible",)})
    ] = zCFD_Field(
        default=None,
        gt=0,
        le=180,
        title="Mesh Quality Remediation Angle Threshold",
        description=(
            "Non-orthogonality angle in degrees above which an interior face uses a "
            "first-order convective flux. Non-orthogonality is the angle between the "
            "line joining the two cell centres and the face normal. Unset leaves "
            "every face at the scheme's order."
        ),
        advanced=True,
    )


class CompressibleSolverSettingsPropertiesModel(SolverSettingsPropertiesModel):
    type: Literal["compressible"] = "compressible"


class IncompressibleSolverSettingsPropertiesModel(SolverSettingsPropertiesModel):
    type: Literal["incompressible"] = "incompressible"
    linear_solver_options: IncompressibleLinearSolverOptions = zCFD_Field(
        default_factory=IncompressibleLinearSolverOptions,
        title="Linear Solver Options",
        description="Linear solver options for incompressible solver (tighter defaults).",
    )
