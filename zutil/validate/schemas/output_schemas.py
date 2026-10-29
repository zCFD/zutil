import difflib

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
)
from pydantic.json_schema import SkipJsonSchema
from typing import List, Optional, Literal
from zutil.validate.schemas.base_types import zCFD_Field
from .base_types import Vector3d
from .callables import FORCE_TRANSFORM, ForceTransformFunction
from .output_variables import (
    AVERAGED_VARIABLES,
    CONTEXTS,
    get_valid_variables_for_equation_type,
)

# Output format types
output_format_types = Literal["none", "vtk", "ensight", "fwensight", "native"]


def _closest_variable(name: str, valid_variables: List[str]) -> Optional[str]:
    """The valid variable `name` was probably meant to be, or None.

    A misspelled variable used to cost nothing at validation time and was then
    dropped by the solver with a log warning, so the case ran and wrote no such
    field. The two spellings that go wrong that way are a name written with
    spaces where the solver has underscores (`gauge pressure`) and a name
    written in the wrong case, so both are checked before falling back to
    a general near-match. Near-matches are ranked by the fewest single-letter
    edits from what was written; between equally near names, one that
    abbreviates what was written (`walldist` for `walldistance`) wins over one
    that adds to it (`walldistancezone`), which names another field.
    """
    folded = {v.replace(" ", "_").lower(): v for v in valid_variables}
    written = name.replace(" ", "_").lower()
    respelled = folded.get(written)
    if respelled is not None:
        return respelled

    matches = difflib.get_close_matches(
        name, valid_variables, n=len(valid_variables), cutoff=0.8
    )
    if not matches:
        return None

    def rank(candidate: str) -> tuple:
        spelled = candidate.replace(" ", "_").lower()
        return _edits(spelled, written), not _abbreviates(spelled, written)

    return min(matches, key=rank)


def _edits(a: str, b: str) -> int:
    """The Levenshtein distance between `a` and `b`."""
    previous = list(range(len(b) + 1))
    for i, letter in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (letter != other),
                )
            )
        previous = current
    return previous[-1]


def _abbreviates(short: str, long: str) -> bool:
    """Whether `short` is `long` with letters left out, starting on the same letter."""
    if not short or not long or short[0] != long[0]:
        return False
    rest = iter(long)
    return all(letter in rest for letter in short)


def validate_output_variables(
    variables: List[str],
    equation_type: str,
    variable_context: str,
    turbulence_model: str = None,
    has_mesh_velocity: bool = False,
    has_fsi: bool = False,
) -> List[str]:
    """
    Validate that all variables in the list are appropriate for the given equation type.

    Args:
        variables: List of variable names to validate
        equation_type: The equation type (e.g., 'euler', 'viscous', 'rans')
        variable_context: Either 'volume' or 'surface'
        turbulence_model: Optional turbulence model for RANS equations
        has_mesh_velocity: Whether a mesh has 'motion' or an FSI boundary
        has_fsi: Whether a boundary carries an FSI coupling

    Returns:
        The validated list of variables

    Raises:
        ValueError: If any variables are not valid for the equation type
    """
    if not variables:
        return variables

    valid_variables = get_valid_variables_for_equation_type(
        equation_type, variable_context, turbulence_model, has_mesh_velocity, has_fsi
    )
    invalid_variables = [var for var in variables if var not in valid_variables]

    if invalid_variables:
        message = (
            f"Variables {invalid_variables} are not valid for equation type "
            f"'{equation_type}' in {variable_context} context."
        )
        suggestions = [
            (var, suggestion)
            for var, suggestion in (
                (var, _closest_variable(var, valid_variables))
                for var in invalid_variables
            )
            if suggestion is not None
        ]
        if suggestions:
            message += (
                " Did you mean "
                + ", ".join(
                    f"'{var}' -> '{suggestion}'" for var, suggestion in suggestions
                )
                + "?"
            )
        message += f" Valid variables include: {valid_variables[:10]}..."
        raise ValueError(message)

    return variables


class FrequencyModel(BaseModel):
    """Model for specifying output frequency."""

    model_config = ConfigDict(extra="forbid")

    surface_data: int = zCFD_Field(
        int(1e8),
        ge=1,
        title="Surface Data",
        description="Frequency for surface variable output",
        updatable=True,
    )
    volume_data: int = zCFD_Field(
        int(1e8),
        ge=1,
        title="Volume Data",
        description="Frequency for volume variable output",
        updatable=True,
    )
    volume_interpolate: int = zCFD_Field(
        int(1e8),
        ge=1,
        title="Volume Interpolate",
        description="Frequency for volume data interpolation",
        advanced=True,
        updatable=True,
    )
    fwh_interpolate: int = zCFD_Field(
        int(1e8),
        ge=1,
        title="FWH Interpolate",
        description="Frequency for FWH surface interpolation",
        advanced=True,
        updatable=True,
    )
    fwh_wall_data: int = zCFD_Field(
        int(1e8),
        ge=1,
        title="FWH Wall Data",
        description="Frequency for FWH wall data output",
        advanced=True,
        updatable=True,
    )
    checkpoint: int = zCFD_Field(
        int(1e8),
        ge=1,
        title="Checkpoint",
        description="Frequency for writing checkpoint/restart files",
        updatable=True,
    )

    volume_data_start: int = zCFD_Field(
        1,
        ge=0,
        title="Volume Data Start",
        description="Cycle to start volume data output",
        advanced=True,
    )
    surface_data_start: int = zCFD_Field(
        1,
        ge=0,
        title="Surface Data Start",
        description="Cycle to start surface data output",
        advanced=True,
    )
    volume_interpolate_start: int = zCFD_Field(
        1,
        ge=0,
        title="Volume Interpolate Start",
        description="Cycle to start volume data interpolation",
        advanced=True,
    )
    checkpoint_start: int = zCFD_Field(
        1,
        ge=0,
        title="Checkpoint Start",
        description="Cycle to start writing checkpoint/restart files",
        advanced=True,
    )
    fwh_interpolate_start: int = zCFD_Field(
        1,
        ge=0,
        title="FWH Interpolate Start",
        description="Cycle to start FWH surface interpolation",
        advanced=True,
    )
    fwh_wall_data_start: int = zCFD_Field(
        1,
        ge=0,
        title="FWH Wall Data Start",
        description="Cycle to start FWH wall data output",
        advanced=True,
    )


class SolutionModel(BaseModel):
    """Solution output configuration."""

    model_config = ConfigDict(extra="forbid")

    frequency: FrequencyModel = zCFD_Field(
        default_factory=lambda: FrequencyModel(),
        title="Frequency",
        description="Output frequency (every N cycles)",
    )

    @field_validator("frequency", mode="before")
    @classmethod
    def validate_frequency(cls, v):
        """Coerce int to FrequencyModel so C++ always receives a dict."""
        if isinstance(v, FrequencyModel):
            # Already-constructed instance (e.g. re-validating a model
            # built elsewhere) - pass through unchanged.
            return v

        if isinstance(v, dict):
            return FrequencyModel.model_validate(v)

        if isinstance(v, int):
            if v < 1:
                raise ValueError("Frequency must be >= 1")
            # Expand plain int to full FrequencyModel for C++ compatibility
            return FrequencyModel(
                surface_data=v,
                volume_data=v,
                volume_interpolate=int(1e8),
                fwh_interpolate=int(1e8),
                fwh_wall_data=int(1e8),
                checkpoint=v,
            )

        raise ValueError(
            f"Frequency must be either an integer or a dictionary, got {type(v)}"
        )

    format: Optional[output_format_types] = zCFD_Field(
        default="vtk", title="Format", description="Output file format"
    )

    # Variable specification
    surface_variables: Optional[List[str]] = zCFD_Field(
        default=None,
        title="Surface Variables",
        description="Variables to output on surfaces",
        updatable=True,
    )
    volume_variables: Optional[List[str]] = zCFD_Field(
        default=None,
        title="Volume Variables",
        description="Variables to output in volume",
        updatable=True,
    )

    scripts: Optional[List[str]] = zCFD_Field(
        default=None,
        title="Catalyst Scripts",
        description=(
            "ParaView Catalyst Python scripts, run on the volume solution each time "
            "it is written. Paths are passed to Catalyst as written, so a relative "
            "path is taken from the directory the run is started in."
        ),
        advanced=True,
    )

    fwh_interpolate: Optional[list[str]] = zCFD_Field(
        default=None,
        title="FWH Interpolate",
        description="List of stl files for FWH surface interpolation",
        advanced=True,
    )

    fwh_wall_data: bool = zCFD_Field(
        default=False,
        title="FWH Wall Data",
        description=(
            "Record impermeable FWH data on every wall: the pressure and, for viscous "
            "equations, the wall shear stress. On a mesh that overset hole cutting can "
            "blank, each face is also flagged active or inactive, so a wall covered by "
            "another mesh does not radiate."
        ),
        advanced=True,
    )

    # Output options
    write_restart: Optional[bool] = zCFD_Field(
        default=True,
        title="Write Restart",
        description="Write restart files",
        advanced=True,
    )
    binary: Optional[bool] = zCFD_Field(
        default=True,
        title="Binary",
        description="Use binary format",
        advanced=True,
    )

    # Precision
    precision: Optional[Literal["single", "double"]] = zCFD_Field(
        default="single",
        title="Precision",
        description="Output precision",
        advanced=True,
    )
    global_node_ids: bool = zCFD_Field(
        default=True,
        title="Global Node IDs",
        description=(
            "Retain the local->global node map so VTKHDF output can stitch partitions "
            "together without a clean-to-grid filter"
        ),
        advanced=True,
    )
    output_solver_cycle: bool = zCFD_Field(
        default=False,
        title="Output Solver Cycle",
        description=(
            "Stamp VTKHDF output steps with the solver cycle number instead of "
            "physical time. Every step of a steady run carries the same physical "
            "time, so the default leaves the appended steps indistinguishable to "
            "a viewer; the cycle number makes them steppable"
        ),
        advanced=True,
    )
    ghost_cells: bool = zCFD_Field(
        default=True,
        title="Ghost Cells",
        description=(
            "Emit a one-cell-deep ghost layer in the VTKHDF volume output: the partition "
            "halo cells the solver already exchanges, marked DUPLICATECELL, so filters that "
            "interpolate cell data to points (isosurfaces, for instance) are correct across "
            "partition interfaces instead of showing a seam"
        ),
        advanced=True,
    )


class ForceModel(BaseModel):
    """Force calculation configuration."""

    model_config = ConfigDict(extra="forbid")

    zones: List[int] = zCFD_Field(
        ..., title="Zones", description="Zone IDs for force calculation"
    )
    name: Optional[str] = zCFD_Field(
        default=None, title="Name", description="Name of the force component"
    )

    # Reference values
    reference_area: float = zCFD_Field(
        default=1.0,
        gt=0,
        title="Reference Area",
        description="Reference area for force coefficients",
    )
    reference_length: float = zCFD_Field(
        default=1.0,
        gt=0,
        title="Reference Length",
        description="Reference length for moment coefficients",
    )
    reference_pressure: float = zCFD_Field(
        default=0.0,
        ge=0,
        title="Reference Pressure",
        description="Reference pressure for coefficients",
    )
    transform: SkipJsonSchema[Optional[ForceTransformFunction]] = zCFD_Field(
        default=None,
        title="Transform",
        description=FORCE_TRANSFORM.description(
            "Function applied to each force and moment vector before it is reported"
        ),
    )
    reference_point: Vector3d = zCFD_Field(
        default=[0.0, 0.0, 0.0],
        title="Reference Point",
        description="Reference point for moment calculations",
    )

    # Reporting options
    dimensional: bool = zCFD_Field(
        default=False,
        title="Dimensional",
        description="Report dimensional forces (not coefficients)",
    )
    inertial_frame: bool = zCFD_Field(
        default=True,
        title="Inertial Frame",
        description="Report forces in inertial frame",
    )


class MonitorModel(BaseModel):
    """Monitor point configuration."""

    model_config = ConfigDict(extra="forbid")

    point: Vector3d = zCFD_Field(
        ..., title="Point", description="Monitor point coordinates [x, y, z]"
    )
    variables: List[str] = zCFD_Field(
        ..., title="Variables", description="Variables to monitor at this point"
    )
    name: Optional[str] = zCFD_Field(
        default=None, title="Name", description="Monitor point name"
    )


class MassFlowModel(BaseModel):
    """Mass flow calculation configuration."""

    model_config = ConfigDict(extra="forbid")

    zones: List[int] = zCFD_Field(
        ..., title="Zones", description="Zone IDs for mass flow calculation"
    )


class ReportModel(BaseModel):
    """Report output configuration."""

    model_config = ConfigDict(extra="forbid")

    frequency: int = zCFD_Field(
        ...,
        title="Frequency",
        ge=1,
        description="Report frequency (every N cycles)",
        updatable=True,
    )

    # Variables to report
    variables: Optional[List[str]] = zCFD_Field(
        default=None, title="Variables", description="Variables to include in reports"
    )

    # Report types
    monitor: Optional[dict[str, MonitorModel]] = zCFD_Field(
        default=None, title="Monitor", description="Monitor point configuration"
    )
    forces: Optional[dict[str, ForceModel]] = zCFD_Field(
        default=None, title="Forces", description="Force calculation zones"
    )
    mass_flow: Optional[dict[str, MassFlowModel]] = zCFD_Field(
        default=None, title="Mass Flow", description="Mass flow calculation zones"
    )
    scale_residuals_by_volume: bool = zCFD_Field(
        default=False,
        title="Scale Residuals by Volume",
        description="Scale residuals by cell volume in reports",
    )
    residual_statistics: bool = zCFD_Field(
        default=True,
        advanced=True,
        title="Residual Statistics",
        description=(
            "Add the infinity norm of the true residual, its RMS, the participation ratio Neff "
            "and the worst cell's share of the total to the report, four columns per equation. "
            "Neff is the number of cells the residual is effectively spread over, so a small "
            "value against a large mesh means the field is converged everywhere except a handful "
            "of cells; the resshare_* output variables show which. On by default; set it False to "
            "save an extra pass over the residual field and two MPI reductions per report."
        ),
    )


class OutputModel(BaseModel):
    """
    Complete output configuration for zCFD simulations.

    Controls solution file output, reporting, monitoring, and post-processing
    data generation during the simulation.
    """

    model_config = ConfigDict(extra="forbid")

    # Core output components
    solution: SolutionModel = zCFD_Field(
        ..., title="Solution", description="Solution file output settings"
    )
    report: ReportModel = zCFD_Field(
        ..., title="Report", description="Report and monitoring settings"
    )

    # Global output settings
    compute_average_and_rms: bool = zCFD_Field(
        default=False,
        title="Compute Average and RMS",
        description="Compute time-averaged and RMS statistics",
        updatable=True,
    )
    average_start_time_cycle: int = zCFD_Field(
        default=0,
        ge=0,
        title="Average Start Time Cycle",
        description="Cycle to start averaging",
        updatable=True,
    )

    # File naming
    output_directory: Optional[str] = zCFD_Field(
        default=None, title="Output Directory", description="Output directory path"
    )


def _validate_output_variables(
    output_settings, equation_type: str, equation_config=None, full_model=None
):
    """
    Validate that output variables are appropriate for the given equation type.

    Args:
        output_settings: The output settings to validate
        equation_type: Type of equations ('euler', 'viscous', 'RANS', etc.)
        equation_config: Configuration object for the equations (optional)
        full_model: Full model for context (optional)

    Raises:
        ValueError: If output variables are invalid for the equation type
    """

    # Helper to access attributes or dict keys.
    def get_val(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    # Extract turbulence model information for RANS equations
    turbulence_model = None
    if equation_config is not None and equation_type.endswith("rans"):
        turbulence = get_val(equation_config, "turbulence")
        if turbulence is not None:
            turbulence_model = get_val(turbulence, "model")

    def _block_has_fsi(block) -> bool:
        """Whether one mesh block has an FSI boundary."""
        bcs = get_val(block, "boundary_conditions")
        return isinstance(bcs, dict) and any(
            get_val(bc_config, "fsi") is not None for bc_config in bcs.values()
        )

    def _block_moves(block) -> bool:
        """Whether one mesh block has 'motion' in its 'transforms'."""
        transforms = get_val(block, "transforms")
        return transforms is not None and get_val(transforms, "motion") is not None

    blocks = []
    if full_model:
        # Zones and boundary conditions belong to a mesh
        blocks.append(full_model)
        models = get_val(full_model, "model")
        if isinstance(models, dict):
            blocks.extend(models.values())
    has_fsi = any(_block_has_fsi(block) for block in blocks)
    has_mesh_velocity = has_fsi or any(_block_moves(block) for block in blocks)

    solution = get_val(output_settings, "solution")
    if solution is not None:
        for context in ("volume", "surface"):
            variables = get_val(solution, f"{context}_variables")
            if variables:
                validate_output_variables(
                    variables,
                    equation_type,
                    context,
                    turbulence_model,
                    has_mesh_velocity,
                    has_fsi,
                )

    # Validate monitor-point variables (report.monitor.*.variables). Legacy
    # checked these against the same volume-variable list used for
    # solution.volume_variables (a monitor samples volume fields at a point).
    report = get_val(output_settings, "report")
    if report is not None:
        monitor = get_val(report, "monitor")
        if monitor:
            for monitor_name, monitor_model in monitor.items():
                monitor_variables = get_val(monitor_model, "variables")
                if monitor_variables:
                    try:
                        validate_output_variables(
                            monitor_variables,
                            equation_type,
                            "volume",
                            turbulence_model,
                            has_mesh_velocity,
                            has_fsi,
                        )
                    except ValueError as e:
                        raise ValueError(f"monitor '{monitor_name}': {e}")

    if not get_val(output_settings, "compute_average_and_rms"):
        lists = (
            [get_val(solution, f"{c}_variables") for c in CONTEXTS] if solution else []
        )
        monitor = get_val(report, "monitor") if report else None
        lists += [get_val(m, "variables") for m in (monitor or {}).values()]
        requested = [
            n for names in lists for n in names or () if n in AVERAGED_VARIABLES
        ]
        if requested:
            raise ValueError(
                f"Requesting {', '.join(dict.fromkeys(requested))} needs 'compute average and rms' "
                "set to true in these output settings."
            )
