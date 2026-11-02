"""
Output variables available to zCFD, declared once as `zCFDOutputField`s.

Every name the solver will write is one entry in `OUTPUT_FIELDS`, carrying
everything known about it in one place: which physics tier makes it available
in volume and in surface output, what family it belongs to, what it means and
whether it is a short alias of a longer name. Availability, documentation and
UI grouping are therefore three views of one declaration rather than three
lists that can drift apart.

Availability is tiered: an equation type offers the tiers it inherits, so
`euler` gets base + euler, `viscous` adds the viscous tier, and `rans`/`les`
add the LES and RANS tiers (both end up with the same set). `transition`
fields are offered on top of those when the turbulence model solves for them,
`mesh velocity` fields only when the case has mesh velocity (a mesh with
'motion', moving or as a frame, or an FSI boundary), and `modal` fields only when it has an
FSI boundary.

The legacy `*_VARIABLES` / `*_ADDITIONS` frozensets below are derived views of
the registry, kept so existing importers need no change.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =============================================================================
# CATEGORIES — the families a field is grouped under, in display order
# =============================================================================


@dataclass(frozen=True)
class OutputCategory:
    """One family of output variables, as offered to a user choosing them."""

    key: str
    label: str
    hint: str


OUTPUT_CATEGORIES: Tuple[OutputCategory, ...] = (
    OutputCategory("primitive", "Flow", "Primitive solution variables"),
    OutputCategory("derived", "Derived", "Quantities computed from the solution"),
    OutputCategory("gradient", "Gradients", "Gradient vectors and tensors"),
    OutputCategory("turbulence", "Turbulence", "Turbulence model quantities"),
    OutputCategory("wall", "Wall", "Wall and boundary-layer quantities"),
    OutputCategory(
        "forces", "Forces & moments", "Per-face force and moment contributions"
    ),
    OutputCategory("mesh", "Mesh & geometry", "Cell and zone geometry"),
    OutputCategory("solver", "Solver", "Residuals, limiters and numerical diagnostics"),
    OutputCategory("overset", "Overset", "Overset data"),
    OutputCategory("fsi", "FSI & modal", "Modal deformation and mesh motion"),
    OutputCategory("other", "Other", "Undocumented"),
)

CATEGORIES_BY_KEY: Dict[str, OutputCategory] = {c.key: c for c in OUTPUT_CATEGORIES}


# =============================================================================
# TIERS — what makes a field available, and to which equation types
# =============================================================================

# "transition" is gated on the turbulence model rather than the equation type
TIERS = (
    "base",
    "euler",
    "viscous",
    "les",
    "rans",
    "transition",
    "mesh velocity",
    "modal",
)

#: Turbulence models that solve the transition equations, and so offer the
#: "transition" tier on top of whatever the equation type provides.
TRANSITION_TURBULENCE_MODELS = ("sst-transition",)

# Equation type -> the tiers it inherits. `les` and `rans` are deliberately
# identical: LES-tier fields are offered to both.
TIERS_BY_EQUATION: Dict[str, Tuple[str, ...]] = {
    "euler": ("base", "euler"),
    "viscous": ("base", "euler", "viscous"),
    "rans": ("base", "euler", "viscous", "les", "rans"),
    "les": ("base", "euler", "viscous", "les", "rans"),
    "incomp viscous": ("base", "euler", "viscous"),
    "incomp rans": ("base", "euler", "viscous", "les", "rans"),
    "incomp les": ("base", "euler", "viscous", "les", "rans"),
}

CONTEXTS = ("volume", "surface")


@dataclass(frozen=True)
class zCFDOutputField:
    """One variable zCFD can write, and everything known about it.

    `volume_tier` / `surface_tier` say what makes the field available in that
    context, or are None where it is not written there at all. The two differ
    for a handful of fields — `eddy`, for instance, needs an LES or RANS model
    to have a meaning in the volume but is written on viscous walls.
    """

    name: str
    category: str
    description: str = ""
    #: Canonical field this is a short spelling of, e.g. `p` -> `pressure`.
    alias_of: Optional[str] = None
    volume_tier: Optional[str] = None
    surface_tier: Optional[str] = None

    def tier_in(self, context: str) -> Optional[str]:
        """The tier offering this field in `context`, or None if never."""
        return self.volume_tier if context == "volume" else self.surface_tier

    def is_available(
        self,
        context: str,
        tiers,
        has_mesh_velocity: bool = False,
        has_fsi: bool = False,
    ) -> bool:
        """Whether an equation type offering `tiers` writes this field."""
        tier = self.tier_in(context)
        if tier is None:
            return False
        if tier == "mesh velocity":
            return has_mesh_velocity or has_fsi
        if tier == "modal":
            return has_fsi
        return tier in tiers

    def as_dict(self) -> dict:
        """The field as served to clients; alias_of omitted when it is None."""
        payload = {"category": self.category, "description": self.description}
        if self.alias_of is not None:
            payload["alias_of"] = self.alias_of
        return payload


def _base(name, category, description="", alias_of=None) -> zCFDOutputField:
    """Available to every equation type, in both contexts."""
    return zCFDOutputField(name, category, description, alias_of, "base", "base")


def _both(name, category, tier, description="", alias_of=None) -> zCFDOutputField:
    """Available from `tier` upwards, in both contexts."""
    return zCFDOutputField(name, category, description, alias_of, tier, tier)


def _volume(name, category, tier, description="", alias_of=None) -> zCFDOutputField:
    return zCFDOutputField(name, category, description, alias_of, tier, None)


def _surface(name, category, tier, description="", alias_of=None) -> zCFDOutputField:
    return zCFDOutputField(name, category, description, alias_of, None, tier)


# =============================================================================
# THE REGISTRY
# Grouped by category for reading; availability is the tier on each entry.
# =============================================================================

OUTPUT_FIELDS: Tuple[zCFDOutputField, ...] = (
    (
        # --- Primitive -----------------------------------------------------------
        _base("pressure", "primitive", "Static pressure (Pa)"),
        _base("p", "primitive", "Static pressure (Pa)", alias_of="pressure"),
        _base("density", "primitive", "Density (kg/m^3)"),
        _base("rho", "primitive", "Density (kg/m^3)", alias_of="density"),
        _base("velocity", "primitive", "Velocity vector (m/s)"),
        _base("V", "primitive", "Velocity vector (m/s)", alias_of="velocity"),
        _base("v", "primitive", "Velocity vector (m/s)", alias_of="velocity"),
        _base("temperature", "primitive", "Temperature (K)"),
        _base("T", "primitive", "Temperature (K)", alias_of="temperature"),
        _base("t", "primitive", "Temperature (K)", alias_of="temperature"),
        _base("mach", "primitive", "Mach number"),
        _base("m", "primitive", "Mach number", alias_of="mach"),
        _base(
            "gauge_pressure",
            "primitive",
            "Pressure relative to the reference value (Pa)",
        ),
        # --- Derived -------------------------------------------------------------
        _base("potentialtemperature", "derived", "Potential temperature (K)"),
        _base("cp", "derived", "Pressure coefficient"),
        _base("totalcp", "derived", "Total pressure coefficient"),
        _base(
            "ek",
            "derived",
            "Kinetic energy contained in the cell, 0.5 rho |V|^2 times the cell volume (non-dimensional)",
        ),
        _base("enstrophy", "derived", "Enstrophy - squared magnitude of vorticity"),
        _base("velocitynd", "derived", "Non-dimensional velocity"),
        _volume("vorticity", "derived", "euler", "Vorticity vector (1/s)"),
        _volume("vorticitynd", "derived", "euler", "Non-dimensional vorticity"),
        _volume(
            "Qcriterion", "derived", "euler", "Q criterion - vortex core identification"
        ),
        _volume("Qcriterionnd", "derived", "euler", "Non-dimensional Q criterion"),
        _volume("helicity", "derived", "euler", "Helicity"),
        _volume("helicitynd", "derived", "euler", "Non-dimensional helicity"),
        _both("viscosity", "derived", "viscous", "Dynamic viscosity (Pa.s)"),
        _both(
            "mu", "derived", "viscous", "Dynamic viscosity (Pa.s)", alias_of="viscosity"
        ),
        _both("nu", "derived", "viscous", "Kinematic viscosity (m^2/s)"),
        # --- Gradients -----------------------------------------------------------
        _base("pressuregrad", "gradient", "Pressure gradient vector"),
        _base("densitygrad", "gradient", "Density gradient vector"),
        _base("velocitygrad", "gradient", "Velocity gradient tensor"),
        _base("temperaturegrad", "gradient", "Temperature gradient vector"),
        _volume(
            "turbulencekineticenergygrad",
            "gradient",
            "rans",
            "Turbulence kinetic energy gradient",
        ),
        _volume(
            "turbulenceeddyfrequencygrad",
            "gradient",
            "rans",
            "Turbulence eddy frequency gradient",
        ),
        # --- Turbulence ----------------------------------------------------------
        # Eddy viscosity needs a turbulence model in the volume, but is written on
        # any viscous wall.
        zCFDOutputField(
            "eddy",
            "turbulence",
            "Eddy viscosity (Pa.s)",
            volume_tier="les",
            surface_tier="viscous",
        ),
        _volume("reynoldstress", "turbulence", "les", "Reynolds stress tensor"),
        _volume("ti", "turbulence", "rans", "Turbulence intensity"),
        _both(
            "turbulencekineticenergy",
            "turbulence",
            "rans",
            "Turbulence kinetic energy (J/kg)",
        ),
        _both(
            "turbulenceeddyfrequency",
            "turbulence",
            "rans",
            "Turbulence eddy frequency (1/s)",
        ),
        _both("nuTilde", "turbulence", "rans", "Spalart-Allmaras working variable"),
        _volume("menterf1", "turbulence", "rans", "Menter SST blending function"),
        _both("lesregion", "turbulence", "rans", "DES/DDES/IDDES region flag"),
        _volume(
            "viscouslengthscale",
            "turbulence",
            "rans",
            "Minimum distance between a cell and its neighbours",
        ),
        _volume(
            "turbulenceintermittency",
            "turbulence",
            "transition",
            "Transition model intermittency",
        ),
        _volume(
            "transitionreynoldsnumber",
            "turbulence",
            "transition",
            "Transition momentum-thickness Reynolds number",
        ),
        # --- Wall / boundary layer -----------------------------------------------
        _base("walldist", "wall", "Distance to the nearest wall (m)"),
        _both("walldistancezone", "wall", "rans", "Wall-distance zone id"),
        _surface("yplus", "wall", "viscous", "Non-dimensional wall distance"),
        _surface("ut", "wall", "viscous", "Friction velocity at the wall"),
        _surface("cf", "wall", "viscous", "Skin friction coefficient"),
        _surface("roughness", "wall", "rans", "Wall roughness height"),
        _surface(
            "heatflux",
            "wall",
            "viscous",
            "Wall heat flux (W/m^2), positive into the fluid",
        ),
        _surface("twall", "wall", "viscous", "Wall temperature (K)"),
        _base(
            "velocitynearnd", "wall", "Non-dimensional velocity at the near-wall cell"
        ),
        # --- Forces & moments ----------------------------------------------------
        _surface(
            "pressureforce", "forces", "euler", "Pressure component of force on a face"
        ),
        _surface(
            "pressuremoment",
            "forces",
            "euler",
            "Pressure component of moment on a face",
        ),
        _surface(
            "pressuremomentx", "forces", "euler", "Pressure moment about the x axis"
        ),
        _surface(
            "pressuremomenty", "forces", "euler", "Pressure moment about the y axis"
        ),
        _surface(
            "pressuremomentz", "forces", "euler", "Pressure moment about the z axis"
        ),
        _surface(
            "frictionforce",
            "forces",
            "viscous",
            "Skin friction component of force on a face",
        ),
        _surface(
            "frictionmoment",
            "forces",
            "viscous",
            "Skin friction component of moment on a face",
        ),
        _surface(
            "frictionmomentx",
            "forces",
            "viscous",
            "Skin friction moment about the x axis",
        ),
        _surface(
            "frictionmomenty",
            "forces",
            "viscous",
            "Skin friction moment about the y axis",
        ),
        _surface(
            "frictionmomentz",
            "forces",
            "viscous",
            "Skin friction moment about the z axis",
        ),
        # --- Mesh & geometry -----------------------------------------------------
        _base("zone", "mesh", "Boundary/mesh zone identifier"),
        _base("centre", "mesh", "Cell or face centre location"),
        _base("cell_centre", "mesh", "Cell centre location"),
        _volume("cellvolume", "mesh", "euler", "Cell volume"),
        _volume(
            "cell_volupdate", "mesh", "euler", "Cell volume update rate (moving mesh)"
        ),
        _volume(
            "cellcolour",
            "mesh",
            "euler",
            "Partition/colouring index used by the mesh partitioner",
        ),
        _volume(
            "cell_velocity",
            "mesh",
            "euler",
            "Velocity of the mesh at each cell centre from mesh motion (m/s)",
        ),
        _volume(
            "cellzone",
            "mesh",
            "euler",
            "Cell zone number of each cell, as stored in the mesh file",
        ),
        _volume(
            "GlobalCellId",
            "mesh",
            "euler",
            "Global cell index, as named in the invalid-cell log lines",
        ),
        # --- Solver & numerics ---------------------------------------------------
        _base("timestep", "solver", "Local timestep"),
        _volume("turbulencetimestep", "solver", "rans", "Local turbulence timestep"),
        _base("densitylimiter", "solver", "Density limiter value"),
        _base("klimiter", "solver", "Turbulence kinetic energy limiter value"),
        _base("omegalimiter", "solver", "Turbulence eddy frequency limiter value"),
        _volume(
            "sagradientlimiter",
            "solver",
            "rans",
            "Spalart-Allmaras gradient limiter value",
        ),
        _volume(
            "celllimiter",
            "solver",
            "euler",
            "Cell-based reconstruction limiter value for density "
            "(1 = unlimited); written by the 'barth jespersen', "
            "'michalak gooch' and 'venkatakrishnan' limiters",
        ),
        _volume(
            "celllimiterpressure",
            "solver",
            "euler",
            "Cell-based reconstruction limiter value for pressure "
            "(1 = unlimited); written by the 'barth jespersen', "
            "'michalak gooch' and 'venkatakrishnan' limiters",
        ),
        _volume(
            "roelowdissipation",
            "solver",
            "euler",
            "Roe low-dissipation blending factor",
        ),
        _volume(
            "FailCell",
            "solver",
            "euler",
            "1 in a cell whose solution is invalid, 0 elsewhere",
        ),
        _volume(
            "badcells",
            "solver",
            "euler",
            "Largest non-orthogonality among the cell's first-order faces "
            "under mesh quality remediation",
        ),
        _volume("sponge", "solver", "euler", "Sponge layer damping coefficient"),
        _volume(
            "MomentumSource",
            "solver",
            "euler",
            "Momentum source term (fluid zone sources)",
        ),
        _both("timestepratio", "solver", "euler", "Local time step ratio"),
        _volume(
            "artificial_viscosity", "solver", "euler", "Artificial viscosity applied"
        ),
        _volume("cellorder", "solver", "euler", "Spatial order used in each cell"),
        _volume(
            "lambdavariable", "solver", "euler", "Spectral radius of the flux Jacobian"
        ),
        _both(
            "lescfl",
            "solver",
            "rans",
            "CFL on the LES length scale used by the DES CFL shield: the larger of the "
            "grid relative transport CFL and the eddy turnover CFL",
        ),
        _both(
            "lesturnover",
            "solver",
            "rans",
            "Eddy turnover contribution to lescfl on its own, so the binding mechanism "
            "is visible",
        ),
        _both(
            "fdshield",
            "solver",
            "rans",
            "DES/DDES/IDDES shielding function f_d: 1 in resolved LES, 0 in the "
            "shielded RANS region",
        ),
        # --- Running average and RMS (output settings -> compute_average_and_rms)
        _base("V_avg", "primitive", "Running average of the velocity vector (m/s)"),
        _base("V_rms", "primitive", "RMS of the velocity vector (m/s)"),
        _base("p_avg", "primitive", "Running average of the static pressure (Pa)"),
        _base("p_rms", "primitive", "RMS of the static pressure (Pa)"),
        _base("T_avg", "primitive", "Running average of the temperature (K)"),
        _base("T_rms", "primitive", "RMS of the temperature (K)"),
        # --- Overset -------------------------------------------------------------
        _base("overset", "overset", "Overset blanking/donor status"),
        _volume("parent", "overset", "euler", "Parent cell/zone id (overset)"),
        _surface(
            "modaldeformation", "fsi", "modal", "Modal deformation of the surface"
        ),
        _both(
            "meshvelocity",
            "fsi",
            "mesh velocity",
            "Mesh velocity from mesh motion (m/s)",
        ),
    )
    + tuple(
        _surface(
            f"modeshapedisplacement_{ii}",
            "fsi",
            "modal",
            f"Displacement of mode shape {ii}",
        )
        for ii in range(1, 15)
    )
    + tuple(
        _surface(
            f"modeshaperotation_{ii}", "fsi", "modal", f"Rotation of mode shape {ii}"
        )
        for ii in range(1, 11)
    )
)

OUTPUT_FIELDS_BY_NAME: Dict[str, zCFDOutputField] = {f.name: f for f in OUTPUT_FIELDS}

AVERAGED_VARIABLES = frozenset(
    f.name for f in OUTPUT_FIELDS if f.name.endswith(("_avg", "_rms"))
)


# =============================================================================
# DERIVED VIEWS — the frozensets this module used to declare by hand
# =============================================================================


def _tier_names(context: str, tier: str) -> frozenset:
    return frozenset(f.name for f in OUTPUT_FIELDS if f.tier_in(context) == tier)


def _names_for(context: str, tiers: Tuple[str, ...]) -> frozenset:
    return frozenset(f.name for f in OUTPUT_FIELDS if f.tier_in(context) in tiers)


BASE_VARIABLES = _tier_names("volume", "base") | _tier_names("surface", "base")

CFD_VOLUME_ADDITIONS = _tier_names("volume", "euler")
VISCOUS_VOLUME_ADDITIONS = _tier_names("volume", "viscous")
LES_VOLUME_ADDITIONS = _tier_names("volume", "les")
RANS_VOLUME_ADDITIONS = _tier_names("volume", "rans")
TRANSITION_VOLUME_ADDITIONS = _tier_names("volume", "transition")

CFD_SURFACE_ADDITIONS = _tier_names("surface", "euler")
VISCOUS_SURFACE_ADDITIONS = _tier_names("surface", "viscous")
RANS_SURFACE_ADDITIONS = _tier_names("surface", "rans")

EULER_VOLUME_VARS = _names_for("volume", TIERS_BY_EQUATION["euler"])
VISCOUS_VOLUME_VARS = _names_for("volume", TIERS_BY_EQUATION["viscous"])
RANS_VOLUME_VARS = _names_for("volume", TIERS_BY_EQUATION["rans"])
LES_VOLUME_VARS = _names_for("volume", TIERS_BY_EQUATION["les"])

EULER_SURFACE_VARS = _names_for("surface", TIERS_BY_EQUATION["euler"])
VISCOUS_SURFACE_VARS = _names_for("surface", TIERS_BY_EQUATION["viscous"])
RANS_SURFACE_VARS = _names_for("surface", TIERS_BY_EQUATION["rans"])
LES_SURFACE_VARS = _names_for("surface", TIERS_BY_EQUATION["les"])

MODAL_SURFACE_VARIABLES = _names_for("surface", ("mesh velocity", "modal"))
MODAL_VOLUME_VARIABLES = _names_for("volume", ("mesh velocity", "modal"))

VOLUME_VARS_BY_EQUATION = {
    eq: _names_for("volume", tiers) for eq, tiers in TIERS_BY_EQUATION.items()
}
SURFACE_VARS_BY_EQUATION = {
    eq: _names_for("surface", tiers) for eq, tiers in TIERS_BY_EQUATION.items()
}


# =============================================================================
# DYNAMIC FIELDS (var_i, resvar_i, vargrad_i)
# One per solved equation, so generated rather than registered.
# =============================================================================

# Number of equations per type
EQUATION_TO_NUM_EQ = {
    "euler": 5,
    "viscous": 5,
    "les": 5,
    "incomp viscous": 5,
    "incomp les": 5,
    "rans": 7,
    "incomp rans": 7,
}

# Special turbulence model overrides
TURBULENCE_MODEL_NUM_EQ = {
    "sa-neg": 6,
    "sst-transition": 9,
}

# (prefix, "{}"-templated description keyed on the equation index)
_DYNAMIC_FIELDS = (
    ("var_", "Solution value of solved equation {}"),
    ("resvar_", "Residual of solved equation {}"),
    (
        "resshare_",
        "Share of the total squared residual held by each cell, equation {}. Sums to one over "
        "the mesh; threshold at 1/Neff (reported when output settings › report › "
        "residual statistics is on) to isolate the cells carrying the residual",
    ),
    ("vargrad_", "Gradient of solved equation {}"),
)


def get_num_equations(equation_type: str, turbulence_model: str = None) -> int:
    """How many equations this configuration solves."""
    eq_type = equation_type.lower()
    num_eq = EQUATION_TO_NUM_EQ.get(eq_type, 5)
    if eq_type.endswith("rans") and turbulence_model:
        num_eq = TURBULENCE_MODEL_NUM_EQ.get(turbulence_model.lower(), num_eq)
    return num_eq


def get_dynamic_fields(
    equation_type: str, turbulence_model: str = None
) -> List[zCFDOutputField]:
    """The per-equation fields (var_i, resvar_i, vargrad_i), described."""
    num_eq = get_num_equations(equation_type, turbulence_model)
    return [
        zCFDOutputField(
            f"{prefix}{ii}",
            "solver",
            template.format(ii),
            volume_tier="base",
            surface_tier="base",
        )
        for prefix, template in _DYNAMIC_FIELDS
        for ii in range(1, num_eq + 1)
    ]


def get_dynamic_variables_for_equation_type(
    equation_type: str, turbulence_model: str = None
) -> List[str]:
    """
    Generate dynamic variables like var_i, resvar_i, vargrad_i based on equation type.

    Args:
        equation_type: The equation type (e.g., 'euler', 'viscous', 'rans', etc.)
        turbulence_model: Optional turbulence model for RANS equations

    Returns:
        List of dynamic variable names
    """
    return [f.name for f in get_dynamic_fields(equation_type, turbulence_model)]


def get_fsi_modal_variables() -> dict:
    """Get FSI/modal model variables for surface and volume contexts."""
    return {
        "surface": list(MODAL_SURFACE_VARIABLES),
        "volume": list(MODAL_VOLUME_VARIABLES),
    }


# =============================================================================
# LOOKUP
# =============================================================================

#: Reported for a name with no registry entry, so an unrecognised variable is
#: still describable rather than raising.
UNKNOWN_FIELD = zCFDOutputField("", "other", "")


def describe_variable(name: str) -> dict:
    """Category and definition for one variable name, as a plain dict.

    Never raises: a name with no entry is reported as category "other" with
    an empty description.
    """
    field = OUTPUT_FIELDS_BY_NAME.get(name)
    if field is not None:
        return field.as_dict()

    for prefix, template in _DYNAMIC_FIELDS:
        index = name[len(prefix) :]
        if name.startswith(prefix) and index.isdigit():
            return {"category": "solver", "description": template.format(index)}

    return UNKNOWN_FIELD.as_dict()


def get_output_fields(
    equation_type: str,
    variable_context: str,
    turbulence_model: str = None,
    has_mesh_velocity: bool = False,
    has_fsi: bool = False,
) -> List[zCFDOutputField]:
    """Every field this configuration can write, in name order.

    Args:
        equation_type: The equation type (e.g., 'euler', 'viscous', 'rans')
        variable_context: Either 'volume' or 'surface'
        turbulence_model: Optional turbulence model for RANS equations
        has_mesh_velocity: Whether a mesh has 'motion' or an FSI boundary
        has_fsi: Whether a boundary carries an FSI coupling

    Raises:
        ValueError: if `equation_type` is not one zCFD solves.
    """
    tiers = TIERS_BY_EQUATION.get(equation_type.lower())
    if tiers is None:
        raise ValueError(f"Unknown equation type: {equation_type}")

    if turbulence_model and turbulence_model.lower() in TRANSITION_TURBULENCE_MODELS:
        tiers = tiers + ("transition",)

    fields = [
        f
        for f in OUTPUT_FIELDS
        if f.is_available(variable_context, tiers, has_mesh_velocity, has_fsi)
    ]
    fields.extend(get_dynamic_fields(equation_type, turbulence_model))
    return sorted(fields, key=lambda f: f.name)


def get_valid_variables_for_equation_type(
    equation_type: str,
    variable_context: str,
    turbulence_model: str = None,
    has_mesh_velocity: bool = False,
    has_fsi: bool = False,
) -> List[str]:
    """
    Get all valid variables for a given equation type and context.

    Args:
        equation_type: The equation type (e.g., 'euler', 'viscous', 'rans')
        variable_context: Either 'volume' or 'surface'
        turbulence_model: Optional turbulence model for RANS equations
        has_mesh_velocity: Whether a mesh has 'motion' or an FSI boundary
        has_fsi: Whether a boundary carries an FSI coupling

    Returns:
        List of valid variable names as strings
    """
    return [
        f.name
        for f in get_output_fields(
            equation_type,
            variable_context,
            turbulence_model,
            has_mesh_velocity,
            has_fsi,
        )
    ]


# =============================================================================
# DEFAULTS
# =============================================================================

# Variables written when a case names none of its own. These are the lists
# the solver falls back to, spelled in full rather than as aliases, so that
# what a UI shows pre-selected is what an unspecified case actually writes.
DEFAULT_VARIABLES = {
    "surface": ("pressure", "temperature", "density", "velocity", "yplus", "ut"),
    "volume": ("velocity", "temperature", "pressure", "density", "mach"),
}


def get_variable_catalogue(
    equation_type: str,
    variable_context: str,
    turbulence_model: str = None,
    has_mesh_velocity: bool = False,
    has_fsi: bool = False,
) -> dict:
    """The fields valid for one equation type and context, as JSON.

    Same arguments and same valid set as
    `get_valid_variables_for_equation_type`, plus what each field means, what
    categories are in use and which fields a case gets by default. This is
    what the `/zcfd/output-variables` endpoint serves, so that a UI documents
    variables without keeping a table of its own.

    Returns::

        {
          "variables":  [name, ...]                     sorted,
          "metadata":   {name: {category, description, alias_of?}, ...},
          "categories": [{"key", "label", "hint"}, ...] in display order,
          "defaults":   [name, ...]                     narrowed to `variables`,
        }
    """
    fields = get_output_fields(
        equation_type, variable_context, turbulence_model, has_mesh_velocity, has_fsi
    )
    names = [f.name for f in fields]
    used = {f.category for f in fields}
    offered = set(names)

    return {
        "variables": names,
        "metadata": {f.name: f.as_dict() for f in fields},
        "categories": [
            {"key": c.key, "label": c.label, "hint": c.hint}
            for c in OUTPUT_CATEGORIES
            if c.key in used
        ],
        "defaults": [
            name
            for name in DEFAULT_VARIABLES.get(variable_context, ())
            if name in offered
        ],
    }
