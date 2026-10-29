"""Which version of each block a deck uses.

Each table lists, in order, the deck settings that select a version and
the version they select; the first row whose settings all match wins.
``None`` matches a setting left unset, and a row with no settings always
matches. A field takes its table with ``zCFD_Field(variants=...)``, and the
same table drives the validator, the zui form and the reference docs.
"""

from . import convergence_schemas as conv
from . import equations_schemas as eqn
from . import fluid_model_schemas as fld
from . import numerical_schemes_schemas as num
from . import reference_conditions_schemas as refs
from . import solver_settings_schemas as sset
from . import time_settings_schemas as ts
from .variants import NotAvailable

SOLVER = "solver.solver_settings.type"
EQUATIONS = "solver.equations.type"
TIME_TYPE = "solver.time_settings.type"
TIME_KIND = "solver.time_settings.kind"

INCOMPRESSIBLE_GTS = NotAvailable(
    "The incompressible solver cannot run under global time stepping: the "
    "coupled and SIMPLE schemes march an inner pseudo-time loop, which global "
    "time stepping does not have. Use time_settings kind 'dual time stepping' "
    "for an unsteady incompressible run, or type 'steady'."
)

# fmt: off
SOLVER_SETTINGS = [
    ({SOLVER: "compressible"},   sset.CompressibleSolverSettingsPropertiesModel),
    ({SOLVER: "incompressible"}, sset.IncompressibleSolverSettingsPropertiesModel),
]

EQUATIONS_BLOCK = [
    ({SOLVER: "compressible",   EQUATIONS: "euler"},   eqn.EulerEquations),
    ({SOLVER: "compressible",   EQUATIONS: "viscous"}, eqn.ViscousEquations),
    ({SOLVER: "compressible",   EQUATIONS: "rans"},    eqn.RANSEquations),
    ({SOLVER: "compressible",   EQUATIONS: "les"},     eqn.LESEquations),
    ({SOLVER: "incompressible", EQUATIONS: "viscous"}, eqn.IncompViscousEquations),
    ({SOLVER: "incompressible", EQUATIONS: "rans"},    eqn.IncompRANSEquations),
    ({SOLVER: "incompressible", EQUATIONS: "les"},     eqn.IncompLESEquations),
]

NUMERICAL_SCHEME = [
    ({SOLVER: "compressible",   EQUATIONS: "euler"},   num.FVCompressibleEulerNumericalScheme),
    ({SOLVER: "compressible",   EQUATIONS: "viscous"}, num.FVCompressibleViscousNumericalScheme),
    ({SOLVER: "compressible",   EQUATIONS: "rans"},    num.FVCompressibleRANSNumericalScheme),
    ({SOLVER: "compressible",   EQUATIONS: "les"},     num.FVCompressibleLESNumericalScheme),
    ({SOLVER: "incompressible", EQUATIONS: "viscous"}, num.FVIncompressibleViscousNumericalScheme),
    ({SOLVER: "incompressible", EQUATIONS: "rans"},    num.FVIncompressibleRANSNumericalScheme),
    ({SOLVER: "incompressible", EQUATIONS: "les"},     num.FVIncompressibleLESNumericalScheme),
]

FLUID_PROPERTIES = [
    ({SOLVER: "compressible",   EQUATIONS: "euler"},   fld.FluidModelBase),
    ({SOLVER: "compressible",   EQUATIONS: "viscous"}, fld.ViscousFluidModel),
    ({SOLVER: "compressible",   EQUATIONS: "rans"},    fld.ViscousFluidModel),
    ({SOLVER: "compressible",   EQUATIONS: "les"},     fld.ViscousFluidModel),
    ({SOLVER: "incompressible", EQUATIONS: "viscous"}, fld.IncompressibleFluidModel),
    ({SOLVER: "incompressible", EQUATIONS: "rans"},    fld.IncompressibleFluidModel),
    ({SOLVER: "incompressible", EQUATIONS: "les"},     fld.IncompressibleFluidModel),
]

REFERENCE_CONDITION = [
    ({EQUATIONS: "euler"},   refs.EulerReferenceConditions),
    ({EQUATIONS: "viscous"}, refs.ViscousReferenceConditions),
    ({EQUATIONS: "rans"},    refs.RANSReferenceConditions),
    ({EQUATIONS: "les"},     refs.RANSReferenceConditions),
]

TIME_SETTINGS = [
    ({TIME_TYPE: "steady"},                                                           ts.SteadyTimeSettings),
    ({SOLVER: "incompressible", TIME_TYPE: "unsteady", TIME_KIND: "global timestepping"}, INCOMPRESSIBLE_GTS),
    ({TIME_TYPE: "unsteady", TIME_KIND: "global timestepping"},                        ts.GTSTimeSettings),
    ({TIME_TYPE: "unsteady", TIME_KIND: "dual time stepping"},                         ts.DTSTimeSettings),
    ({TIME_TYPE: None},                                                               ts.SteadyTimeSettings),
]

CONVERGENCE_CONTROL = [
    ({SOLVER: "compressible"},   conv.CompressibleConvergenceControl),
    ({SOLVER: "incompressible"}, conv.IncompressibleConvergenceControl),
]

MESH_CONVERGENCE_CONTROL = [
    ({SOLVER: "compressible"},   conv.CompressibleConvergenceControl),
    ({SOLVER: "incompressible"}, conv.IncompressibleMeshConvergenceControl),
]
# fmt: on
