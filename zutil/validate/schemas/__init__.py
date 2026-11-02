"""
Schema definitions for zCFD validation.

This package contains modular Pydantic schema definitions organized by component type.
"""

# Import all schema modules for easy access
from . import base_types
from . import boundary_conditions_schemas
from . import equations_schemas
from . import fluid_model_schemas
from . import fluid_zones_schemas
from . import initialisation_schemas
from . import numerical_schemes_schemas
from . import output_schemas
from . import output_variables
from . import reference_conditions_schemas
from . import solver_settings_schemas
from . import solver_schemas
from . import time_settings_schemas
from . import convergence_schemas
from . import ui_utils

__all__ = [
    "base_types",
    "boundary_conditions_schemas",
    "equations_schemas",
    "fluid_model_schemas",
    "fluid_zones_schemas",
    "initialisation_schemas",
    "numerical_schemes_schemas",
    "output_schemas",
    "output_variables",
    "reference_conditions_schemas",
    "solver_settings_schemas",
    "solver_schemas",
    "time_settings_schemas",
    "convergence_schemas",
    "ui_utils",
]
