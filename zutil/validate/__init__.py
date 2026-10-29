"""
zCFD Validate - Modern validation tools for zCFD

This package provides Pydantic-based validation for zCFD configuration files
with sophisticated context-aware validation and clear error reporting.
"""

from .deck import DeckReport, check_deck
from .defaults import strip_defaults
from .validate_input import validate_zcfd_input
from .validate_zcfd_schema import validate_zcfd_comprehensive
from .error_handling import UnsupportedDeckError
from .error_handling import ValidationErrorHandler as ZCFDValidationError
from .zcfd_schema import ZCFDInputModelBase
from .schemas.base_types import in_case_directory

__version__ = "1.0.0"

__all__ = [
    "check_deck",
    "DeckReport",
    "validate_zcfd_input",
    "validate_zcfd_comprehensive",
    "ZCFDValidationError",
    "UnsupportedDeckError",
    "ZCFDInputModelBase",
    "in_case_directory",
    "strip_defaults",
]
