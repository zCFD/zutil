"""The exception a failed deck check raises, carrying the terminal report as its text."""


class ValidationErrorHandler(Exception):
    """A deck failed validation; ``str()`` is the text to show the engineer."""


class UnsupportedDeckError(ValidationErrorHandler, ValueError):
    """A deck the v2 schema never sees: legacy, override or of another version."""


ZCFDValidationError = ValidationErrorHandler
