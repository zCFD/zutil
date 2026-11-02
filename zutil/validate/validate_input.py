"""
A raising view of :func:`zutil.validate.deck.check_deck`.

Kept for callers that import this name; new code calls ``check_deck``.
"""

from pathlib import Path
from typing import Union

from .deck import check_deck


def validate_zcfd_input(input_source: Union[str, Path, dict]) -> object:
    """Validate a deck and return its model, raising on any error.

    Raises :class:`~zutil.validate.error_handling.UnsupportedDeckError`, a
    ``ValueError`` and a ``ValidationErrorHandler``, for a legacy, override or
    unsupported-version deck, and ``ValidationErrorHandler`` for a deck the
    schema rejects or a deprecated codepath; a deck that cannot be
    loaded raises whatever loading it raised. Warnings go to the
    ``zutil.validate`` logger.
    """
    report = check_deck(input_source)
    report.log_warnings()
    return report.model_or_raise()


__all__ = ["validate_zcfd_input"]
