"""zutil.control — CFL ramp functions used by the time-marching schema."""

from .ramp import linear_ramp, linear_stepped_ramp, exponential_ramp, sequence_ramps

__all__ = [
    "linear_ramp",
    "linear_stepped_ramp",
    "exponential_ramp",
    "sequence_ramps",
]
