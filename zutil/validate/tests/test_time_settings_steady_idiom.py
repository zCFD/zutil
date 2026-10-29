"""
DTS steady-idiom guard.

v1 explicitly treated `total time == time step` as STEADY
(ExplicitSolver.py: num_real_time_cycle = 0, dual time stepping off), so no
v1 deck with that shape ever time-marched. Accepting it as DTS runs a
genuine dual-time step and doubles the cycle budget, so DTSTimeSettings
rejects the idiom loudly.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from zutil.validate.schemas.time_settings_schemas import DTSTimeSettings


def _dts(total_time, time_step):
    return {
        "type": "unsteady",
        "kind": "dual time stepping",
        "total_time": total_time,
        "time_step": time_step,
        "order": "second",
    }


class TestSteadyIdiomRejected:
    def test_equal_total_and_step_rejected(self):
        with pytest.raises(ValidationError, match="steady idiom"):
            TypeAdapter(DTSTimeSettings).validate_python(_dts(1.0, 1.0))

    def test_genuine_dts_accepted(self):
        ts = TypeAdapter(DTSTimeSettings).validate_python(_dts(0.005, 1e-5))
        assert ts.total_time == 0.005
