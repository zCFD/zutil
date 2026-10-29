"""
TimeScheme.stage coverage.

The runtime supports a fixed set of stage values: 1, 3, 4, 5, or the
string "rk third order tvd". Any other value must be rejected at validation
rather than failing per-MPI-rank at solver setup.

TimeScheme is a discriminated union keyed on 'name'
(EulerScheme/RungeKuttaScheme/ImplicitEulerScheme/LUSGSScheme) and a type
alias rather than a constructible class, so these tests build the concrete
variant directly.
"""

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.convergence_schemas import (
    EulerScheme,
    RungeKuttaScheme,
)


class TestRKStageConstrained:
    @pytest.mark.parametrize("stage", [1, 3, 4, 5, "rk third order tvd"])
    def test_supported_stages_accepted(self, stage):
        scheme = RungeKuttaScheme(name="runge kutta", stage=stage)
        assert scheme.stage == stage

    @pytest.mark.parametrize("stage", [2, 6, 0, -1, "third order", "rk3"])
    def test_unsupported_stages_rejected(self, stage):
        with pytest.raises(ValidationError):
            RungeKuttaScheme(name="runge kutta", stage=stage)

    def test_stage_still_optional(self):
        scheme = RungeKuttaScheme(name="runge kutta")
        assert scheme.stage is None

    def test_stage_has_no_effect_under_euler(self):
        """euler is single-stage by construction; 'stage' is accepted (see
        EulerScheme.stage docstring — real-world decks carry it as dead
        boilerplate) but never consulted."""
        scheme = EulerScheme(name="euler")
        assert scheme.stage is None
