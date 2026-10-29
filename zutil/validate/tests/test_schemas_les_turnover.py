"""Field constraints on TurbulenceModel.les_turnover_coefficient.

The coefficient weights the eddy turnover term of the DES CFL shield. Its
schema default is None, not 1.0: the solver (SolverModelConst::setConstants)
supplies the real default of 1.0 only when the deck omits the key, so a deck
that never mentions the field must leave it unset here rather than carry a 1.0
that would mask a future change to the solver default.

The unsteady-only clause (When solver.time_settings.type == unsteady) is
applicability machinery enforced on the full ZCFDInputModelBase, not on this
model in isolation; it is covered by test_applicability.py (the mechanism) and
by test_real_schema_applies_when_paths_all_resolve (this field's When path).
"""

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.equations_schemas import TurbulenceModel


def test_default_is_none_so_the_solver_default_stands():
    assert TurbulenceModel(model="sst").les_turnover_coefficient is None


def test_accepts_a_weight():
    assert (
        TurbulenceModel(
            model="sst", les_turnover_coefficient=0.5
        ).les_turnover_coefficient
        == 0.5
    )


def test_zero_is_allowed_and_recovers_the_transport_only_criterion():
    assert (
        TurbulenceModel(
            model="sst", les_turnover_coefficient=0.0
        ).les_turnover_coefficient
        == 0.0
    )


def test_negative_weight_is_rejected():
    with pytest.raises(ValidationError):
        TurbulenceModel(model="sst", les_turnover_coefficient=-1.0)
