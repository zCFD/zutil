"""Every tip-loss correction the schema offers gives the BET a usable factor."""

import typing

import pytest

pytest.importorskip("mpi4py")

from zutil.validate.schemas.fluid_zones_schemas import (
    turbine_tip_loss_correction_types,
)
from zutil.zWind.zBem import zTurbineBetBase


def _turbine(correction):
    turbine = object.__new__(zTurbineBetBase)
    turbine.tip_loss_correction = correction
    turbine.tip_loss_correction_r = 0.8
    return turbine


@pytest.mark.parametrize(
    "correction", typing.get_args(turbine_tip_loss_correction_types)
)
@pytest.mark.parametrize("radial_position", [0.3, 0.9])
def test_tip_loss_factor_is_a_fraction(correction, radial_position):
    factor = _turbine(correction).get_tip_loss_factor(radial_position, None)
    assert isinstance(factor, float)
    assert 0.0 < factor <= 1.0
