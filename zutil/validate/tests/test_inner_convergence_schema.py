"""The inner_convergence block is shared by both solver families, but one of
its fields is not: 'target_ratio' is a floor on InnerRatio, which only the
compressible solver produces. On an incompressible deck it would be accepted
and never tested, so the schema withholds it there rather than swallow it.
'target_orders' carries the same default for both families by design - the
incompressible measure was calibrated to it.
"""

import pytest

from zutil.validate.schemas.convergence_schemas import InnerConvergence
from zutil.validate.tests.test_buoyancy_placement import (
    _incompressible_buoyancy_config,
)
from zutil.validate.tests.test_core_orchestration import _minimal_valid_config
from pydantic import ValidationError
from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _dual_time(config: dict, inner: dict) -> dict:
    ts = config["solver"]["time_settings"]
    ts.update(
        {
            "type": "unsteady",
            "kind": "dual time stepping",
            "total_time": 1.0,
            "time_step": 0.1,
        }
    )
    config["solver"]["convergence_control"]["inner_convergence"] = inner
    return config


def _compressible() -> dict:
    config = _minimal_valid_config()
    config["solver"]["convergence_control"] = {"cycles": 10}
    return config


class TestTargetRatioIsCompressibleOnly:
    def test_compressible_deck_may_set_it(self) -> None:
        model = ZCFDInputModelBase.model_validate(
            _dual_time(_compressible(), {"target_ratio": 1.0e-3})
        )
        assert model.solver.convergence_control.inner_convergence.target_ratio == 1.0e-3

    def test_incompressible_deck_may_not(self) -> None:
        with pytest.raises(ValidationError, match="target_ratio"):
            ZCFDInputModelBase.model_validate(
                _dual_time(_incompressible_buoyancy_config(), {"target_ratio": 1.0e-3})
            )

    def test_incompressible_deck_without_it_is_fine(self) -> None:
        model = ZCFDInputModelBase.model_validate(
            _dual_time(_incompressible_buoyancy_config(), {"target_orders": 1.0})
        )
        assert model.solver.convergence_control.inner_convergence.target_ratio is None


class TestTargetOrdersDefault:
    def test_shared_default_is_one_order(self) -> None:
        assert InnerConvergence().target_orders == 1.0


class TestDualTimeSteppingOnly:
    """The block is withheld unless the deck runs dual time stepping.

    A steady run's "inner" loop is the whole solve, and global time stepping
    has no inner loop, so an accepted block would validate and do nothing.
    """

    def test_steady_deck_is_rejected(self) -> None:
        config = _compressible()
        config["solver"]["convergence_control"]["inner_convergence"] = {
            "target_orders": 1.0
        }
        with pytest.raises(ValidationError, match="inner_convergence"):
            ZCFDInputModelBase.model_validate(config)

    def test_global_timestepping_deck_is_rejected(self) -> None:
        config = _dual_time(_compressible(), {"target_orders": 1.0})
        config["solver"]["time_settings"]["kind"] = "global timestepping"
        # GTS demands an explicit time scheme; supply it so the only defect
        # left is the inner_convergence block itself
        config["solver"]["time_settings"]["scheme"] = {"name": "runge kutta"}
        with pytest.raises(ValidationError, match="inner_convergence"):
            ZCFDInputModelBase.model_validate(config)

    def test_dual_time_deck_is_accepted(self) -> None:
        model = ZCFDInputModelBase.model_validate(
            _dual_time(_compressible(), {"target_orders": 1.0})
        )
        assert model.solver.convergence_control.inner_convergence.target_orders == 1.0
