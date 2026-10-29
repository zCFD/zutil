"""
Monitor-point variable validation.

report.monitor.*.variables is checked against the equation-dependent
volume-variable list via the _validate_output_variables machinery.
"""

import pytest

from zutil.validate.schemas.output_schemas import (
    OutputModel,
    ReportModel,
    SolutionModel,
    _validate_output_variables,
)


def _output_settings(monitor):
    return OutputModel(
        solution=SolutionModel(),
        report=ReportModel(frequency=10, monitor=monitor),
    )


class TestMonitorVariableValidation:
    def test_valid_monitor_variables_pass(self):
        output_settings = _output_settings(
            {"MP_1": {"point": [0.0, 0.0, 0.0], "variables": ["p", "rho"]}}
        )
        # Should not raise.
        _validate_output_variables(output_settings, "euler")

    def test_invalid_monitor_variable_raises(self):
        output_settings = _output_settings(
            {"MP_1": {"point": [0.0, 0.0, 0.0], "variables": ["yplus"]}}
        )
        with pytest.raises(ValueError, match="MP_1"):
            _validate_output_variables(output_settings, "euler")

    def test_multiple_monitor_points_all_checked(self):
        output_settings = _output_settings(
            {
                "MP_1": {"point": [0.0, 0.0, 0.0], "variables": ["p"]},
                "MP_2": {"point": [1.0, 0.0, 0.0], "variables": ["not_a_real_var"]},
            }
        )
        with pytest.raises(ValueError, match="MP_2"):
            _validate_output_variables(output_settings, "euler")

    def test_no_monitor_points_does_not_raise(self):
        output_settings = _output_settings(None)
        _validate_output_variables(output_settings, "euler")

    def test_rans_specific_variable_valid_for_rans_but_not_euler(self):
        # 'eddy' (eddy viscosity) is a viscous/RANS variable, not valid for euler.
        output_settings = _output_settings(
            {"MP_1": {"point": [0.0, 0.0, 0.0], "variables": ["eddy"]}}
        )
        with pytest.raises(ValueError):
            _validate_output_variables(output_settings, "euler")
