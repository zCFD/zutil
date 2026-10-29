"""
Direct accept/reject coverage for:
  - SolutionModel.validate_frequency (output_schemas.py)
  - _validate_output_variables's volume/surface branches (output_schemas.py)
    — test_schemas_monitor_variables.py already covers the monitor branch.
  - get_valid_variables_for_equation_type's unknown-equation-type reject
    path (output_variables.py).
"""

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.output_schemas import (
    FrequencyModel,
    OutputModel,
    ReportModel,
    SolutionModel,
    _validate_output_variables,
)
from zutil.validate import validate_zcfd_comprehensive
from zutil.validate.schemas.output_variables import (
    get_valid_variables_for_equation_type,
)
from zutil.validate.tests.test_output_variable_names import _deck


class TestSolutionModelValidateFrequency:
    def test_int_frequency_expands_to_frequency_model(self):
        solution = SolutionModel(frequency=5)
        assert isinstance(solution.frequency, FrequencyModel)
        assert solution.frequency.surface_data == 5
        assert solution.frequency.volume_data == 5
        assert solution.frequency.checkpoint == 5

    def test_dict_frequency_accepted(self):
        solution = SolutionModel(frequency={"surface_data": 10, "volume_data": 20})
        assert isinstance(solution.frequency, FrequencyModel)
        assert solution.frequency.surface_data == 10
        assert solution.frequency.volume_data == 20

    def test_default_frequency_is_a_frequency_model(self):
        solution = SolutionModel()
        assert isinstance(solution.frequency, FrequencyModel)

    def test_zero_int_frequency_rejected(self):
        with pytest.raises(ValidationError, match="must be >= 1"):
            SolutionModel(frequency=0)

    def test_negative_int_frequency_rejected(self):
        with pytest.raises(ValidationError, match="must be >= 1"):
            SolutionModel(frequency=-5)

    def test_wrong_type_frequency_rejected(self):
        with pytest.raises(ValidationError, match="must be either an integer"):
            SolutionModel(frequency="every 5 cycles")

    def test_already_constructed_frequency_model_accepted(self):
        """validate_frequency's before-validator has an
        `isinstance(v, FrequencyModel)` passthrough branch ahead of the
        dict/int branches, so an already-constructed FrequencyModel instance
        (e.g. round-tripping, or building SolutionModel programmatically
        from parts) is accepted unchanged instead of being rejected with a
        misleading "must be either an integer or a dictionary" message.
        """
        freq = FrequencyModel(surface_data=7, volume_data=7, checkpoint=7)
        solution = SolutionModel(frequency=freq)
        assert solution.frequency is freq
        assert solution.frequency.surface_data == 7


class TestCatalystScripts:
    """The solver hands each listed script to ParaView Catalyst."""

    def test_a_list_of_scripts_is_accepted(self):
        solution = SolutionModel(scripts=["catalyst1.py", "catalyst2.py"])
        assert solution.scripts == ["catalyst1.py", "catalyst2.py"]

    def test_unset_by_default(self):
        assert SolutionModel().scripts is None

    def test_a_single_string_is_rejected(self):
        with pytest.raises(ValidationError):
            SolutionModel(scripts="catalyst1.py")


class TestValidateOutputVariablesVolumeSurfaceBranches:
    def _output_settings(self, volume_variables=None, surface_variables=None):
        return OutputModel(
            solution=SolutionModel(
                volume_variables=volume_variables,
                surface_variables=surface_variables,
            ),
            report=ReportModel(frequency=10),
        )

    def test_valid_volume_variables_pass(self):
        output_settings = self._output_settings(volume_variables=["p", "rho"])
        _validate_output_variables(output_settings, "euler")

    def test_invalid_volume_variable_raises(self):
        output_settings = self._output_settings(volume_variables=["not_a_real_var"])
        with pytest.raises(ValueError, match="not_a_real_var"):
            _validate_output_variables(output_settings, "euler")

    def test_valid_surface_variables_pass(self):
        output_settings = self._output_settings(surface_variables=["cp"])
        _validate_output_variables(output_settings, "euler")

    def test_invalid_surface_variable_raises(self):
        output_settings = self._output_settings(surface_variables=["not_a_real_var"])
        with pytest.raises(ValueError, match="not_a_real_var"):
            _validate_output_variables(output_settings, "euler")

    def test_rans_only_volume_variable_invalid_for_euler(self):
        # 'eddy' (eddy viscosity) is a viscous/RANS variable, not valid for euler.
        output_settings = self._output_settings(volume_variables=["eddy"])
        with pytest.raises(ValueError):
            _validate_output_variables(output_settings, "euler")

    def test_no_variables_set_does_not_raise(self):
        output_settings = self._output_settings()
        _validate_output_variables(output_settings, "euler")


class TestAveragedVariablesNeedAveraging:
    """The solver throws when it writes an averaged or RMS variable whose
    storage 'compute average and rms' never allocated."""

    def _output_settings(self, **kwargs):
        return OutputModel(
            solution=SolutionModel(volume_variables=["p", "p_avg"]),
            report=ReportModel(frequency=10),
            **kwargs,
        )

    def test_without_averaging_rejected(self):
        with pytest.raises(ValueError) as info:
            _validate_output_variables(self._output_settings(), "euler")
        assert str(info.value) == (
            "Requesting p_avg needs 'compute average and rms' set to true in "
            "these output settings."
        )

    def test_monitor_variable_without_averaging_rejected(self):
        output_settings = OutputModel(
            solution=SolutionModel(),
            report=ReportModel(
                frequency=10,
                monitor={"probe": {"point": [0.0, 0.0, 0.0], "variables": ["V_rms"]}},
            ),
        )
        with pytest.raises(ValueError, match="Requesting V_rms needs"):
            _validate_output_variables(output_settings, "euler")

    def test_with_averaging_accepted(self):
        _validate_output_variables(
            self._output_settings(compute_average_and_rms=True), "euler"
        )


class TestGetValidVariablesForEquationTypeUnknownType:
    def test_known_equation_type_returns_variables(self):
        variables = get_valid_variables_for_equation_type("euler", "volume")
        assert "p" in variables

    def test_unknown_equation_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown equation type"):
            get_valid_variables_for_equation_type("not_a_real_equation", "volume")


REPORT_FORCES = (
    "'forces' is not a setting of 'output settings': forces are reported "
    'from \'report\', as "report": {"forces": {"<name>": {"zones": [...]}}}.'
)


@pytest.mark.parametrize(
    "path",
    [("solver", "output settings"), ("model", "block", "output settings")],
    ids=["solver", "mesh"],
)
def test_forces_beside_report_is_refused_with_a_pointer_to_report(path):
    deck = _deck()
    block = deck
    for key in path[:-1]:
        block = block[key]
    block.setdefault(path[-1], {"report": {"frequency": 10}})
    block[path[-1]]["forces"] = {"wing": {"zones": [1]}}
    _, nodes = validate_zcfd_comprehensive(deck)
    errors = [
        (n["field_path"], n["display_message"])
        for n in nodes
        if n.get("severity", "error") == "error"
    ]
    owner = ".".join(path[:-1])
    assert errors == [(f"{owner}.output_settings.forces", REPORT_FORCES)]
