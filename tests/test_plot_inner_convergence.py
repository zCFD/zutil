"""plot_inner_convergence draws one curve per real time step; which steps it
draws is pure logic, split out as inner_convergence_steps so it can be pinned
without a report file or a figure."""

import os

# zutil.plot selects the ipympl notebook backend on import unless batch mode is
# asked for; the CI unit env has no ipympl, so ask before the import.
os.environ.setdefault("BATCH_ANALYSIS", "1")

import pytest

plot_report = pytest.importorskip(
    "zutil.plot.plot_report",
    reason="the interactive plot stack (ipywidgets) is not installed",
)
inner_convergence_steps = plot_report.inner_convergence_steps


def test_startup_step_is_dropped() -> None:
    # Real time cycle 0 is the startup transient; its curve is not comparable
    assert inner_convergence_steps([0, 1, 2, 3], max_curves=12) == [1, 2, 3]


def test_startup_only_run_still_plots() -> None:
    # A run that never left cycle 0 should show it rather than nothing
    assert inner_convergence_steps([0], max_curves=12) == [0]


def test_thinning_spans_the_run_rather_than_truncating() -> None:
    steps = list(range(1, 25))
    shown = inner_convergence_steps(steps, max_curves=12)
    assert shown[0] == 1
    assert shown[-1] >= 23  # stride 2 keeps odd steps up to 23
    assert len(shown) <= 13


def test_few_steps_are_all_shown() -> None:
    assert inner_convergence_steps([1, 2, 3], max_curves=12) == [1, 2, 3]


def test_duplicate_rows_collapse_in_order() -> None:
    # The report has one row per inner cycle, so each step appears many times
    assert inner_convergence_steps([1, 1, 1, 2, 2, 3], max_curves=12) == [1, 2, 3]
