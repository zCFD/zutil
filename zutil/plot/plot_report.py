"""
Copyright (c) 2012-2024, Zenotech Ltd
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Zenotech Ltd nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ZENOTECH LTD BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Helper functions for plotting zCFD report files
"""

import matplotlib.pyplot as plt
import os
import ipywidgets.widgets as widgets
from IPython.display import display
import pandas as pd
import re
from zutil.fileutils import (
    zCFD_Result,
    get_zcfd_result,
    zCFD_Result,
    zCFD_Overset_Result,
)
from pathlib import Path
from typing import Optional
import warnings


class zCFD_Plots:
    """helper class to produce plots for zCFD report files"""

    def __init__(self, control_file_str) -> None:
        self.result = get_zcfd_result(control_file_str)

        if isinstance(self.result, zCFD_Overset_Result):
            self.overset = True
        elif isinstance(self.result, zCFD_Result):
            self.overset = False
        else:
            raise ValueError("Unknown result type")

        self.reports = []

        if self.overset:
            for result in self.result.overset_cases:
                self.reports.append(Report(result.control_file_path))
        else:
            self.reports.append(Report(self.result.control_file_path))

    def plot_residuals(self) -> None:
        """Plot residuals for all reports"""
        for report in self.reports:
            report.plot_residuals()

    def plot_forces(self, variables=None, mean: int = 20) -> None:
        """Plot forces for all reports"""
        for ii, report in enumerate(self.reports):
            report.plot_forces(variables=variables, mean=mean, report_index=ii)

    def plot_performance(self) -> None:
        """Plot performance for all reports"""
        if self.overset:
            # only one log file exists here for overset cases
            print("Log file information is currently limited for overset cases")
            _performance_plot(self.result.log_file_path)
            self.reports[0].plot_linear_solve_memory_usage()
        else:
            for report in self.reports:
                report.plot_performance()

    def plot_linear_solve_memory_usage(self) -> None:
        """Plot AMGX memory usage for all reports"""
        for report in self.reports:
            report.plot_linear_solve_memory_usage()


class Report(zCFD_Result):
    """Extension of zCFD_Result class to add plotting functions"""

    def __init__(self, control_file: Optional[str] = None) -> None:
        super().__init__(control_file)
        self.cb_container = None
        self.button = None
        self.rolling_avg = 100

        if "BATCH_ANALYSIS" in os.environ:
            self.batch = True
        else:
            self.batch = False

        if not control_file:
            print("Report() is deprecated, use zcfd_plots(path_to_controlfile) instead")
            warnings.warn(
                "Report() is deprecated, use zcfd_plots(path_to_controlfile) instead",
                DeprecationWarning,
            )

    def plot(self, csv_name):
        """code to support legacy plotting functionality"""
        # remove _report.csv from csv_name
        print("plot(csv_name) function is deprecated, use plot_residuals() instead")
        control_name = csv_name.replace("_report.csv", "")
        self.read_control_file(control_name)
        self.plot_residuals()
        # print depreciation warning

    def plot_residuals(self) -> None:
        """Plot residual data from report_file"""
        fig = plt.figure(figsize=(8, 5), dpi=100)
        ax = fig.gca()

        ax.clear()
        ax.set_yscale("log")

        ax.set_xlabel("cycles")
        ax.set_ylabel("RMS residual")

        ax.set_title(self.control_file_stem)

        # y = self.residual_list
        x = "Cycle"

        for y in self.report.residual_list:
            self.report.data.plot(x=x, y=y, ax=ax, legend=False)

        # Turn on major and minor grid lines
        ax.grid(True, "both")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

        if self.batch:
            fig.savefig(self.control_file_stem + "_residuals.png", dpi=100)

        plt.show()
        self.visible_fig = []

    def plot_forces(
        self, variables=None, mean: int = 20, report_index: Optional[int] = None
    ) -> None:
        if self.batch:
            if report_index is not None:
                print("report_index: {}".format(report_index))
            print("Plotting forces for case: {}".format(self.control_file_stem))
            self.plot_forces_batch(variables=variables, mean=mean)
        else:
            self.plot_forces_interactive(mean=mean)

    def _print_variables(self):
        """Print available variables to plot"""
        print("Available variables to plot:")
        available_vars = []
        for var in self.report.data.keys():
            if var not in self.report.residual_list + ["RealTimeStep", "Cycle"]:
                available_vars.append(var)

        print(available_vars)

    def plot_forces_batch(self, variables: Optional[list] = None, mean: int = 20):
        if not variables:
            # print available variables
            self._print_variables()

        else:
            # check if variable is valid:
            for var in variables:
                if var not in self.report.data.keys():
                    print(f"ERROR: Variable '{var}' not found in report data.")
                    self._print_variables()
                else:
                    print(f"Plotting '{var}'...  ")
                    rolling = self.report.data.rolling(self.rolling_avg).mean()
                    fig, ax = _forces_plot(self.report, var)
                    _rolling_avg(rolling, var, ax)
                    fig.savefig(f"{self.control_file_stem}_{var}.png")
                    print(f"... Saved {self.control_file_stem}_{var}.png")

    def plot_forces_interactive(self, mean: int = 20) -> None:
        """Create a checkbox widget to plot force and monitor data from a zCFD report file"""
        # Need to disable autoscroll
        # autoscroll(-1)

        self.rolling_avg = mean

        if self.report.header_list is None:
            return

        if self.cb_container is None:
            self.out = widgets.Output()
            self.checkboxes = []
            self.cb_container = widgets.VBox()
            for h in self.report.header_list:
                if h not in self.report.residual_list and h not in [
                    "RealTimeStep",
                    "Cycle",
                ]:
                    self.checkboxes.append(
                        widgets.Checkbox(description=h, value=False, width=90)
                    )

            row_list = []
            for i in range(0, len(self.checkboxes), 3):
                row = widgets.HBox()
                row.children = self.checkboxes[i : i + 3]
                row_list.append(row)

            self.cb_container.children = [i for i in row_list]

            self.rolling = widgets.IntSlider(
                value=self.rolling_avg,
                min=1,
                max=1000,
                step=1,
                description="Rolling Average:",
            )

            self.button = widgets.Button(description="Update plots")
            self.button.on_click(self.plot_data)

        display(self.out)
        display(self.cb_container)
        display(self.rolling)
        display(self.button)

    def plot_data(self, _) -> None:
        """Plot data from checked boxes in the plot_forces widget"""
        self.rolling_avg = self.rolling.value
        self.out.clear_output()
        with self.out:
            rolling = self.report.data.rolling(self.rolling_avg).mean()
            for cb in self.checkboxes:
                if cb.value:
                    y = cb.description
                    fig, ax = _forces_plot(self.report, y)
                    _rolling_avg(rolling, y, ax)
                    self.visible_fig.append(fig)
                    plt.show()

    def plot_performance(self, log_file_path: Optional[str] = None) -> None:
        """Plot CFD solver performance from a zCFD log file"""

        if log_file_path:
            print(
                "plot_performance(log_file_path) function is deprecated, use plot_performance() instead"
            )
            warnings.warn(
                "plot_performance(log_file_path) function is deprecated, use plot_performance() instead",
                DeprecationWarning,
            )

        _performance_plot(self.log_file_path)

        try:
            self.plot_linear_solve_memory_usage()
        except Exception as e:
            print("Error plotting AMGX memory usage: " + str(e))

    def plot_linear_solve_memory_usage(self) -> None:
        """Plots the linear solver usage from all zCFD log file ranks"""
        # figure out the number of ranks
        memory_usage = []

        for rank_log in self.rank_report_paths:
            memory_usage.append(_get_linear_solve_memory_usage_from_file(rank_log))

        fig = plt.figure(figsize=(8, 5), dpi=100)
        ax = fig.gca()
        ax.grid(True)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("AMGX memory usage (MB)")
        ax.set_title("AMGX Memory Usage for case {}".format(self.control_file_stem))

        for ii, rank_memory_usage in enumerate(memory_usage):
            ax.plot(rank_memory_usage)
            if len(rank_memory_usage) > 0:
                ax.plot(rank_memory_usage, label="Rank {}".format(ii))
                if self.batch:
                    fig.savefig(self.control_file_stem + "_memory_usage.png", dpi=100)
                ax.annotate(
                    "Rank {}".format(ii),
                    (len(rank_memory_usage) - 1, rank_memory_usage[-1]),
                    textcoords="offset points",
                    xytext=(20, 0),
                    ha="center",
                )
            else:
                print(
                    "No amgx memory usage data found in log file: {}".format(
                        self.rank_report_paths[ii]
                    )
                )

        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.1)

        # ax.legend(loc="upper right")
        plt.show()


def _performance_plot(fname: str) -> None:
    if not os.path.isfile(fname):
        print("File not found: " + str(fname))
        return

    elapsed_time = []

    with open(fname) as lfile:
        for line in lfile:
            line = re.findall(r"Timer: Elapsed seconds: [0-9.]+", line)
            if line:
                line = line[0].split(":")[2]
                elapsed_time.append(float(line))

    df = pd.DataFrame(elapsed_time)

    fig = plt.figure(figsize=(8, 5), dpi=100)
    ax = fig.gca()
    ax.grid(True)
    ax.set_xlabel("cycles")
    ax.set_ylabel("Cycle Elapsed time (secs)")
    df.plot(ax=ax, legend=False)
    if "BATCH_ANALYSIS" in os.environ:
        fig.savefig(fname + ".png", dpi=100)
    plt.show()


def _get_linear_solve_memory_usage_from_file(log_file: str) -> list:
    """Extract linear solver memory usage from zCFD log file"""
    if not os.path.isfile(log_file):
        print("File not found: " + str(log_file))
        return []

    memory_usage = []

    with open(log_file, "r") as f:
        lines = f.readlines()
        memory_pattern = re.compile(
            r"Maximum\s+Memory\s+Usage:\s+(\d+(?:\.\d+)?)(\s*(MB|GB|KB))"
        )
        memory_usage = []
        for line in lines:
            match = memory_pattern.search(line)
            if match:
                memory_info = {"value": match.group(1), "unit": match.group(3)}
                memory_usage.append(memory_info)

    main_solve_memory_usage = []

    for ii, memory_info in enumerate(memory_usage):
        # extract every other value
        if ii % 2 == 0:
            value = float(memory_info["value"])
            unit = memory_info["unit"]
            if unit == "KB":
                value *= 1e-3
            elif unit == "GB":
                value *= 1e3
            main_solve_memory_usage.append(value)

    return main_solve_memory_usage


def _forces_plot(report, y_axis):
    fig = plt.figure()
    ax = fig.gca()
    report.data.plot(x="Cycle", y=y_axis, ax=ax, legend=False)
    last_val = report.data[y_axis].tail(1).values

    ax.grid(True)
    ax.set_xlabel("cycles")
    ax.set_ylabel(y_axis)
    ax.set_title(str(y_axis) + " - " + str(last_val))
    return fig, ax


def _rolling_avg(rolling, y_axis, ax):
    rolling.plot(x="Cycle", y=y_axis, ax=ax, legend=False)
    # $\downarrow$ $\uparrow$ $\leftrightarrow$
    rolling_grad = rolling[y_axis].tail(2).values[0] - rolling[y_axis].tail(2).values[1]
    trend = r"$\leftrightarrow$"
    if rolling_grad > 0.0:
        trend = r"$\downarrow$"
    elif rolling_grad < 0.0:
        trend = r"$\uparrow$"
    ax.set_title(ax.get_title() + " " + trend)
