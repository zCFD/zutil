from builtins import str
from builtins import range
from builtins import object
from zutil.post import get_csv_data
import ipywidgets.widgets as widgets
from IPython import display as dp

from zutil.plot import plt
from zutil.plot import display
from zutil.plot import batch
from zutil.plot import pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from IPython.display import Javascript


def autoscroll(threshhold):
    if threshhold == 0:  # alway scroll !not good
        javastring = """
        IPython.OutputArea.prototype._should_scroll = function(lines) {
            return true;
        }
        """
    elif threshhold == -1:  # never scroll !not good
        javastring = """
        IPython.OutputArea.prototype._should_scroll = function(lines) {
            return false;
        }
        """
    else:
        javastring = "IPython.OutputArea.auto_scroll_threshold = " + str(threshhold)
    display(Javascript(javastring))


class Report(object):
    def __init__(self):
        self.header_list = None
        self.cb_container = None
        self.button = None
        self.rolling_avg = 100

    def plot_test(self, report_file):

        self.resildual_checkboxes = []
        cb_container = widgets.VBox()
        display(cb_container)
        for h in self.header_list:
            if h not in self.residual_list and h not in ["RealTimeStep", "Cycle"]:
                self.checkboxes.append(
                    widgets.Checkbox(description=h, value=False, width=90)
                )

        row_list = []
        for i in range(0, len(self.checkboxes), 3):
            row = widgets.HBox()
            row.children = self.checkboxes[i : i + 3]
            row_list.append(row)

        cb_container.children = [i for i in row_list]

    def read_data(self, report_file):
        self.data = get_csv_data(report_file, header=True).dropna(axis=1, how="all")

        # Check for restart by
        restart_file = report_file.rsplit(".csv", 1)[0] + ".restart.csv"
        if os.path.isfile(restart_file):
            self.restart_data = get_csv_data(restart_file, header=True).dropna(
                axis=1, how="all"
            )
            # Get first entry in new data
            restart_cycle = self.data["Cycle"].iloc[0]
            self.restart_data = self.restart_data[
                self.restart_data["Cycle"] < restart_cycle
            ]
            # Merge restart data with data
            self.data = pd.concat([self.restart_data, self.data], ignore_index=True)

        # if self.append_index > 0:
        #    self.data = self.data.add_suffix('_' + str(self.append_index))

        self.header_list = list(self.data)
        self.residual_list = []
        for h in self.header_list:
            if h.startswith("rho") or h == "p" or h.startswith("turbEqn"):
                append_str = ""
                # if append_index > 0:
                #    append_str = '_'+str(append_index)
                self.residual_list.append(h + append_str)

    def plot_multiple(self, report_file_list, variable_list, out_file):

        fig = plt.figure(figsize=(8, 5), dpi=100)

        handles_ = []
        for f in report_file_list:

            if not os.path.isfile(f):
                print("File not found: " + str(f))
                continue

            case_name = f[:-11]

            vars = []
            with open(f, "r") as ff:
                vars = ff.readline().split()

            data = np.genfromtxt(f, skip_header=1)
            for v in range(0, len(vars) - 2):

                variable_name = vars[v + 2]
                if len(variable_list) != 0 and variable_name not in variable_list:
                    continue
                line, = plt.semilogy(
                    data[:, 1], data[:, v + 2], label=case_name + " " + variable_name
                )
                handles_.append(line)

        plt.legend(handles=handles_)
        plt.xlabel("cycles")
        plt.ylabel("RMS residual")
        plt.show()
        if batch:
            fig.savefig(out_file + ".png", dpi=100)
        plt.show()
        self.visible_fig = []

    def plot(self, refile):
        fig = plt.figure(figsize=(8, 5), dpi=100)
        ax = fig.gca()

        self.report_file = refile

        def animate(i):
            ax.clear()
            ax.set_yscale("log")

            ax.set_xlabel("cycles")
            ax.set_ylabel("RMS residual")

            self.append_index = 0
            if not isinstance(self.report_file, list):
                report_file_list = [self.report_file]
                ax.set_title(self.report_file[: self.report_file.rindex("_report.csv")])
            else:
                report_file_list = self.report_file
                if len(report_file_list) > 1:
                    self.append_index = 1

            for report_file in report_file_list:
                if not os.path.isfile(self.report_file):
                    print("File not found: " + str(self.report_file))
                    continue

                self.read_data(self.report_file)

                # y = self.residual_list
                x = "Cycle"
                if self.append_index > 0:
                    x = x + "_" + str(self.append_index)
                for y in self.residual_list:
                    self.data.plot(x=x, y=y, ax=ax, legend=False)
                self.append_index = self.append_index + 1

            # Turn on major and minor grid lines
            ax.grid(True, "both")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

        animate(0)
        # if not batch:
        #     ani = animation.FuncAnimation(fig, animate, interval=10000)

        if batch:
            fig.savefig(report_file + ".png", dpi=100)
        plt.show()
        self.visible_fig = []

    def plot_data(self, b):

        # Delete visbible figures
        # for f in self.visible_fig:
        #    f.clear()
        #    plt.close(f)
        #    plt.gcf()
        # dp.clear_output(wait=True)
        self.rolling_avg = self.rolling.value
        self.out.clear_output()
        with self.out:
            rolling = self.data.rolling(self.rolling_avg).mean()
            for cb in self.checkboxes:
                if cb.value:
                    h = cb.description
                    fig = plt.figure()
                    ax = fig.gca()
                    y = h
                    self.data.plot(x="Cycle", y=y, ax=ax, legend=False)
                    rolling.plot(x="Cycle", y=y, ax=ax, legend=0)
                    last_val = self.data[h].tail(1).values
                    # $\downarrow$ $\uparrow$ $\leftrightarrow$
                    rolling_grad = (
                        rolling[h].tail(2).values[0]
                        - rolling[h].tail(2).values[1]
                    )
                    trend = r"$\leftrightarrow$"
                    if rolling_grad > 0.0:
                        trend = r"$\downarrow$"
                    elif rolling_grad < 0.0:
                        trend = r"$\uparrow$"
                    ax.set_title(str(h) + " - " + str(last_val) + " " + trend)
                    ax.grid(True)
                    ax.set_xlabel("cycles")
                    ax.set_ylabel(h)
                    self.visible_fig.append(fig)
                    plt.show()
            # display(plt)

    def plot_forces(self, mean=20):

        # Need to disable autoscroll
        # autoscroll(-1)

        self.rolling_avg = mean

        if self.header_list is None:
            return

        if self.cb_container is None:
            self.out = widgets.Output()
            self.checkboxes = []
            self.cb_container = widgets.VBox()
            for h in self.header_list:
                if h not in self.residual_list and h not in ["RealTimeStep", "Cycle"]:
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

    def plot_performance(self, log_file):

        if not os.path.isfile(log_file):
            print("File not found: " + str(log_file))
            return

        elapsed_time = []
        import re

        with open(log_file) as lfile:
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
        if batch:
            fig.savefig(log_file + ".png", dpi=100)
        plt.show()
