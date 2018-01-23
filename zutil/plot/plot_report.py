from zutil.post import get_csv_data
import ipywidgets.widgets as widgets
from IPython import display as dp

from zutil.plot import plt
from zutil.plot import display
from zutil.plot import batch
from zutil.plot import pd

import os


class Report(object):

    def __init__(self):
        self.header_list = None

    def plot_test(self, report_file):

        self.resildual_checkboxes = []
        cb_container = widgets.VBox()
        display(cb_container)
        for h in self.header_list:
            if h not in self.residual_list and h not in ['RealTimeStep', 'Cycle']:
                self.checkboxes.append(widgets.Checkbox(
                    description=h, value=False, width=90))

        row_list = []
        for i in range(0, len(self.checkboxes), 3):
            row = widgets.HBox()
            row.children = self.checkboxes[i:i + 3]
            row_list.append(row)

        cb_container.children = [i for i in row_list]

    def read_data(self, report_file):
        self.data = get_csv_data(report_file, header=True).dropna(axis=1)

        # Check for restart by
        restart_file = report_file.rsplit('.csv', 1)[0] + '.restart.csv'
        if os.path.isfile(restart_file):
            self.restart_data = get_csv_data(
                restart_file, header=True).dropna(axis=1)
            # Get first entry in new data
            restart_cycle = self.data['Cycle'].iloc[0]
            self.restart_data = self.restart_data[
                self.restart_data['Cycle'] < restart_cycle]
            # Merge restart data with data
            self.data = pd.concat(
                [self.restart_data, self.data], ignore_index=True)

        self.header_list = list(self.data)
        self.residual_list = []
        for h in self.header_list:
            if h.startswith('rho'):
                append_str = ''
                # if append_index > 0:
                #    append_str = '_'+str(append_index)
                self.residual_list.append(h + append_str)

    def plot(self, report_file):
        fig = plt.figure(figsize=(8, 5), dpi=100)
        ax = fig.gca()
        ax.set_yscale("log")
        ax.grid(True)
        ax.set_xlabel('cycles')
        ax.set_ylabel('RMS residual')

        append_index = 0
        if not isinstance(report_file, list):
            report_file_list = [report_file]
            ax.set_title(report_file[:report_file.rindex('_report.csv')])
        else:
            report_file_list = report_file
            if len(report_file_list) > 1:
                append_index = 1

        for report_file in report_file_list:
            if not os.path.isfile(report_file):
                print("File not found: " + str(report_file))
                continue

            self.read_data(report_file)

            y = self.residual_list
            self.data.plot(x='Cycle', y=y, ax=ax, legend=False)
            append_index = append_index + 1

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        if batch:
            fig.savefig(report_file + ".png", dpi=100)
        plt.show()
        self.visible_fig = []

    def plot_data(self, b):

        # Delete visbible figures
        for f in self.visible_fig:
            f.clear()
            plt.close(f)
            plt.gcf()
        dp.clear_output(wait=True)

        rolling = self.data.rolling(100).mean()
        for cb in self.checkboxes:
            if cb.value:
                h = cb.description
                fig = plt.figure()
                ax = fig.gca()
                y = h
                self.data.plot(x='Cycle', y=y, ax=ax, legend=False)
                rolling.plot(x=self.data['Cycle'], y=y, ax=ax, legend=0)
                last_val = self.data[h].tail(1).get_values()
                ax.set_title(str(h) + ' - ' + str(last_val))
                ax.grid(True)
                ax.set_xlabel('cycles')
                ax.set_ylabel(h)
                self.visible_fig.append(fig)
        plt.show()

    def plot_forces(self):

        if self.header_list is None:
            return

        self.checkboxes = []
        cb_container = widgets.VBox()
        display(cb_container)
        for h in self.header_list:
            if h not in self.residual_list and h not in ['RealTimeStep', 'Cycle']:
                self.checkboxes.append(widgets.Checkbox(
                    description=h, value=False, width=90))

        row_list = []
        for i in range(0, len(self.checkboxes), 3):
            row = widgets.HBox()
            row.children = self.checkboxes[i:i + 3]
            row_list.append(row)

        cb_container.children = [i for i in row_list]

        button = widgets.Button(description="Update plots")
        button.on_click(self.plot_data)
        display(button)

    def plot_performance(self, log_file):

        if not os.path.isfile(log_file):
            print("File not found: " + str(log_file))
            return

        elapsed_time = []
        import re
        with open(log_file) as lfile:
            for line in lfile:
                line = re.findall(r'Timer: Elapsed seconds: [0-9.]+', line)
                if line:
                    line = line[0].split(':')[2]
                    elapsed_time.append(float(line))

        df = pd.DataFrame(elapsed_time)

        fig = plt.figure(figsize=(8, 5), dpi=100)
        ax = fig.gca()
        ax.grid(True)
        ax.set_xlabel('cycles')
        ax.set_ylabel('Cycle Elapsed time (secs)')
        df.plot(ax=ax, legend=False)
        if batch:
            fig.savefig(log_file + ".png", dpi=100)
        plt.show()
