"""
Copyright (c) 2012-2025, Zenotech Ltd
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


Helper functions for plotting FWH results
"""

import matplotlib.pyplot as plt
from typing import Optional, Union
from zutil.plot import (
    get_figure,
    x_label,
    y_label,
    set_title,
    set_legend,
    plt_logo_stamp,
)
from collections.abc import Iterable
import pandas as pd

from zutil.analysis.acoustic import (
    calculate_PSD,
    calculate_thirdoctave_bands,
    convert_to_dB,
)

from zutil.fileutils import read_CAA_file


def plot_thirdoctave(
    p: list,
    time: Union[Iterable, float],
    sampling_frequency: float = 1.0,
    title: str = "Third Octave Band Data",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    A_weighting: bool = False,
) -> plt.Axes:
    """
    Plots third-octave band data for given observer data.

    Args:
        p (Arraylike): Pressure data array
        time [list, float]: time array, or timestep
        sampling_frequency (float = 1.0): Sampling frequency in Hz.
    Optional:
        title (str): Title of the plot.
        label (str): Label for the plot.
        ax (plt.Axes): Matplotlib Axes object to plot on. If None, a new figure is created.
        A_weighting (bool = False): If True, apply A-weighting to the bands.
    """
    # Initialize figure and set labels

    if ax is None:
        # create new figure if no axes are provided
        print("Creating new figure for third-octave plot")
        fig = get_figure()
        ax = fig.gca()
        ax.grid(True)

    else:
        fig = ax.figure

    x_label(ax, "Frequency [Hz]")
    y_label(ax, "dB [1/3 Octave]")

    # calculate Power Spectral Density (PSD) using Welch's method
    f1, Pxx_den1 = calculate_PSD(p, time, sampling_frequency=sampling_frequency)
    third_octave_freq, third_octave_data = calculate_thirdoctave_bands(
        f1, Pxx_den1, A_weighting=A_weighting
    )
    third_octave_data_db = convert_to_dB(third_octave_data)

    # Plot third-octave data for the current lifter
    ax.semilogx(third_octave_freq, third_octave_data_db, label=label)

    # Finalize plot
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])

    if label:
        # Add legend if label is provided
        set_legend(ax, "upper right")
    if title:
        # Set plot title if provided
        set_title(ax, title)

    return ax


def plot_observer(
    p: list,
    T: list,
    title: str = "Observer Data",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots observer data for given observer data.

    Args:
        p (Arraylike): Pressure data array
        T (Arraylike): Time data array
    Optional:
        title (str): Title of the plot.
        label (str): Label for the plot.
        ax (plt.Axes): Matplotlib Axes object to plot on. If None, a new figure is created.
    """
    # Initialize figure and set labels
    if ax is None:
        # create new figure if no axes are provided
        fig = get_figure()
        ax = fig.gca()
        ax.grid(True)

    else:
        fig = ax.figure

    # Set plot title
    if title:
        set_title(ax, title)

    x_label(ax, "Time [s]")
    y_label(ax, "Pressure [Pa]")

    # Plot observer data for the current lifter
    ax.plot(T, p, label=label)

    # Finalize plot
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])

    if label:
        set_legend(ax, "upper right")

    return ax


def plot_PSD(
    p: list,
    T: list,
    sampling_frequency: float = 1.0,
    title: str = "PSD",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    db_offset=None,
) -> plt.Axes:
    """
    Plots Power Spectral Density (PSD) for given observer data.

    Args:
        p (Arraylike): Pressure data array
        T (Arraylike): Time data array
        sampling_frequency (float = 1.0): Sampling frequency in Hz.
    Optional:
        title (str): Title of the plot.
        label (str): Label for the plot.
        ax (plt.Axes): Matplotlib Axes object to plot on. If None, a new figure is created.
    """
    # Initialize figure and set labels
    if ax is None:
        # create new figure if no axes are provided
        fig = get_figure()
        ax = fig.gca()
        ax.grid(True)
        print("Creating new figure for PSD plot")
    else:
        fig = ax.figure

    # Set plot title
    if title:
        set_title(ax, title)

    x_label(ax, "Frequency [Hz]")
    y_label(ax, "PSD [dB]")

    plt.ylim(0, 100)  # Set y-axis limits for PSD plot

    # Compute and plot Power Spectral Density (PSD) for p
    f, Pxx_den = calculate_PSD(p, T, sampling_frequency)
    Pxx_den_db = convert_to_dB(Pxx_den)

    if db_offset:
        Pxx_den_db = [val + db_offset for val in Pxx_den_db]

    ax.semilogx(f, Pxx_den_db, label=label)  # Plot on a logarithmic scale

    if label:
        # Add legend if label is provided
        set_legend(ax, "upper right")

    set_title(ax, title)

    return ax


def plot_all_PSD(
    filepath: str, sampling_frequency: float = 1.0, title: str = "PSD", stamp=False
) -> None:
    """
    Plot all PSDs from a given observer file

    Args:
        filepath (str): Path to the file containing observer data.
        sampling_frequency (float): Sampling frequency in Hz.
    """
    # Read the file and extract data
    data = pd.read_csv(filepath)

    # create new figure
    fig = get_figure()
    ax = fig.gca()
    ax.grid(True)

    time = data["Time"].to_list()
    # Loop through each observer and plot its PSD
    for column in data.columns:
        if column != "Time":
            p = data[column].to_list()
            # Plot PSD for the current observer
            plot_PSD(
                p,
                time,
                sampling_frequency=sampling_frequency,
                label=column,
                ax=ax,
                title=title,
            )

    if stamp:
        plt_logo_stamp(ax, location=(0.1, 0.9))
    fig.tight_layout()
    plt.legend()
    plt.show()


def plot_all_thirdoctave(
    filepath: str,
    sampling_frequency: float = 1.0,
    title: str = "Third Octave Band Data",
    stamp=False,
) -> None:
    """
    Plot all third-octave bands from a given observer file

    Args:
        filepath (str): Path to the file containing observer data.
        sampling_frequency (float): Sampling frequency in Hz.
    """
    # Read the file and extract data
    data = pd.read_csv(filepath)

    # create new figure
    fig = get_figure()
    ax = fig.gca()
    ax.grid(True)

    time = data["Time"].to_list()
    # Loop through each observer and plot its third-octave band data
    for column in data.columns:
        if column != "Time":
            p = data[column].to_list()
            # Plot third-octave band data for the current observer
            plot_thirdoctave(
                p,
                time,
                sampling_frequency=sampling_frequency,
                label=column,
                ax=ax,
                title=title,
            )
    if stamp:
        plt_logo_stamp(ax, location=(0.1, 0.9))
    fig.tight_layout()

    plt.legend()
    plt.show()


def plot_CAA_PSD(
    filepath: str,
    sampling_frequency: float = 1.0,
    title: str = "CAA PSD",
    stamp=False,
    label=None,
    ax=None,
    db_offset=None,
) -> plt.Axes:
    """
    Plot all PSDs from a given CAA observer file

    Args:
        filepath (str): Path to the file containing observer data.
        sampling_frequency (float): Sampling frequency in Hz.
    """
    p, time = read_CAA_file(filepath)
    ax = plot_PSD(
        p,
        time,
        sampling_frequency=sampling_frequency,
        label=label,
        ax=ax,
        title=title,
        db_offset=db_offset,
    )

    if stamp:
        plt_logo_stamp(ax, location=(0.1, 0.9))
    plt.legend()
    plt.show()

    return ax
