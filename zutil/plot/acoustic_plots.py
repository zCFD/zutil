"""
Copyright (c) 2012-2026, Zenotech Ltd
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from typing import Optional, Union
from zutil.plot import (
    get_titled_figure,
    x_label,
    y_label,
    set_ticks,
    set_legend,
)
from zutil.plot import colour as cl
from collections.abc import Iterable
import pandas as pd

from zutil.analysis.acoustic import (
    calculate_PSD,
    calculate_thirdoctave_bands,
    convert_to_dB,
)

_TICK_MANTISSAS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)


def _axes(
    ax: Optional[plt.Axes], title: Optional[str], subtitle: Optional[str]
) -> plt.Axes:
    """The supplied axes, or a new house-style titled figure's axes."""
    if ax is None:
        _fig, ax = get_titled_figure(title or "", subtitle)
    return ax


def _data_lines(ax: plt.Axes) -> list:
    return [line for line in ax.lines if line.get_label() != "_zero"]


def _log_frequency_ticks(f_range: tuple) -> list:
    """Round-number tick positions (1, 1.5, 2, 3, 4, 5, 6, 8 per decade) inside f_range."""
    lo, hi = f_range
    ticks = []
    for decade in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1):
        for mantissa in _TICK_MANTISSAS:
            tick = mantissa * 10.0**decade
            if lo <= tick <= hi:
                ticks.append(tick)
    return ticks


def apply_frequency_range(ax: plt.Axes, f_range: tuple) -> None:
    """Limit a log-frequency axis to f_range, size the y axis to the plotted data inside it
    in 10 dB steps, and label the ticks in plain Hz."""
    lo, hi = f_range
    ax.set_xlim(lo, hi)
    in_band = np.concatenate(
        [
            np.asarray(line.get_ydata())[
                (np.asarray(line.get_xdata()) >= lo)
                & (np.asarray(line.get_xdata()) <= hi)
            ]
            for line in _data_lines(ax)
        ]
    )
    if in_band.size:
        ax.set_ylim(
            10.0 * np.floor((in_band.min() - 5.0) / 10.0),
            10.0 * np.ceil((in_band.max() + 5.0) / 10.0),
        )
    ax.set_xticks(_log_frequency_ticks(f_range))
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_thirdoctave(
    p: list,
    time: Union[Iterable, float],
    sampling_frequency: float = 1.0,
    title: Optional[str] = "Third Octave Band Data",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    A_weighting: bool = False,
    subtitle: Optional[str] = None,
    f_range: Optional[tuple] = None,
) -> plt.Axes:
    """
    Plots third-octave band data for given observer data.

    Args:
        p (Arraylike): Pressure data array
        time [list, float]: time array, or timestep
        sampling_frequency (float = 1.0): Sampling frequency in Hz.
    Optional:
        title (str): Figure title, used when a new figure is created.
        label (str): Label for the plot.
        ax (plt.Axes): Matplotlib Axes object to plot on. If None, a new figure is created.
        A_weighting (bool = False): If True, apply A-weighting to the bands.
        subtitle (str): Second title line, used when a new figure is created.
        f_range (tuple): (low, high) Hz window to show; the dB axis follows the data inside it.
    """
    ax = _axes(ax, title, subtitle)

    f1, Pxx_den1 = calculate_PSD(p, time, sampling_frequency=sampling_frequency)
    third_octave_freq, third_octave_data = calculate_thirdoctave_bands(
        f1, Pxx_den1, A_weighting=A_weighting
    )
    third_octave_data_db = convert_to_dB(third_octave_data)

    ax.semilogx(third_octave_freq, third_octave_data_db, label=label, linewidth=1.5)
    if f_range:
        apply_frequency_range(ax, f_range)

    x_label(ax, "Frequency (Hz)")
    y_label(ax, "dB (1/3 octave)")
    set_ticks(ax)
    if label:
        set_legend(ax, "upper right")
    return ax


def plot_observer(
    p: list,
    T: list,
    title: Optional[str] = "Observer Data",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    subtitle: Optional[str] = None,
    p0: Optional[float] = None,
) -> plt.Axes:
    """
    Plots observer pressure history.

    Args:
        p (Arraylike): Pressure data array
        T (Arraylike): Time data array
    Optional:
        title (str): Figure title, used when a new figure is created.
        label (str): Label for the plot.
        ax (plt.Axes): Matplotlib Axes object to plot on. If None, a new figure is created.
        subtitle (str): Second title line, used when a new figure is created.
        p0 (float): Ambient pressure; when given the acoustic pressure p - p0 is plotted on a
            y axis symmetric about zero.
    """
    ax = _axes(ax, title, subtitle)
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)
    if p0 is not None:
        p = p - p0

    ax.plot(T, p, label=label, linewidth=1.0)

    lines = _data_lines(ax)
    ax.set_xlim(
        min(np.min(line.get_xdata()) for line in lines),
        max(np.max(line.get_xdata()) for line in lines),
    )
    if p0 is not None:
        amplitude = 1.15 * max(np.max(np.abs(line.get_ydata())) for line in lines)
        ax.set_ylim(-amplitude, amplitude)
        if not any(line.get_label() == "_zero" for line in ax.lines):
            ax.axhline(0.0, color=cl.zeno_grey, linewidth=0.8, label="_zero")

    x_label(ax, "Time (s)")
    y_label(ax, r"$\mathbf{p - p_0}$ (Pa)" if p0 is not None else "Pressure (Pa)")
    set_ticks(ax)
    if label:
        set_legend(ax, "upper right")
    return ax


def plot_PSD(
    p: list,
    T: list,
    sampling_frequency: float = 1.0,
    title: Optional[str] = "PSD",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    db_offset=None,
    subtitle: Optional[str] = None,
    f_range: Optional[tuple] = None,
) -> plt.Axes:
    """
    Plots Power Spectral Density (PSD) for given observer data.

    Args:
        p (Arraylike): Pressure data array
        T (Arraylike): Time data array
        sampling_frequency (float = 1.0): Sampling frequency in Hz.
    Optional:
        title (str): Figure title, used when a new figure is created.
        label (str): Label for the plot.
        ax (plt.Axes): Matplotlib Axes object to plot on. If None, a new figure is created.
        db_offset (float): Added to every dB value.
        subtitle (str): Second title line, used when a new figure is created.
        f_range (tuple): (low, high) Hz window to show; the dB axis follows the data inside it.
    """
    ax = _axes(ax, title, subtitle)

    f, Pxx_den = calculate_PSD(p, T, sampling_frequency)
    Pxx_den_db = convert_to_dB(Pxx_den)
    if db_offset:
        Pxx_den_db = [val + db_offset for val in Pxx_den_db]

    ax.semilogx(f, Pxx_den_db, label=label, linewidth=1.5)
    if f_range:
        apply_frequency_range(ax, f_range)

    x_label(ax, "Frequency (Hz)")
    y_label(ax, "PSD (dB)")
    set_ticks(ax)
    if label:
        set_legend(ax, "upper right")
    return ax


def plot_all_PSD(
    filepath: str,
    sampling_frequency: float = 1.0,
    title: str = "PSD",
    stamp=True,
    f_range: Optional[tuple] = None,
) -> None:
    """
    Plot all PSDs from a given observer file on one house-style figure.

    Args:
        filepath (str): Path to the file containing observer data.
        sampling_frequency (float): Sampling frequency in Hz.
    Optional:
        title (str): Figure title.
        stamp: Kept for compatibility; the figure always carries the logo.
        f_range (tuple): (low, high) Hz window to show.
    """
    data = pd.read_csv(filepath)
    fig, ax = get_titled_figure(title)

    time = data["Time"].to_list()
    for column in data.columns:
        if column != "Time":
            plot_PSD(
                data[column].to_list(),
                time,
                sampling_frequency=sampling_frequency,
                label=column,
                ax=ax,
                f_range=f_range,
            )
    plt.show()


def plot_all_thirdoctave(
    filepath: str,
    sampling_frequency: float = 1.0,
    title: str = "Third Octave Band Data",
    stamp=True,
    f_range: Optional[tuple] = None,
) -> None:
    """
    Plot all third-octave bands from a given observer file on one house-style figure.

    Args:
        filepath (str): Path to the file containing observer data.
        sampling_frequency (float): Sampling frequency in Hz.
    Optional:
        title (str): Figure title.
        stamp: Kept for compatibility; the figure always carries the logo.
        f_range (tuple): (low, high) Hz window to show.
    """
    data = pd.read_csv(filepath)
    fig, ax = get_titled_figure(title)

    time = data["Time"].to_list()
    for column in data.columns:
        if column != "Time":
            plot_thirdoctave(
                data[column].to_list(),
                time,
                sampling_frequency=sampling_frequency,
                label=column,
                ax=ax,
                f_range=f_range,
            )
    plt.show()
