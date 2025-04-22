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

Helper functions for producing matplotlib plots formatted to a standardised `zcfd` house style
"""

from zutil.plot import plt
from zutil.plot import colour as cl
from zutil.plot import font as ft


def get_figure(plt: plt, x: int = 8, y: int = 5) -> plt.figure:
    """Create a matyplotlib Figure of windowsize (x,y)"""
    return plt.figure(figsize=(x, y), dpi=100, facecolor="w", edgecolor=cl.zeno_orange)


def x_label(ax: plt.axis, text: str) -> None:
    """Set a string "text" as the x axis label"""
    ax.set_xlabel(
        text, fontsize=ft.axis_font_size, fontweight="bold", color=cl.zeno_grey
    )


def y_label(ax: plt.axis, text: str) -> None:
    """Set a string "text" as the y axis label"""
    ax.set_ylabel(
        text, fontsize=ft.axis_font_size, fontweight="bold", color=cl.zeno_grey
    )


def set_title(ax: plt.axis, text: str) -> None:
    """Set a string "text" as the figure title"""
    ax.set_title(text, fontsize=ft.title_font_size)


def set_suptitle(fig: plt.figure, text: str) -> None:
    """Set a string "text" as the subplot title"""
    fig.suptitle(text, fontsize=24, fontweight="normal", color=cl.zeno_grey)


def set_legend(
    ax: plt.axis, location: str, label_list: list = None, bbox_to_anchor=None
) -> None:
    """Create a legend for the plotted variables following the strings specified in the label list"""
    if label_list is not None:
        legend = ax.legend(
            loc=location,
            scatterpoints=1,
            numpoints=1,
            shadow=False,
            fontsize=ft.legend_font,
            labels=label_list,
            bbox_to_anchor=bbox_to_anchor,
        )
    else:
        legend = ax.legend(
            loc=location,
            scatterpoints=1,
            numpoints=1,
            shadow=False,
            fontsize=ft.legend_font,
            bbox_to_anchor=bbox_to_anchor,
        )
    legend.get_frame().set_facecolor("white")
    return legend


def set_ticks(ax: plt.axes) -> None:
    """Format axis ticks"""
    ax.tick_params(axis="x")

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(ft.axis_tick_font_size)
        tick.label1.set_fontweight("normal")
        tick.label1.set_color(cl.zeno_orange)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(ft.axis_tick_font_size)
        tick.label1.set_fontweight("normal")
        tick.label1.set_color(cl.zeno_orange)
