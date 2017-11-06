import numpy as np
import matplotlib
import os
import pandas as pd

#  This is required when running in script mode
batch = False
if 'BATCH_ANALYSIS' in os.environ:
    batch = True

from matplotlib.rcsetup import all_backends

if batch:
    if 'Agg' in all_backends:
        matplotlib.use('Agg')
    else:
        matplotlib.use('agg')
else:
    if 'nbagg' in all_backends:
        matplotlib.use('nbagg')
    else:
        matplotlib.use('nbAgg')

matplotlib.rcParams['backend_fallback'] = False

from matplotlib import pylab, mlab, pyplot
plt = pyplot
if batch:
    plt.ioff()
else:
    plt.ion()

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs

from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

import font as ft
import colour as cl

from plot_report import Report


def get_figure(plt, x=8, y=5):
    return plt.figure(figsize=(x, y), dpi=100, facecolor='w', edgecolor=cl.zeno_orange)


def x_label(ax, text):
    ax.set_xlabel(text, fontsize=ft.axis_font_size,
                  fontweight='bold', color=cl.zeno_grey)


def y_label(ax, text):
    ax.set_ylabel(text, fontsize=ft.axis_font_size,
                  fontweight='bold', color=cl.zeno_grey)


def set_title(ax, text):
    ax.set_title(text, fontsize=ft.title_font_size)


def set_suptitle(fig, text):
    fig.suptitle(text, fontsize=24, fontweight='normal', color=cl.zeno_grey)


def set_legend(ax, location, label_list=None):
    if label_list is not None:
        legend = ax.legend(loc=location, scatterpoints=1, numpoints=1, shadow=False,
                           fontsize=ft.legend_font, labels=label_list)
    else:
        legend = ax.legend(loc=location, scatterpoints=1, numpoints=1, shadow=False,
                           fontsize=ft.legend_font)
    legend.get_frame().set_facecolor('white')
    return legend


def set_ticks(ax):

    ax.tick_params(axis='x')

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ft.axis_tick_font_size)
        tick.label.set_fontweight('normal')
        tick.label.set_color(cl.zeno_orange)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ft.axis_tick_font_size)
        tick.label.set_fontweight('normal')
        tick.label.set_color(cl.zeno_orange)
