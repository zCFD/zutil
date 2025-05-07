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

Module init for ploting zCFD functions- only dependencies are matplotlib, numpy, and pandas
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.rcsetup import all_backends

import os

# Configure rendering for remote or interactive widgets
if "BATCH_ANALYSIS" in os.environ:
    batch = True
else:
    batch = False

# https://matplotlib.org/stable/users/explain/figure/backends.html

matplotlib.rcParams.update({"backend_fallback": False})
matplotlib.rcParams.update({"figure.max_open_warning": 0})


if batch:
    # Script mode- don't render figures in interactive widgets
    if "Agg" in all_backends:
        matplotlib.use("Agg")
    else:
        matplotlib.use("agg")
    plt.ioff()
else:
    # Interactive mode- use nbAgg backend
    matplotlib.use("module://ipympl.backend_nbagg")
    plt.ion()

# Messy, but ensures compatibility with exisiting test harness structure for now
from .zplot import get_figure
from .zplot import x_label
from .zplot import y_label
from .zplot import set_title
from .zplot import set_suptitle
from .zplot import set_legend
from .zplot import set_ticks

from .plot_report import Report
from .plot_report import zCFD_Plots

from . import font as ft
from . import colour as cl
