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

Init module for zutil.post, requires paraview.simple
"""

import os

try:
    import paraview as pv
    import paraview.simple as pvs

    from .post import get_case_parameters
    from .post import get_case_parameters_str
    from .post import get_case_success

    from .post import get_case_root, get_case_report
    from .post import for_each

    from .post import ProgressBar
    from .post import get_status_dict

    from .post import cp_profile
    from .post import cf_profile

    from .post import calc_lift_centre_of_action
    from .post import calc_drag_centre_of_action
    from .post import get_monitor_data
    from .post import calc_moment

    from .post import calc_force_wall
    from .post import calc_moment_wall
    from .post import calc_force

    from .post import cp_profile_wall_from_file
    from .post import cf_profile_wall_from_file

    from .post import get_num_procs

    from .post import clean_vtk

    from .post import vtk_logo_stamp
    from .post import vtk_text_stamp
    from .post import plt_logo_stamp

    pv.compatibility.major = 5
    pv.compatibility.minor = 4
    pvs._DisableFirstRenderCameraReset()
except ImportError:
    # check if in zCFD environment
    if "ZCFD_HOME" in os.environ:
        raise ImportError(
            "paraview.simple not found in zCFD environment. "
            "Please check your installation."
        )
    else:
        raise ImportError(
            "ERROR: zutil.post requires paraview.simple to be installed- please use the zCFD environment to run this module."
        ) from None
