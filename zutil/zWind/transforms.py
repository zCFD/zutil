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

Transforms required to convert from global coordinate system to blade element coordinate system

Currently does not allow for deforming blades, therefore all simulations will be rigid
"""

import numpy as np


# indvidual Rotation matrices


def R_xx(a):
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(a), np.sin(a)], [0.0, -np.sin(a), np.cos(a)]]
    )


def R_yy(a):
    return np.array(
        [[np.cos(a), 0.0, -np.sin(a)], [0.0, 1.0, 0.0], [np.sin(a), 0.0, np.cos(a)]]
    )


def R_zz(a):
    return np.array(
        [[np.cos(a), np.sin(a), 0.0], [-np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )


# triad combinations
def T_gh(yaw, tilt, azimuth):
    t_yaw = R_zz(yaw)
    t_tilt = R_xx(tilt)
    t_azimuth = R_yy(azimuth)
    return t_azimuth @ t_tilt @ t_yaw


def T_hb(azimuth, cone, pitch):
    t_azimuth = R_yy(azimuth)
    t_cone = R_zz(cone)
    t_pitch = R_xx(pitch)
    return t_pitch @ t_cone @ t_azimuth


def T_bh(azimuth, cone, pitch):
    t_hb = T_hb(cone, pitch)
    return np.linalg.inv(t_hb)


def T_bc(twist, sweep, prebend):
    t_twist = R_xx(twist)
    t_sweep = R_yy(sweep)
    t_prebend = R_zz(prebend)
    return t_twist @ t_sweep @ t_prebend


def global2chord(yaw, tilt, azimuth, cone, pitch, twist, sweep, prebend):
    t_gh = T_gh(yaw, tilt, azimuth)
    t_hb = T_hb(cone, pitch)
    t_bc = T_bc(twist, sweep, prebend)
    return t_bc @ t_hb @ t_gh
