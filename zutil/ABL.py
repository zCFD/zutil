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
"""


import math
import numpy as np
from typing import Union


def coriolis_parameter(latitude_degree: float) -> float:
    """
    Return the coriolis parameter
    """
    # Earth rotation
    omega = 7.2722e-5  # rad/self
    return 2.0 * omega * math.sin(math.radians(latitude_degree))


def ekman_layer_height(friction_velocity: float, coriolis_parameter: float) -> float:
    """
    Returns the height of the atmospheric boundary layer - Geostrophic height
    for neutral conditions this is the height of the ABL
    """
    return friction_velocity / (6.0 * coriolis_parameter)


def friction_velocity(
    wind_speed: Union[float, int],
    height: Union[float, int],
    roughness_length: float,
    kappa: float = 0.41,
) -> float:
    """
    Returns the friction velocity
    """
    return wind_speed * kappa / math.log(height / roughness_length)


def wind_speed(
    height: Union[float, int],
    friction_velocity: Union[float, int],
    roughness_length: Union[float, int],
    kappa: float = 0.41,
) -> float:
    """
    Returns the wind speed at a given height

    May want to consider Deaves & Harris (1978) model for high speed wind
    """
    return friction_velocity / kappa * math.log(height / roughness_length)


def wind_speed_array(
    height_array: np.array,
    friction_velocity: Union[float, int],
    roughness_length: Union[float, int],
    kappa: float = 0.41,
) -> np.array:
    """
    Returns the wind speed at a given height

    May want to consider Deaves & Harris (1978) model for high speed wind
    """
    return friction_velocity / kappa * np.log((height_array / roughness_length))


def wind_direction_to_beta(wind_dir_deg: float) -> float:
    """
    Beta = 0.0 -> [1.0,0.0,0.0]
    Wind = 0.0 -> [0.0,-1.0,0.0]

    Therefore Beta = Wind + 90.0
    """
    return 360.0 - (wind_dir_deg + 90.0)


def vel_to_vxy(vel: Union[list, tuple]) -> float:
    """
    Speed of wind in xy plane

    vel = Velocity vector
    """
    return math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])


def vel_to_vxy_dir(vel: list) -> float:
    """
    Direction of wind in xy plane
    """
    vel_mag = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
    vxy_dir = math.asin(vel[0] / vel_mag)
    vxy_dir = 360.0 - math.degrees(vxy_dir)
    return vxy_dir


def vel_to_upflow(vel: list) -> float:
    """
    Wind upflow angle
    """
    vel_mag = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
    upflow_angle = math.asin(vel[2] / vel_mag)
    upflow_angle = math.degrees(upflow_angle)
    return upflow_angle
