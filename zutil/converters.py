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

Converter utilities
"""

import numpy as np


def vector_from_wind_dir(wind_dir_degree: float, wind_speed: float = 1.0) -> list:
    """
    Return vector given a wind direction and wind speed
    Wind dir = 0.0 -> [0.0,-1.0,0.0]
    Wind dir = 90.0 -> [-1.0,0.0,0.0]

    Wind dir - Meteorological wind direction
    (direction from which wind is blowing)

    u -> Zone Velocity (Towards East)
    v -> Meridional Velocity (Towards North)

    """

    return [
        -wind_speed * np.sin(np.deg2rad(wind_dir_degree)),
        -wind_speed * np.cos(np.deg2rad(wind_dir_degree)),
        0.0,
    ]


def wind_direction(uvel: float, vvel: float) -> float:
    """
    Calculate meteorological wind direction from velocity vector
    """
    return np.rad2deg(np.arctan2(-uvel, -vvel))


def vector_from_angle(alpha: float, beta: float, mag: float = 1.0) -> list:
    """
    Return vector given alpha and beta in degrees based on ESDU definition
    """
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    vec = [0.0, 0.0, 0.0]
    vec[0] = mag * np.cos(alpha) * np.cos(beta)
    vec[1] = mag * np.sin(beta)
    vec[2] = mag * np.sin(alpha) * np.cos(beta)
    return vec


def angle_from_vector(vec: list) -> tuple[float, float]:
    """
    Return vector given alpha and beta in degrees based on ESDU definition
    """
    mag = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

    beta = np.arcsin(vec[1] / mag)
    alpha = np.arccos(vec[0] / (mag * np.cos(beta)))
    alpha = np.rad2deg(alpha)
    beta = np.rad2deg(beta)
    return (alpha, beta)


def rotate_vector(vec: list, alpha_degree: float, beta_degree: float) -> list:
    """
    Rotate vector by alpha and beta based on ESDU definition
    """
    alpha = np.deg2rad(alpha_degree)
    beta = np.deg2rad(beta_degree)
    rot = [0.0, 0.0, 0.0]
    rot[0] = (
        np.cos(alpha) * np.cos(beta) * vec[0]
        + np.sin(beta) * vec[1]
        + np.sin(alpha) * np.cos(beta) * vec[2]
    )
    rot[1] = (
        -np.cos(alpha) * np.sin(beta) * vec[0]
        + np.cos(beta) * vec[1]
        - np.sin(alpha) * np.sin(beta) * vec[2]
    )
    rot[2] = -np.sin(alpha) * vec[0] + np.cos(alpha) * vec[2]
    return rot


def feet_to_meters(val: float) -> float:
    """Convert feet to meters"""
    return val * 0.3048


def pressure_from_alt(alt: float) -> float:
    """
    Calculate pressure in Pa from altitude in m using standard atmospheric tables
    """
    return 101325.0 * (1.0 - 2.25577e-5 * alt) ** 5.25588


def rankine_to_kelvin(rankine: float) -> float:
    """Convert ranking to kelvin"""
    return rankine * 0.555555555


def dot(vec1: list, vec2: list) -> float:
    """Returns the dot product of two vectors - to be depreciated in favour of just using np.dot"""
    return np.dot(vec1, vec2)


def mag(vec: list) -> float:
    """Return the magnitude of a vector- to be depreciated in favour of just using np.linalg.norm"""
    return np.linalg.norm(vec)


def R_2vect(vector_orig: np.ndarray, vector_fin: np.ndarray) -> None:
    """
    Calculates the rotation matrix required to rotate from one vector to another.

    Args:
        R: The 3x3 rotation matrix to update (3x3 NumPy array).
        vector_orig: The unrotated vector in the reference frame (3x1 NumPy array).
        vector_fin: The rotated vector in the reference frame (3x1 NumPy array).
    """
    R = np.zeros((3, 3))
    # Normalize vectors and handle degenerate cases
    unit_orig = vector_orig / np.linalg.norm(vector_orig)
    unit_fin = vector_fin / np.linalg.norm(vector_fin)
    if np.allclose(unit_orig, unit_fin):
        # Vectors are equal or opposite, no rotation needed
        R[:] = np.eye(3)
        return R

    # Calculate and normalize rotation axis
    axis = np.cross(unit_orig, unit_fin)
    axis_len = np.linalg.norm(axis)
    axis = axis / axis_len if axis_len else np.zeros_like(axis)

    # Calculate rotation angle and trigonometric functions
    angle = np.arccos(np.dot(unit_orig, unit_fin))
    ca = np.cos(angle)
    sa = np.sin(angle)

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    R[0, 0] = 1.0 + (1.0 - ca) * (x**2 - 1.0)
    R[0, 1] = -z * sa + (1.0 - ca) * x * y
    R[0, 2] = y * sa + (1.0 - ca) * x * z
    R[1, 0] = z * sa + (1.0 - ca) * x * y
    R[1, 1] = 1.0 + (1.0 - ca) * (y**2 - 1.0)
    R[1, 2] = -x * sa + (1.0 - ca) * y * z
    R[2, 0] = -y * sa + (1.0 - ca) * x * z
    R[2, 1] = x * sa + (1.0 - ca) * y * z
    R[2, 2] = 1.0 + (1.0 - ca) * (z**2 - 1.0)
    return R


def vector_vector_rotate(
    vec: np.ndarray, axis: np.ndarray, origin: np.ndarray, theta: float
) -> np.ndarray:
    """
    Rotates a vector around an arbitrary axis passing through an origin.

    Args:
        vec: The vector to be rotated (3x1 NumPy array).
        axis: The rotation axis (3x1 NumPy array).
        origin: The point through which the axis passes (3x1 NumPy array).
        theta: The rotation angle in radians.

    Returns:
        The rotated vector (3x1 NumPy array).
    """

    # Ensure inputs are NumPy arrays
    vec, axis, origin = np.asarray(vec), np.asarray(axis), np.asarray(origin)

    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)

    # Calculate intermediate terms for efficiency
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    v = vec - origin
    v_proj = np.dot(v, axis) * axis  # Projection of v onto axis

    # Perform rotation using vectorized operations
    rotated_vec = (
        v_proj * t
        + v * c  # Unrotated component along axis
        + np.cross(axis, v)  # Rotated component perpendicular to axis
        * s  # Rotation around axis
        + origin  # Translate back to original origin
    )

    return rotated_vec


def unit_vector(vector: list) -> list:
    """Return the unit vector in the direction of the original vector"""
    return vector / np.linalg.norm(vector)


def angle_between(v1: list, v2: list) -> float:
    """Return the interior angle between two vectors"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_matrix(axis: list, theta: float) -> np.array:
    """Return the rotation matrix required to rotate a point about a specied axis, by a specified angle"""
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotate(coord: list, axis: list, ang: float) -> list:
    """Rotate a coordinate about a specified axis, by a specified angle"""
    c2 = np.dot(rotation_matrix(axis, ang), (coord[0], coord[1], coord[2]))
    return [c2[0], c2[1], c2[2]]


def project_to_plane(
    pt: np.array, plane_point: np.array, plane_normal: np.array
) -> np.array:
    """
    Projects a point onto a plane defined by a point and a normal vector.

    Args:
        pt (np.ndarray): The 3D point to be projected (shape: (3,)).
        plane_point (np.ndarray): A point on the plane (shape: (3,)).
        plane_normal (np.ndarray): The normal vector of the plane (shape: (3,)).

    Returns:
        np.ndarray: The projected point (shape: (3,)).
    """

    # Calculate the vector from the plane point to the input point
    v = pt - plane_point

    # Project the vector onto the plane normal
    projection = np.dot(v, plane_normal) * plane_normal

    # Subtract the projection from the vector to get the projected point
    projected_point = pt - projection

    return projected_point


def rpm2radps(rpm: float) -> float:
    """Converts rotations per minute to radians per second"""
    return 2 * np.pi * rpm / 60


def radps2rps(radps: float) -> float:
    """Converts radians per second to rotations per second.

    Needed for evaluation of propellor thrust coefficient"""
    return radps / (2 * np.pi)
