"""
Copyright (c) 2012-2017, Zenotech Ltd
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

from past.builtins import execfile
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
import math
import sys
from os import path
import numpy as np
import imp


def get_parameters_from_file(filename):
    conf = filename
    mymodule = __import__(conf)
    # Force a reload just in case it has already been loaded
    imp.reload(mymodule)
    return getattr(sys.modules[conf], "parameters")


def include(filename):
    """
    include a file by executing it. This imports everything including
    variables into the calling module
    """
    if path.exists(filename):
        exec(compile(open(filename, "rb").read(), filename, "exec"))


def get_zone_info(module_name):
    try:
        # mymodule = __import__(module_name)
        # Force a reload just in case it has already been loaded
        # reload(mymodule)
        # return mymodule
        import importlib

        return importlib.import_module(module_name)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return None


def get_default_zone_info():
    import inspect

    _, filename, linenumber, _, _, _ = inspect.stack()[1]
    return get_zone_info(path.split(path.splitext(filename)[0])[1] + "_zone")


def find_next_zone(parameters, zone_prefix):
    # Find next available
    found = False
    counter = 1
    while not found:
        key = zone_prefix + "_" + str(counter)
        if key in parameters:
            counter += 1
        else:
            found = True
    return key


def vector_from_wind_dir(wind_dir_degree, wind_speed=1.0):
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
        -wind_speed * math.sin(math.radians(wind_dir_degree)),
        -wind_speed * math.cos(math.radians(wind_dir_degree)),
        0.0,
    ]


def wind_direction(uvel, vvel):
    """
    Calculate meteorological wind direction from velocity vector
    """
    return math.degrees(math.atan2(-uvel, -vvel))


def vector_from_angle(alpha, beta, mag=1.0):
    """
    Return vector given alpha and beta in degrees based on ESDU definition
    """
    alpha = math.radians(alpha)
    beta = math.radians(beta)
    vec = [0.0, 0.0, 0.0]
    vec[0] = mag * math.cos(alpha) * math.cos(beta)
    vec[1] = mag * math.sin(beta)
    vec[2] = mag * math.sin(alpha) * math.cos(beta)
    return vec


def angle_from_vector(vec):
    """
    Return vector given alpha and beta in degrees based on ESDU definition
    """
    mag = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

    beta = math.asin(old_div(vec[1], mag))
    alpha = math.acos(old_div(vec[0], (mag * math.cos(beta))))
    alpha = math.degrees(alpha)
    beta = math.degrees(beta)
    return (alpha, beta)


def rotate_vector(vec, alpha_degree, beta_degree):
    """
    Rotate vector by alpha and beta based on ESDU definition
    """
    alpha = math.radians(alpha_degree)
    beta = math.radians(beta_degree)
    rot = [0.0, 0.0, 0.0]
    rot[0] = (
        math.cos(alpha) * math.cos(beta) * vec[0]
        + math.sin(beta) * vec[1]
        + math.sin(alpha) * math.cos(beta) * vec[2]
    )
    rot[1] = (
        -math.cos(alpha) * math.sin(beta) * vec[0]
        + math.cos(beta) * vec[1]
        - math.sin(alpha) * math.sin(beta) * vec[2]
    )
    rot[2] = -math.sin(alpha) * vec[0] + math.cos(alpha) * vec[2]
    return rot


def feet_to_meters(val):
    return val * 0.3048


def pressure_from_alt(alt):
    """
    Calculate pressure in Pa from altitude in m using standard atmospheric tables
    """
    return 101325.0 * math.pow((1.0 - 2.25577e-5 * alt), 5.25588)


def to_kelvin(rankine):
    return rankine * 0.555555555


# def non_dim_time(dim_time):
#    speed = 0.2 * math.sqrt(1.4 * 287.0 * 277.77)
#    non_dim_speed = 0.2 * math.sqrt(0.2)
#    return dim_time * speed / non_dim_speed


def dot(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


def mag(vec):
    return math.sqrt(dot(vec, vec))


def R_2vect(R, vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.

    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.

    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::

              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |


    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """
    # Python module imports.
    from math import acos, atan2, cos, pi, sin
    from numpy import array, cross, dot, float64, hypot, zeros
    from numpy.linalg import norm
    from random import gauss, uniform

    # Convert the vectors to unit vectors.
    vector_orig = old_div(vector_orig, norm(vector_orig))
    vector_fin = old_div(vector_fin, norm(vector_fin))

    # The rotation axis (normalised).
    axis = cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = old_div(axis, axis_len)

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = acos(dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = cos(angle)
    sa = sin(angle)

    # Calculate the rotation matrix elements.
    R[0, 0] = 1.0 + (1.0 - ca) * (x ** 2 - 1.0)
    R[0, 1] = -z * sa + (1.0 - ca) * x * y
    R[0, 2] = y * sa + (1.0 - ca) * x * z
    R[1, 0] = z * sa + (1.0 - ca) * x * y
    R[1, 1] = 1.0 + (1.0 - ca) * (y ** 2 - 1.0)
    R[1, 2] = -x * sa + (1.0 - ca) * y * z
    R[2, 0] = -y * sa + (1.0 - ca) * x * z
    R[2, 1] = x * sa + (1.0 - ca) * y * z
    R[2, 2] = 1.0 + (1.0 - ca) * (z ** 2 - 1.0)


def vector_vector_rotate(vec, axis, origin, theta):
    # Rotate vector
    temp = [0.0, 0.0, 0.0]

    temp[0] = (
        (
            origin[0] * (axis[1] * axis[1] + axis[2] * axis[2])
            - axis[0] * (origin[1] * axis[1] + origin[2] * axis[2] - dot(axis, vec))
        )
        * (1.0 - math.cos(theta))
        + vec[0] * math.cos(theta)
        + (
            -origin[2] * axis[1]
            + origin[1] * axis[2]
            - axis[2] * vec[1]
            + axis[1] * vec[2]
        )
        * math.sin(theta)
    )
    temp[1] = (
        (
            origin[1] * (axis[0] * axis[0] + axis[2] * axis[2])
            - axis[1] * (origin[0] * axis[0] + origin[2] * axis[2] - dot(axis, vec))
        )
        * (1.0 - math.cos(theta))
        + vec[1] * math.cos(theta)
        + (
            origin[2] * axis[0]
            - origin[0] * axis[2]
            + axis[2] * vec[0]
            - axis[0] * vec[2]
        )
        * math.sin(theta)
    )
    temp[2] = (
        (
            origin[2] * (axis[0] * axis[0] + axis[1] * axis[1])
            - axis[2] * (origin[0] * axis[0] + origin[1] * axis[1] - dot(axis, vec))
        )
        * (1.0 - math.cos(theta))
        + vec[2] * math.cos(theta)
        + (
            -origin[1] * axis[0]
            + origin[0] * axis[1]
            - axis[1] * vec[0]
            + axis[0] * vec[1]
        )
        * math.sin(theta)
    )

    return temp


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotate(coord, axis, ang):
    c2 = np.dot(rotation_matrix(axis, ang), (coord[0], coord[1], coord[2]))
    return [c2[0], c2[1], c2[2]]


def turbine_thrust_interpolate(u_inf, thrust_coef_curve):
    wsc = np.zeros((2, len(thrust_coef_curve)))
    i = 0
    for t in thrust_coef_curve:
        wsc[0][i] = t[0]
        wsc[1][i] = t[1]
        i += 1

    tc = np.interp(u_inf, wsc[0], wsc[1])
    return tc


def turbine_speed_interpolate(u_inf, tip_speed_curve):
    wsc = np.zeros((2, len(tip_speed_curve)))
    i = 0
    for t in tip_speed_curve:
        wsc[0][i] = t[0]
        wsc[1][i] = t[1]
        i += 1

    ts = np.interp(u_inf, wsc[0], wsc[1])
    return ts


# area of polygon specified as counter-clockwise vertices (x,y) where last = first


def polygon_area(x, y):
    a = 0.0
    for i in range(len(x) - 1):
        a += 0.5 * (x[i + 1] + x[i]) * (y[i + 1] - y[i])
    return a


def trapezoid(x, y):
    a = 0
    for i in range(len(x) - 1):
        a += 0.5 * (y[i + 1] + y[i]) * (x[i + 1] - x[i])
    return a


# Optimal power coefficient as a function of (a function of) tip speed ratio
# "A Compact, Closed-form Solution for the Optimum, Ideal Wind Turbine" (Peters, 2012)


def glauert_peters(y):
    p1 = 16.0 * (1.0 - 2.0 * y) / (27.0 * (1.0 + y / 4.0))
    p2 = old_div(
        (math.log(2.0 * y) + (1.0 - 2.0 * y) + 0.5 * (1.0 - 2.0 * y) ** 2),
        ((1.0 - 2.0 * y) ** 3),
    )
    p3 = (
        1.0
        + (457.0 / 1280.0) * y
        + (51.0 / 640.0) * y ** 2
        + y ** 3 / 160.0
        + 3.0 / 2.0 * y * p2
    )
    power_coeff = p1 * p3
    return power_coeff


# Golden section search: Given f with a single local max in [a,b], gss returns interval [c,d] with d-c <= tol.


def gss(f, a, b, tol=1e-5):
    invphi = old_div((math.sqrt(5) - 1), 2)  # 1/phi
    invphi2 = old_div((3 - math.sqrt(5)), 2)  # 1/phi^2
    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)
    n = int(math.ceil(old_div(math.log(old_div(tol, h)), math.log(invphi))))
    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)[0]
    yd = f(d)[0]
    for k in range(n - 1):
        if yc > yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)[0]
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)[0]
    if yc > yd:
        return (a, d)
    else:
        return (c, b)


def create_annulus(turbine_zone_dict):
    from mpi4py import MPI
    from numpy import zeros, array, dot, linalg, cross

    if "verbose" in turbine_zone_dict:
        verbose = turbine_zone_dict["verbose"]
    else:
        verbose = False

    if "number of segments" in turbine_zone_dict:
        number_of_segments = turbine_zone_dict["number of segments"]
    else:
        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            print("NO NUMBER OF SEGMENTS SPECIFIED - SETTING TO DEFAULT 12")
        number_of_segments = 12

    if "inner radius" in turbine_zone_dict:
        ri = turbine_zone_dict["inner radius"]
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("NO INNER RADIUS SPECIFIED")

    if "outer radius" in turbine_zone_dict:
        ro = turbine_zone_dict["outer radius"]
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("NO OUTER RADIUS SPECIFIED")

    disc_centre = turbine_zone_dict["centre"]
    disc_normal = turbine_zone_dict["normal"]

    rotor_swept_area = (math.pi * ro * ro) - (math.pi * ri * ri)

    annulus = []
    dtheta = math.radians(360.0 / number_of_segments)
    theta = 0.0
    total_area = 0.0
    for i in range(number_of_segments):
        r = ri
        while r < ro:
            dr = dtheta * max(r,0.01*ro) / (1.0 - 0.5 * dtheta)
            max_r = r + dr
            if max_r > ro:
                dr = ro - r
            rp = r + 0.5 * dr
            da = dtheta * rp * dr
            disc_theta = i * dtheta + 0.5 * dtheta

            disc_pt = np.array(
                [rp * math.cos(disc_theta), rp * math.sin(disc_theta), 0.0]
            )

            # Rotate so that z points in the direction of the normal
            R = np.zeros((3, 3))
            vector_orig = np.array([0.0, 0.0, 1.0])
            vector_fin = np.zeros(3)
            for j in range(3):
                vector_fin[j] = disc_normal[j]
            R_2vect(R, vector_orig, vector_fin)

            disc_pt = dot(R, disc_pt)

            # translate to disc centre
            for j in range(3):
                disc_pt[j] += disc_centre[j]

            annulus.append(
                (r, dr, i * dtheta, dtheta, disc_pt[0], disc_pt[1], disc_pt[2])
            )
            total_area += da
            r = r + dr

    return annulus


def zone_default(dict, key, default_val, verbose=True):
    from mpi4py import MPI

    if key in dict:
        value = dict[key]
    else:
        value = default_val
        dict[key] = default_val
        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            if "name" in dict:
                print(
                    "Turbine zone "
                    + str(dict["name"])
                    + " missing: "
                    + str(key)
                    + " - setting to "
                    + str(default_val)
                )
            else:
                print(
                    "Turbine zone missing name and missing: "
                    + str(key)
                    + " - setting to "
                    + str(default_val)
                )
    return value


def calculate_aerofoil_section_area(tzd):
    upper = zone_default(
        tzd["aerofoil profile"],
        "upper surface",
        [[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]],
        True,
    )
    lower = zone_default(
        tzd["aerofoil profile"],
        "lower surface",
        [[0.0, 0.0], [0.5, -0.1], [1.0, 0.0]],
        True,
    )
    x = np.concatenate((np.array(lower).T[0], np.array(upper).T[0][::-1][1:]))
    y = np.concatenate((np.array(lower).T[1], np.array(upper).T[1][::-1][1:]))
    aerofoil_section_area = polygon_area(x, y)
    tzd["aerofoil section area"] = aerofoil_section_area
    return aerofoil_section_area


def calculate_rotor_moment(tzd):
    from mpi4py import MPI

    nblades = zone_default(tzd, "number of blades", 3, True)
    blade_material_density = zone_default(
        tzd, "mean blade material density", 200.0, True
    )
    if "aerofoil section area" in tzd:
        aerofoil_section_area = tzd["aerofoil section area"]
    else:
        aerofoil_section_area = calculate_aerofoil_section_area(tzd)
    blade_chord = zone_default(tzd, "blade chord", [[0.0, 0.1], [1.0, 0.1]], True)
    ri = zone_default(tzd, "inner radius", 1.0, True)
    ro = zone_default(tzd, "outer radius", 30.0, True)
    rotor_moment = 0.0
    for r in np.linspace(ri, ro, 100):
        dr = (ro - ri) / 100.0
        c = (
            np.interp(r / ro, np.array(blade_chord).T[0], np.array(blade_chord).T[1])
            * ro
        )
        rotor_moment += r * r * c * c * dr
    rotor_moment = (
        rotor_moment * blade_material_density * aerofoil_section_area * nblades
    )
    tzd["rotor moment of inertia"] = rotor_moment
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("rotor moment of inertia = " + str(rotor_moment))
    return rotor_moment


def create_turbine_segments(
    turbine_zone_dict,
    v0,
    v1,
    v2,
    density,
    turbine_name_dict={},
    turbine_name="",
    annulusVel=None,
    annulusTi=None,
):
    from mpi4py import MPI

    verbose = zone_default(turbine_zone_dict, "verbose", True, False)
    number_of_segments = zone_default(
        turbine_zone_dict, "number of segments", 12, verbose
    )
    rotation_direction = zone_default(
        turbine_zone_dict, "rotation direction", "clockwise", verbose
    )  # when viewed from the front
    ri = zone_default(turbine_zone_dict, "inner radius", 1.0, verbose)
    ro = zone_default(turbine_zone_dict, "outer radius", 30.0, verbose)
    rotor_swept_area = math.pi * (ro * ro - ri * ri)
    disc_normal = zone_default(turbine_zone_dict, "normal", [1.0, 0.0, 0.0], verbose)
    disc_centre = zone_default(turbine_zone_dict, "centre", [0.0, 0.0, 0.0], verbose)
    up = zone_default(turbine_zone_dict, "up", [0.0, 0.0, 1.0], verbose)
    yaw = zone_default(turbine_zone_dict, "yaw", 0.0, verbose)
    auto_yaw = zone_default(turbine_zone_dict, "auto yaw", False, verbose)
    tilt = zone_default(turbine_zone_dict, "tilt", 0.0, verbose)
    inertia = zone_default(turbine_zone_dict, "inertia", False, verbose)
    model = zone_default(turbine_zone_dict, "model", "simple", verbose)
    status = zone_default(turbine_zone_dict, "status", "on", verbose)
    use_glauert_power = zone_default(
        turbine_zone_dict, "use glauert power", False, verbose
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
        print(model)
    induction = "induction" in model
    bet = "blade element theory" in model
    simple = "simple" in model or "direct" in model
    bet_prop = "blade element propellor" in model
    if not (induction or bet or simple or bet_prop):
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("NO MODEL SPECIFIED - DEFAULT TO SIMPLE MODEL")
        simple = True

    annulus_metrics = create_annulus(turbine_zone_dict)
    global bet_kernel_calls

    if inertia:
        dt = zone_default(turbine_zone_dict, "dt", 0.1, verbose)
        if "rotor moment of inertia" in turbine_zone_dict:
            rotor_moment = turbine_zone_dict["rotor moment of inertia"]
        else:
            rotor_moment = calculate_rotor_moment(turbine_zone_dict)

    if bet_prop:
        temp = np.reshape(annulusVel, (-1, 3)).T
        u_ref = math.sqrt(
            np.mean(temp[0]) ** 2 + np.mean(temp[1]) ** 2 + np.mean(temp[2]) ** 2
        )
        nblades = zone_default(turbine_zone_dict, "number of blades", 3, verbose)
        aerofoil_cl = zone_default(
            turbine_zone_dict, "aerofoil cl", [[-90.0, 0.0], [90.0, 0.0]], verbose
        )
        aerofoil_cd = zone_default(
            turbine_zone_dict, "aerofoil cd", [[-90.0, 1.0], [90.0, 1.0]], verbose
        )
        blade_chord = zone_default(
            turbine_zone_dict, "blade chord", [[0.0, 0.1], [1.0, 0.1]], verbose
        )
        blade_twist = zone_default(
            turbine_zone_dict, "blade twist", [[0.0, 25.0], [1.0, 0.0]], verbose
        )  # degrees
        omega = zone_default(turbine_zone_dict, "omega", 0.0, verbose)
        ts = omega * ro / u_ref
        tip_loss_correction = "tip loss correction" in turbine_zone_dict
        if tip_loss_correction:
            tip_loss_correction_model = zone_default(
                turbine_zone_dict, "tip loss correction", "none", verbose
            )
            tip_loss_correction_r = zone_default(
                turbine_zone_dict, "tip loss correction radius", 0.0, verbose
            )
    elif bet:
        bet_kernel_calls = 0
        temp = np.reshape(annulusVel, (-1, 3)).T
        u_ref = math.sqrt(
            np.mean(temp[0]) ** 2 + np.mean(temp[1]) ** 2 + np.mean(temp[2]) ** 2
        )
        nblades = zone_default(turbine_zone_dict, "number of blades", 3, verbose)
        aerofoil_cl = zone_default(
            turbine_zone_dict, "aerofoil cl", [[-90.0, 0.0], [90.0, 0.0]], verbose
        )
        aerofoil_cd = zone_default(
            turbine_zone_dict, "aerofoil cd", [[-90.0, 1.0], [90.0, 1.0]], verbose
        )
        blade_chord = zone_default(
            turbine_zone_dict, "blade chord", [[0.0, 0.1], [1.0, 0.1]], verbose
        )
        blade_twist = zone_default(
            turbine_zone_dict, "blade twist", [[0.0, 25.0], [1.0, 0.0]], verbose
        )  # degrees
        blade_pitch_range = zone_default(
            turbine_zone_dict, "blade pitch range", [-10.0, 10.0], verbose
        )  # degrees
        blade_pitch_step = zone_default(
            turbine_zone_dict, "blade pitch step", 1.0, verbose
        )  # degrees
        blade_pitch = zone_default(
            turbine_zone_dict, "blade pitch", 0.0, verbose
        )  # degrees
        blade_pitch_tol = zone_default(
            turbine_zone_dict, "blade pitch tol", 0.01, verbose
        )  # degrees
        dt = zone_default(turbine_zone_dict, "dt", 0.1, verbose)  # seconds
        rated_power = zone_default(
            turbine_zone_dict, "rated power", 2.3e6, verbose
        )  # Watts
        # m/s environmental limit (2009)
        tip_speed_limit = zone_default(
            turbine_zone_dict, "tip speed limit", 80.0, verbose
        )
        # turbulence intensity range [0,1]
        damage_ti = zone_default(turbine_zone_dict, "damage ti", 0.15, verbose)
        damage_speed = zone_default(
            turbine_zone_dict, "damage speed", 10.0, verbose
        )  # m/s
        friction_loss = zone_default(
            turbine_zone_dict, "friction loss", 0.01, verbose
        )  # friction slow down
        cut_in_speed = zone_default(
            turbine_zone_dict, "cut in speed", 1.0, verbose
        )  # m/s
        cut_out_speed = zone_default(
            turbine_zone_dict, "cut out speed", 99.0, verbose
        )  # m/s
        thrust_factor = zone_default(turbine_zone_dict, "thrust factor", 1.0, verbose)
        omega = zone_default(turbine_zone_dict, "omega", 0.0, verbose)
        tip_loss_correction = "tip loss correction" in turbine_zone_dict
        if tip_loss_correction:
            tip_loss_correction_model = zone_default(
                turbine_zone_dict, "tip loss correction", "none", verbose
            )
            tip_loss_correction_r = zone_default(
                turbine_zone_dict, "tip loss correction radius", 0.0, verbose
            )
        if (u_ref < cut_in_speed) or (u_ref > cut_out_speed):
            omega = 0.0
        ts = omega * ro / u_ref
        if induction:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("CANNOT USE BLADE ELEMENT THEORY WITH INDUCTION MODEL")
            induction = False
    else:
        power_model = zone_default(turbine_zone_dict, "power model", None, False)
        if power_model == "glauert":
            use_glauert_power = True
        u_ref = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
        if "thrust coefficient curve" in turbine_zone_dict:
            tc = np.interp(
                u_ref,
                np.array(turbine_zone_dict["thrust coefficient curve"]).T[0],
                np.array(turbine_zone_dict["thrust coefficient curve"]).T[1],
            )
        elif "thrust coefficient" in turbine_zone_dict:
            tc = turbine_zone_dict["thrust coefficient"]
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("NO THRUST COEFFICIENT SPECIFIED")

        if "tip speed ratio curve" in turbine_zone_dict:
            ts = np.interp(
                u_ref,
                np.array(turbine_zone_dict["tip speed ratio curve"]).T[0],
                np.array(turbine_zone_dict["tip speed ratio curve"]).T[1],
            )
        elif "tip speed ratio" in turbine_zone_dict:
            ts = turbine_zone_dict["tip speed ratio"]
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("NO TIP SPEED RATIO SPECIFIED")
        omega = old_div(ts * u_ref, ro)

    if induction:
        u_infty = u_ref
    else:
        # Assuming 1D momentum theory and the Betz limit
        u_infty = (3.0 / 2.0) * u_ref

    betz_power = 0.5 * density * u_infty ** 3 * rotor_swept_area * (16.0 / 27.0)
    if use_glauert_power:
        if "glauert power curve" not in turbine_zone_dict:
            gp_curve = []
            ts_vals = np.arange(0.0, 20.0, 0.1)
            b_vals = np.arange(0.3334, 0.5, 0.0001)
            peters_lr_vals = []
            for b in b_vals:
                peters_lr_vals.append(
                    old_div(
                        math.sqrt(1.0 + b) * (1.0 - 2.0 * b), math.sqrt(3.0 * b - 1.0)
                    )
                )
            for ts_val in ts_vals:
                b0 = np.interp(ts, peters_lr_vals[::-1], b_vals[::-1])
                y = 3.0 * b0 - 1.0
                gp = 0.5 * density * u_infty ** 3 * rotor_swept_area * glauert_peters(y)
                gp.append([ts_val, gp])
            turbine_zone_dict["glauert power curve"] = gp
        glauert_power = np.interp(
            ts,
            turbine_zone_dict["glauert power curve"].T[0],
            turbine_zone_dict["glauert power curve"].T[1],
        )

    if verbose and (MPI.COMM_WORLD.Get_rank() == 0):
        print("tip speed ratio = " + str(ts))
        print("rotational speed = " + str(omega) + " rad/s")
        print("wind speed = " + str(u_ref) + " m/s")
        print("rotor swept area = " + str(rotor_swept_area) + " m^2")
        print("density = " + str(density) + " kg/m^3")
        print("number of segments = " + str(number_of_segments))

    if not bet_prop:

        def yaw_control(yaw, tilt, disc_normal, up, auto_yaw, annulusVel):
            if auto_yaw:
                temp = np.reshape(annulusVel, (-1, 3)).T
                u_normal = [-np.mean(temp[0]), -np.mean(temp[1]), -np.mean(temp[2])]
                ang = angle_between(u_normal, disc_normal)
                if np.degrees(ang) > 10.0:
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print(
                            "Auto_yaw: geometric disc normal and local flow angle too large: "
                            + str(np.degrees(ang))
                        )
                else:
                    if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                        print(
                            "Auto-yaw: set disc normal to disc-averaged velocity normal"
                        )
                    yaw = math.degrees(angle_between(disc_normal, u_normal))
                    disc_normal = u_normal
            else:
                disc_normal = rotate(disc_normal, up, math.radians(yaw))
            tilt_axis = np.cross(disc_normal, up)
            disc_normal = rotate(disc_normal, tilt_axis, math.radians(tilt))
            if np.dot(disc_normal, up) < 0.0 and MPI.COMM_WORLD.Get_rank() == 0:
                print("Tilting wrong way!")
            return yaw, unit_vector(disc_normal)

        yaw, disc_normal = yaw_control(yaw, tilt, disc_normal, up, auto_yaw, annulusVel)
        if (MPI.COMM_WORLD.Get_rank() == 0) and verbose:
            print("disc_normal = " + str(disc_normal))

    if bet_prop:
        annulus = []
        theta = 0.0
        total_area = 0.0
        total_thrust = 0.0
        total_torque = 0.0
        angular_induction = 0.0

        avindex = 0
        # annulus_metrics = (r, dr, i * dtheta, dtheta, disc_pt[0], disc_pt[1], disc_pt[2])
        for am in annulus_metrics:
            rp = am[0] + 0.5 * am[1]
            da = am[3] * rp * am[1]
            ulocal = np.reshape(annulusVel, (-1, 3))[avindex]
            rvec = unit_vector(
                [am[4] - disc_centre[0], am[5] - disc_centre[1], am[6] - disc_centre[2]]
            )
            if rotation_direction == "clockwise":
                local_omega_vec = np.cross(rvec, disc_normal)
            else:
                local_omega_vec = np.cross(disc_normal, rvec)
            v_n = -np.dot(ulocal, disc_normal)
            v_r = np.dot(ulocal, local_omega_vec)
            if (abs((rp * omega) + v_r)) > 0.0:
                theta_rel = math.atan(v_n / ((rp * omega) - v_r))
            else:
                theta_rel = math.pi / 2.0
            urel = math.sqrt((rp * omega - v_r) ** 2 + v_n ** 2)
            beta_twist = np.interp(
                old_div(rp, ro), np.array(blade_twist).T[0], np.array(blade_twist).T[1]
            )
            chord = (
                np.interp(
                    old_div(rp, ro),
                    np.array(blade_chord).T[0],
                    np.array(blade_chord).T[1],
                )
                * ro
            )
            beta = math.radians(beta_twist)
            alpha = beta - theta_rel
            cl = np.interp(
                math.degrees(alpha),
                np.array(aerofoil_cl).T[0],
                np.array(aerofoil_cl).T[1],
            )
            cd = np.interp(
                math.degrees(alpha),
                np.array(aerofoil_cd).T[0],
                np.array(aerofoil_cd).T[1],
            )
            if tip_loss_correction:
                rstar = tip_loss_correction_r * ro
                if rp > rstar:
                    tip_loss_factor = math.sqrt(
                        1.0 - ((rp - rstar) / (ro - rstar)) ** 2
                    )
                    cl = cl * tip_loss_factor
                    cd = cd * tip_loss_factor
            f_L = cl * 0.5 * density * urel ** 2 * chord
            f_D = cd * 0.5 * density * urel ** 2 * chord
            F_L = old_div(nblades, (2.0 * math.pi * rp)) * f_L
            F_D = old_div(nblades, (2.0 * math.pi * rp)) * f_D
            dt = -(F_L * math.cos(theta_rel) - F_D * math.sin(theta_rel)) * da
            dq = -(F_L * math.sin(theta_rel) + F_D * math.cos(theta_rel)) * da
            if rotation_direction == "anticlockwise":
                dq = -dq
            annulus.append((dt, dq, am[0], am[1], am[2], am[3]))
            total_area += da
            total_thrust += dt
            total_torque += math.fabs(dq * rp)
            avindex = avindex + 1
        total_power = total_torque * omega

    elif bet:
        bet_kernel_calls = 0

        # pre-populate the beta_twist and chord values
        for i in range(len(annulus_metrics)):
            rp = annulus_metrics[i][0] + 0.5 * annulus_metrics[i][1]
            beta_twist = np.interp(
                (rp / ro), np.array(blade_twist).T[0], np.array(blade_twist).T[1]
            )
            chord = (
                np.interp(
                    (rp / ro), np.array(blade_chord).T[0], np.array(blade_chord).T[1]
                )
                * ro
            )
            annulus_metrics[i] = annulus_metrics[i] + (beta_twist, chord)

        annulus = len(annulus_metrics) * [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]

        def bet_kernel(beta_pitch):
            global bet_kernel_calls
            bet_kernel_calls = bet_kernel_calls + 1
            total_area = 0.0
            total_thrust = 0.0
            total_torque = 0.0
            angular_induction = 0.0
            avindex = 0
            # check whether any segments are at negative angle of attack.
            alpha_positive = True
            # annulus_metrics = (r, dr, i * dtheta, dtheta, disc_pt[0], disc_pt[1], disc_pt[2], beta_twist, chord)
            for am in annulus_metrics:
                rp = am[0] + 0.5 * am[1]
                da = am[3] * rp * am[1]
                tilt_axis = np.cross(up, disc_normal)
                rvec0 = rotate(up, tilt_axis, math.radians(tilt))
                rvec = unit_vector(
                    [
                        am[4] - disc_centre[0],
                        am[5] - disc_centre[1],
                        am[6] - disc_centre[2],
                    ]
                )
                ulocal = np.reshape(annulusVel, (-1, 3))[avindex]
                if rotation_direction == "clockwise":
                    local_omega_vec = np.cross(rvec, disc_normal)
                else:
                    local_omega_vec = np.cross(disc_normal, rvec)
                omega_air = np.dot(local_omega_vec, ulocal) / rp
                omega_rel = omega - omega_air
                u_ref_local = -np.dot(ulocal, disc_normal)
                urel = math.sqrt((rp * omega_rel) ** 2 + u_ref_local ** 2)
                if (rp * omega_rel) > 0.0:
                    theta_rel = math.atan(old_div(u_ref_local, (rp * omega_rel)))
                else:
                    theta_rel = math.pi / 2.0
                beta_twist = am[7]
                chord = am[8]
                beta = math.radians(beta_pitch + beta_twist)
                alpha = theta_rel - beta
                if alpha < 0.0:
                    alpha_positive = False
                cl = np.interp(
                    math.degrees(alpha),
                    np.array(aerofoil_cl).T[0],
                    np.array(aerofoil_cl).T[1],
                )
                if tip_loss_correction:
                    if tip_loss_correction_model == "elliptic":
                        tlc = math.sqrt(1.0 - (rp / ro) ** 2)
                    elif tip_loss_correction_model == "acos-fit":
                        tlc = (2.0 / math.pi) * math.acos(
                            math.exp(-63.0 * (1.0 - (rp / ro) ** 2))
                        )
                    elif tip_loss_correction_model == "acos shift-fit":
                        tlc = (2.0 / math.pi) * math.acos(
                            math.exp(-48.0 * (1.0 - (rp / ro) ** 2) - 0.5)
                        )
                    elif tip_loss_correction_model == "f-fit":
                        tlc = 1.0 - 2.5 * ((1.0 - (rp / ro) ** 2) ** 0.39) / (
                            (2.0 - (rp / ro) ** 2) ** 64
                        )
                    else:
                        tlc = 1.0
                    cl = cl * tlc  # only apply to lift, not drag.
                cd = np.interp(
                    math.degrees(alpha),
                    np.array(aerofoil_cd).T[0],
                    np.array(aerofoil_cd).T[1],
                )
                f_L = cl * 0.5 * density * urel ** 2 * chord
                f_D = cd * 0.5 * density * urel ** 2 * chord
                F_L = old_div(nblades, (2.0 * math.pi * rp)) * f_L
                F_D = old_div(nblades, (2.0 * math.pi * rp)) * f_D
                dt = (
                    (F_L * math.cos(theta_rel) + F_D * math.sin(theta_rel)) * da
                ) * thrust_factor
                dq = -(F_L * math.sin(theta_rel) - F_D * math.cos(theta_rel)) * da
                if rotation_direction == "clockwise":
                    dq = -dq
                annulus[avindex] = (dt, dq, am[0], am[1], am[2], am[3])
                total_area += da
                total_thrust += dt
                total_torque += math.fabs(dq * rp)
                angular_induction += omega_air * da
                avindex = avindex + 1
            if not alpha_positive:
                if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                    print("WARNING - negative angle of attack ")
            angular_induction = angular_induction / total_area
            return total_torque, total_area, total_thrust, angular_induction, annulus

        def turbine_controller(
            omega,
            rated_power,
            tip_speed_limit,
            damage_ti,
            damage_speed,
            status,
            u_ref,
            cut_in_speed,
            cut_out_speed,
            blade_pitch,
            blade_pitch_step,
        ):
            if status == "off":
                omega = 0.0
            if (u_ref < cut_in_speed) or (u_ref > cut_out_speed):
                omega = 0.0
            # Make sure we are not exceeding the blade tip speed limit
            omega = min(omega, tip_speed_limit / ro) * (1.0 - friction_loss)
            # work out whether we are feathering the blades to shed power:
            blade_pitch_low = max(blade_pitch - blade_pitch_step, blade_pitch_range[0])
            blade_pitch_high = min(blade_pitch + blade_pitch_step, blade_pitch_range[1])
            blade_pitch_opt = np.min(
                gss(bet_kernel, blade_pitch_low, blade_pitch_high, blade_pitch_tol)
            )
            if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                print("Blade pitch opt = " + str(blade_pitch_opt))
            maximum_torque = bet_kernel(blade_pitch_opt)[0]
            if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                print("Maximum torque = " + str(maximum_torque))
            if maximum_torque * omega > rated_power:
                if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                    print("Feathering to reduce power below rated power")
                # Construct a power curve against blade pitch for the current rate of rotation
                blade_pitch_curve = []
                for b in np.arange(blade_pitch_low, blade_pitch_opt, blade_pitch_tol):
                    blade_pitch_curve.append([b, omega * bet_kernel(b)[0]])
                if (MPI.COMM_WORLD.Get_rank() == 0) and verbose:
                    print(
                        "Points on blade pitch curve : " + str(len(blade_pitch_curve))
                    )
                if len(blade_pitch_curve) > 1:
                    # Look up the blade pitch that recovers the rated power
                    blade_pitch = np.interp(
                        rated_power,
                        np.array(blade_pitch_curve).T[1],
                        np.array(blade_pitch_curve).T[0],
                    )
                else:
                    blade_pitch = blade_pitch_low
                if (MPI.COMM_WORLD.Get_rank() == 0) and verbose:
                    print("Rated power blade pitch =  : " + str(blade_pitch))
                total_torque, total_area, total_thrust, angular_induction, annulus = bet_kernel(
                    blade_pitch
                )
                torque_blades = 0.0
            else:
                # Use half of the available torque to accelerate the blades and half to provide power to the generator
                # unless this exceeds a 5% increase in the rate of rotation or the tip speed limit or the rated power.
                blade_pitch = blade_pitch_opt
                total_torque, total_area, total_thrust, angular_induction, annulus = bet_kernel(
                    blade_pitch
                )
                torque_blades = total_torque / 2.0  # Completely arbitrary.
                # modfy the tip speed limit if there is an rpm ramp:
                if "rpm ramp" in turbine_zone_dict:
                    rr = np.asarray(turbine_zone_dict["rpm ramp"])
                    if (u_ref > cut_in_speed) and (u_ref < rr[1][1]):
                        tip_speed_limit = min(
                            tip_speed_limit,
                            ro
                            * np.interp(u_ref, rr.T[0], rr.T[1])
                            * 2.0
                            * np.pi
                            / 60.0,
                        )
                        if MPI.COMM_WORLD.Get_rank() == 0:
                            print(
                                "RPM LIMIT: tip speed limit = " + str(tip_speed_limit)
                            )
                if rotor_moment > 0.0:
                    if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                        print("BET - Rotor Moment Model")
                    torque_blades = min(
                        torque_blades,
                        ((tip_speed_limit / ro) - omega) * rotor_moment / dt,
                    )
                    omega = omega + (torque_blades * dt) / rotor_moment
                else:
                    if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                        print("BET - USING ZERO INERTIA MODEL")
                    omega = omega * 1.1  # Limit the increase to 10%
                    # Do not allow the rotor to over-speed
                    omega = min(omega, tip_speed_limit / ro)
                    torque_blades = 0.0
                # Do not exceed the (approximated) rated power
                torque_power = total_torque - torque_blades
                if torque_power > 0.0:
                    omega = min(omega, rated_power / torque_power)
            # work out whether we are stowing the blades to prevent damage
            damage_alert = False
            if (MPI.COMM_WORLD.Get_rank() == 0) and verbose:
                print("Maximum onset TI: " + str(np.max(annulusTi)))
            for aindex in range(len(np.reshape(annulusVel, (-1, 3)))):
                ulocal = np.reshape(annulusVel, (-1, 3))[aindex]
                ulocalmag = math.sqrt(ulocal[0] ** 2 + ulocal[1] ** 2 + ulocal[2] ** 2)
                tilocal = annulusTi[aindex]
                if (ulocalmag > damage_speed) and (tilocal > damage_ti):
                    damage_alert = True
            if damage_alert:
                if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                    print("Damage alert detected - stowing turbine")
                if omega > 0.1:
                    omega = omega * 0.9  # slow down the turbine
                else:
                    omega = 0.0
                torque_blades = 0.0
            torque_power = total_torque - torque_blades
            return (
                blade_pitch,
                omega,
                torque_blades,
                torque_power,
                total_torque,
                total_area,
                total_thrust,
                angular_induction,
                annulus,
            )

        blade_pitch, omega, torque_blades, torque_power, total_torque, total_area, total_thrust, angular_induction, annulus = turbine_controller(
            omega,
            rated_power,
            tip_speed_limit,
            damage_ti,
            damage_speed,
            status,
            u_ref,
            cut_in_speed,
            cut_out_speed,
            blade_pitch,
            blade_pitch_step,
        )

        turbine_power = torque_power * math.fabs(omega)
        total_power = total_torque * omega

        turbine_zone_dict["omega"] = omega
        turbine_zone_dict["blade pitch"] = blade_pitch
        turbine_zone_dict["yaw"] = yaw

        if MPI.COMM_WORLD.Get_rank() == 0:
            turbine_name_dict[turbine_name + "_tilt"] = tilt
            turbine_name_dict[turbine_name + "_yaw"] = yaw
            turbine_name_dict[turbine_name + "_blade_pitch"] = blade_pitch
            turbine_name_dict[turbine_name + "_ang_ind"] = angular_induction / omega
            turbine_name_dict[turbine_name + "_thrust"] = total_thrust

            if verbose:
                print("status = " + str(status))
                print("rotation rate = " + str(omega) + " radians / sec")
                print("blade pitch = " + str(blade_pitch) + " degrees")
                print("torque power = " + str(torque_power) + " Joules/rad")
                print("torque blades = " + str(torque_blades) + " Joules/rad")
                print(
                    "angular induction = "
                    + str(100.0 * angular_induction / omega)
                    + "%"
                )
                print("bet kernel calls = " + str(bet_kernel_calls))

        # TODO - add turbulence source in wake of turbine.
        # TODO - add restart capability (read data from CSV report file)

    else:
        if power_model == "betz":
            power = betz_power
        elif power_model == "glauert":
            power = glauert_power
        elif "turbine power curve" in turbine_zone_dict:
            power = np.interp(
                u_ref,
                np.array(turbine_zone_dict["turbine power curve"]).T[0],
                np.array(turbine_zone_dict["turbine power curve"]).T[1],
            )
        elif "turbine power" in turbine_zone_dict:
            power = turbine_zone_dict["turbine power"]
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("NO POWER MODEL SPECIFIED - USING BETZ LIMIT")
            power = betz_power
        annulus = []
        # Induction assumes that u_ref is u_inf.  Direct (Simple) assumes that u_ref is at disk.
        if induction:
            # Momentum theory: Ct = 4 * a * ( 1 - a), Cp = 4 * a * ( 1 - a)^2, Betz Optimal rotor: a = 1/3
            if tc > 0.999:
                print("INDUCTION MODEL TC CANNOT EXCEED 1.0: " + str(tc))
            ind_fac = old_div(
                (4.0 - math.sqrt(4.0 * 4.0 - 4.0 * 4.0 * tc)), (2.0 * 4.0)
            )
            if verbose and (MPI.COMM_WORLD.Get_rank() == 0):
                print("Induction factor: ", str(ind_fac))
        dtheta = math.radians(360.0 / number_of_segments)
        target_torque = old_div(power, omega)
        theta = 0.0
        total_area = 0.0
        total_thrust = 0.0
        total_torque = 0.0
        for i in range(number_of_segments):
            r = ri
            while r < ro:
                dr = old_div(dtheta * max(r,0.01*ro), (1.0 - 0.5 * dtheta))
                max_r = r + dr
                if max_r > ro:
                    dr = ro - r
                rp = r + 0.5 * dr
                da = dtheta * rp * dr
                if induction:
                    dt = (
                        0.5
                        * density
                        * u_ref
                        * u_ref
                        * da
                        * 4.0
                        * ind_fac
                        * (1.0 - ind_fac)
                    )
                    lambda_r = old_div(rp * omega, u_ref)
                    if lambda_r > 0.0:
                        ang_ind_fac = -0.5 + math.sqrt(
                            0.25 + old_div(ind_fac * (1.0 - ind_fac), lambda_r ** 2)
                        )
                    else:
                        ang_ind_fac = 0.0
                    dq = (
                        4.0
                        * ang_ind_fac
                        * (1.0 - ind_fac)
                        * 0.5
                        * density
                        * u_ref
                        * omega
                        * rp
                        * rp
                        * da
                        / rp
                    )
                else:
                    dt = 0.5 * density * u_ref * u_ref * da * tc
                    dq = old_div((target_torque * da), (rotor_swept_area * rp))
                if rotation_direction == "anticlockwise":
                    dq = -dq
                annulus.append((dt, dq, r, dr, i * dtheta, dtheta))
                total_area += da
                total_thrust += dt
                total_torque += math.fabs(dq * rp)
                r = r + dr
        specified_thrust = 0.5 * rotor_swept_area * density * u_ref * u_ref * tc

        if MPI.COMM_WORLD.Get_rank() == 0:
            if verbose:
                print("thrust coefficient [specified] = " + str(tc))
                print("thrust [specified] = " + str(specified_thrust))
                print("model specified power = " + str(power) + " Watts")
                print("target torque = " + str(target_torque) + " Joules/rad")

    if MPI.COMM_WORLD.Get_rank() == 0:
        total_power = total_torque * omega
        turbine_name_dict[turbine_name + "_power"] = total_power
        turbine_name_dict[turbine_name + "_uref"] = u_ref
        turbine_name_dict[turbine_name + "_omega"] = omega
        turbine_name_dict[turbine_name + "_thrust"] = total_thrust

        if verbose:
            print("total area = " + str(total_area) + " m^2")
            print("turbine power = " + str(total_power) + " Watts")
            print("total thrust = " + str(total_thrust) + " Newtons")
            print("total torque = " + str(total_torque) + " Joules/rad")
            if not bet_prop:
                print(
                    "% of Betz limit power "
                    + str(old_div(100.0 * total_power, betz_power))
                    + "%"
                )
                if use_glauert_power:
                    print(
                        "% of Glauert optimal power "
                        + str(old_div(100.0 * total_power, glauert_power))
                        + "%"
                    )
    return annulus


def project_to_plane(pt, plane_point, plane_normal):
    from numpy import dot

    # print pt,plane_point,plane_normal
    return pt - dot(pt - plane_point, plane_normal) * plane_normal


# def clockwise_angle(up_vector, pt_vector, plane_normal):
#    from numpy import zeros, array, dot, linalg, cross
#
#    v_dot = dot(up_vector, pt_vector)
#    v_det = dot(plane_normal, cross(up_vector, pt_vector))
#
#    r = math.atan2(v_det, v_dot)
#
#    if r < 0:
#        r += 2.0 * math.pi
#
#    return r


def convolution(
    disc,
    disc_centre,
    disc_radius,
    disc_normal,
    disc_up,
    cell_centre_list,
    cell_volume_list,
):
    from mpi4py import MPI
    import libconvolution as cv
    from numpy import zeros, array, dot, linalg, cross, asarray, ndarray

    cell_centre_list_np = asarray(cell_centre_list)
    cell_volume_list_np = asarray(cell_volume_list)
    kernel_3d = False

    weighted_sum = np.zeros(len(disc))

    weighted_sum = cv.convolution_2dkernel_weights(
        disc,
        disc_centre,
        disc_radius,
        disc_normal,
        disc_up,
        cell_centre_list_np,
        cell_volume_list_np,
    )
    # Need to reduce weighted sum over all processes
    totals = np.zeros_like(weighted_sum)

    MPI.COMM_WORLD.Allreduce(weighted_sum, totals, op=MPI.SUM)
    weighted_sum = totals

    thrust_check_total = 0

    cell_force = np.zeros(len(cell_centre_list_np) * 3)
    thrust_check = cv.convolution_2dkernel_force(
        disc,
        disc_centre,
        disc_radius,
        disc_normal,
        disc_up,
        cell_centre_list_np,
        cell_volume_list_np,
        weighted_sum,
        cell_force,
    )

    thrust_check_array = np.array([thrust_check])
    thrust_check_total_array = np.array([0.0])

    MPI.COMM_WORLD.Allreduce(thrust_check_array, thrust_check_total_array, op=MPI.SUM)
    thrust_check_total = thrust_check_total_array[0]
    thrust_check = thrust_check_total

    # if MPI.COMM_WORLD.Get_rank() == 0:
    #    print 'Convolved total thrust: ',thrust_check

    # thrust_check = 0.0
    total_thrust = 0.0
    for idx, w in enumerate(weighted_sum):
        segment = disc[idx]
        # if w > 0.0:
        #    thrust_check += segment[0]
        total_thrust += segment[0]

    # if MPI.COMM_WORLD.Get_rank() == 0:
    #    print 'Specified total thrust: ',total_thrust

    # Broken: Cell_force_scaled will have a different struct to cell_force
    if thrust_check > 0.0:
        thrust_factor = old_div(total_thrust, thrust_check)
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #    print 'Scaling thrust: ', thrust_factor
        cell_force_scaled = []
        for cell in range(old_div(len(cell_force), 3)):
            cell_force_scaled.append(
                (
                    cell_force[cell * 3 + 0] * thrust_factor,
                    cell_force[cell * 3 + 1] * thrust_factor,
                    cell_force[cell * 3 + 2] * thrust_factor,
                )
            )
        return cell_force_scaled
    else:
        cell_force = ndarray.tolist(cell_force)
        cell_array = iter(cell_force)
        return list(zip(cell_array, cell_array, cell_array))

    cell_force = ndarray.tolist(cell_force)
    cell_array = iter(cell_force)
    return list(zip(cell_array, cell_array, cell_array))


def convolution2(
    disc,
    disc_centre,
    disc_radius,
    disc_normal,
    disc_up,
    cell_centre_list,
    cell_volume_list,
):
    from mpi4py import MPI

    from numpy import zeros, array, dot, linalg, cross

    # from zutil import R_2vect

    cell_force = []

    thrust_check = 0.0

    # Transform disc points to actual location
    disc_pt_list = []
    for segment in disc:
        r = segment[2]
        dr = segment[3]
        theta = segment[4]
        dtheta = segment[5]

        disc_r = r + 0.5 * dr
        disc_theta = theta + 0.5 * dtheta

        disc_pt = array(
            [disc_r * math.cos(disc_theta), disc_r * math.sin(disc_theta), 0.0]
        )
        # print disc_pt
        # Rotate so that z points in the direction of the normal
        R = zeros((3, 3))
        vector_orig = array([0.0, 0.0, 1.0])
        vector_fin = zeros(3)
        for i in range(3):
            vector_fin[i] = disc_normal[i]
        R_2vect(R, vector_orig, vector_fin)

        disc_pt = dot(R, disc_pt)

        # translate to disc centre
        for i in range(3):
            disc_pt[i] += disc_centre[i]

        disc_pt_list.append(disc_pt)

    kernel_3d = False

    weighted_sum = np.zeros(len(disc))

    for cell_idx, cell_centre in enumerate(cell_centre_list):

        cell_dt = [0.0, 0.0, 0.0]
        cell_dq = [0.0, 0.0, 0.0]

        if kernel_3d:
            for idx, segment in enumerate(disc):
                dt = segment[0]
                dq = segment[1]
                r = segment[2]
                dr = segment[3]
                theta = segment[4]
                dtheta = segment[5]

                disc_r = r + 0.5 * dr
                disc_theta = theta + 0.5 * dtheta

                area = dtheta * disc_r * dr

                dt_per_area = old_div(dt, area)
                dq_per_area = old_div(dq, area)

                disc_pt = disc_pt_list[idx]
                # print disc_pt

                distance_vec = array(cell_centre) - disc_pt
                distance = math.sqrt(dot(distance_vec, distance_vec))
                unit_distance_vec = old_div(distance_vec, distance)

                epsilon = 2.0 * dr

                # 3D kernel
                eta = (
                    1.0
                    / ((epsilon ** 2) * math.sqrt(math.pi) ** 3)
                    * math.exp(-(old_div(distance, epsilon)) ** 2)
                )
                # /math.fabs(dot(disc_normal,unit_distance_vec))

                weighted_sum[idx] += max(1.0e-16, eta * cell_volume_list[cell_idx])

                # 1D kernel
                # eta = 1.0/(epsilon*math.sqrt(math.pi)) * math.exp(-(distance/epsilon)**2)

                # print eta,cell_centre,disc_pt

                # Set thrust force
                # for i in range(3):
                #    cell_dt[i] += dt_per_area*disc_normal[i]*eta

                # Set torque force
                # Vector for centre to pt
                # disc_vec = disc_pt - array(disc_centre)
                # unit_disc_vec = disc_vec/linalg.norm(disc_vec)
                # torque_vector = cross(unit_disc_vec,disc_normal)

                # for i in range(3):
                #    cell_dq[i] += dq_per_area*torque_vector[i]*eta
        else:
            # Need for find nearest segment

            plane_pt = project_to_plane(
                array(cell_centre), array(disc_centre), array(disc_normal)
            )

            # print 'Plane pt: ' + str(plane_pt)

            plane_pt_radius = linalg.norm(plane_pt - disc_centre)
            plane_pt_theta = 0.0
            if plane_pt_radius > 0.0:
                # plane_pt_theta  = math.acos(dot([0.0,0.0,1.0],(plane_pt-disc_centre)/plane_pt_radius))
                plane_pt_theta = clockwise_angle(
                    disc_up,
                    old_div((plane_pt - disc_centre), plane_pt_radius),
                    array(disc_normal),
                )

            min_index = -1
            for idx, segment in enumerate(disc):
                r = segment[2]
                dr = segment[3]
                theta = segment[4]
                dtheta = segment[5]

                if plane_pt_theta >= theta and plane_pt_theta <= theta + dtheta:
                    if plane_pt_radius >= r and plane_pt_radius <= r + dr:
                        min_index = idx
                        break

            if min_index != -1:
                segment = disc[min_index]

                dt = segment[0]
                dq = segment[1]
                r = segment[2]
                dr = segment[3]
                theta = segment[4]
                dtheta = segment[5]

                disc_r = r + 0.5 * dr
                disc_theta = theta + 0.5 * dtheta

                area = dtheta * disc_r * dr

                dt_per_area = old_div(dt, area)
                dq_per_area = old_div(dq, area)

                distance_vec = array(cell_centre) - plane_pt
                distance = math.sqrt(dot(distance_vec, distance_vec))

                # epsilon = 2.0*dr
                epsilon = 0.2 * disc_radius

                # 1D kernel
                eta = (
                    1.0
                    / (epsilon * math.sqrt(math.pi))
                    * math.exp(-(old_div(distance, epsilon)) ** 2)
                )

                # Add max as eta may be zero due to underflow in the exponent
                # term
                weighted_sum[min_index] += max(
                    1.0e-16, eta * cell_volume_list[cell_idx]
                )

    # Need to reduce weighted sum over all processes
    totals = np.zeros_like(weighted_sum)

    MPI.COMM_WORLD.Allreduce(weighted_sum, totals, op=MPI.SUM)
    weighted_sum = totals

    for cell_idx, cell_centre in enumerate(cell_centre_list):

        cell_dt = [0.0, 0.0, 0.0]
        cell_dq = [0.0, 0.0, 0.0]

        if kernel_3d:
            for idx, segment in enumerate(disc):
                dt = segment[0]
                dq = segment[1]
                r = segment[2]
                dr = segment[3]
                theta = segment[4]
                dtheta = segment[5]

                disc_r = r + 0.5 * dr
                disc_theta = theta + 0.5 * dtheta

                area = dtheta * disc_r * dr

                dt_per_area = old_div(dt, area)
                dq_per_area = old_div(dq, area)

                disc_pt = disc_pt_list[idx]
                # print disc_pt

                distance_vec = array(cell_centre) - disc_pt
                distance = math.sqrt(dot(distance_vec, distance_vec))
                unit_distance_vec = old_div(distance_vec, distance)

                epsilon = 2.0 * dr

                # 3D kernel
                eta = (
                    1.0
                    / ((epsilon ** 2) * math.sqrt(math.pi) ** 3)
                    * math.exp(-(old_div(distance, epsilon)) ** 2)
                )
                # /math.fabs(dot(disc_normal,unit_distance_vec))

                redistribution_weight = weighted_sum[idx]

                # 1D kernel
                # eta = 1.0/(epsilon*math.sqrt(math.pi)) * math.exp(-(distance/epsilon)**2)

                # print eta,cell_centre,disc_pt

                # Set thrust force
                for i in range(3):
                    cell_dt[i] += (
                        old_div(dt_per_area * area, redistribution_weight)
                        * disc_normal[i]
                        * eta
                    )

                # Set torque force
                # Vector for centre to pt
                disc_vec = disc_pt - array(disc_centre)
                disc_vec_mag = linalg.norm(disc_vec)
                unit_disc_vec = old_div(disc_vec, disc_vec_mag)
                torque_vector = cross(disc_normal, unit_disc_vec)

                # Note converting torque to a force
                for i in range(3):
                    cell_dq[i] += old_div(
                        dq_per_area * area, redistribution_weight
                    ) * old_div(torque_vector[i] * eta / disc_vec_mag)
        else:
            # Need for find nearest segment

            plane_pt = project_to_plane(
                array(cell_centre), array(disc_centre), array(disc_normal)
            )

            # print 'Plane pt: ' + str(plane_pt)

            plane_pt_radius = linalg.norm(plane_pt - disc_centre)

            plane_pt_theta = 0.0
            if plane_pt_radius > 0.0:
                # plane_pt_theta  = math.acos(dot([0.0,0.0,1.0],(plane_pt-disc_centre)/plane_pt_radius))
                plane_pt_theta = clockwise_angle(
                    disc_up,
                    old_div((plane_pt - disc_centre), plane_pt_radius),
                    array(disc_normal),
                )

            min_index = -1
            for idx, segment in enumerate(disc):
                r = segment[2]
                dr = segment[3]
                theta = segment[4]
                dtheta = segment[5]

                if plane_pt_theta >= theta and plane_pt_theta <= theta + dtheta:
                    if plane_pt_radius >= r and plane_pt_radius <= r + dr:
                        min_index = idx
                        break

            if min_index != -1:
                segment = disc[min_index]

                dt = segment[0]
                dq = segment[1]
                r = segment[2]
                dr = segment[3]
                theta = segment[4]
                dtheta = segment[5]

                disc_r = r + 0.5 * dr
                disc_theta = theta + 0.5 * dtheta

                area = dtheta * disc_r * dr

                dt_per_area = old_div(dt, area)
                dq_per_area = old_div(dq, area)

                distance_vec = array(cell_centre) - plane_pt
                distance = math.sqrt(dot(distance_vec, distance_vec))

                # epsilon = 2.0*dr
                epsilon = 0.2 * disc_radius

                # 1D kernel
                eta = (
                    1.0
                    / (epsilon * math.sqrt(math.pi))
                    * math.exp(-(old_div(distance, epsilon)) ** 2)
                )

                redistribution_weight = weighted_sum[min_index]

                # print redistribution_weight

                # print dt,eta,cell_centre,plane_pt

                # Set thrust force
                for i in range(3):
                    cell_dt[i] += (
                        old_div(dt_per_area * area, redistribution_weight)
                        * disc_normal[i]
                        * eta
                    )

                # Set torque force
                # Vector for centre to pt
                disc_vec = plane_pt - array(disc_centre)
                disc_vec_mag = linalg.norm(disc_vec)
                unit_disc_vec = old_div(disc_vec, disc_vec_mag)
                torque_vector = cross(disc_normal, unit_disc_vec)

                # Note converting torque to force
                for i in range(3):
                    cell_dq[i] += old_div(
                        dq_per_area * area, redistribution_weight
                    ) * old_div(torque_vector[i] * eta / disc_vec_mag)

        cell_force.append(
            (cell_dt[0] + cell_dq[0], cell_dt[1] + cell_dq[1], cell_dt[2] + cell_dq[2])
        )

        thrust_check += dot(cell_force[-1], disc_normal) * cell_volume_list[cell_idx]

    thrust_check_total = 0

    thrust_check_array = np.array([thrust_check])
    thrust_check_total_array = np.array([0.0])
    MPI.COMM_WORLD.Allreduce(thrust_check_array, thrust_check_total_array, op=MPI.SUM)
    thrust_check_total = thrust_check_total_array[0]
    thrust_check = thrust_check_total

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Convolved total thrust: ", thrust_check)

    # thrust_check = 0.0
    total_thrust = 0.0
    for idx, w in enumerate(weighted_sum):
        segment = disc[idx]
        # if w > 0.0:
        #    thrust_check += segment[0]
        total_thrust += segment[0]

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Specified total thrust: ", total_thrust)

    if thrust_check > 0.0:
        thrust_factor = old_div(total_thrust, thrust_check)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Scaling thrust: ", thrust_factor)
        cell_force_scaled = []
        for cell in cell_force:
            force = cell
            cell_force_scaled.append(
                (
                    force[0] * thrust_factor,
                    force[1] * thrust_factor,
                    force[2] * thrust_factor,
                )
            )

        return cell_force_scaled
    else:
        return cell_force

    return cell_force


def test_convolution():
    a = create_turbine_segments(0.7, 0.1, 1.0, 6.0, 1, 1)
    b = convolution(a, (0, 0, 0), (1, 0, 0), [(0.0, 0.0, 1.5)], [1.0])
    b = convolution(a, (0, 0, 0), (1, 0, 0), [(0.0, 0.0, 0.99)], [1.0])
    b = convolution(a, (0, 0, 0), (1, 0, 0), [(2.0, 0.0, 0.5)], [1.0])

    print(b)
