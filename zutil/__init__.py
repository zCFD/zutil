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

import math
import sys
import os
import numpy as np


def get_parameters_from_file(filename):
    conf = filename
    mymodule = __import__(conf)
    # Force a reload just in case it has already been loaded
    reload(mymodule)
    return getattr(sys.modules[conf], 'parameters')


def include(filename):
    """
    include a file by executing it. This imports everything including
    variables into the calling module
    """
    if os.path.exists(filename):
        execfile(filename)


def get_zone_info(module_name):
    try:
        #mymodule = __import__(module_name)
        # Force a reload just in case it has already been loaded
        # reload(mymodule)
        # return mymodule
        import importlib
        return importlib.import_module(module_name)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        return None


def get_default_zone_info():
    import inspect
    _, filename, linenumber, _, _, _ = inspect.stack()[1]
    return get_zone_info(os.path.split(os.path.splitext(filename)[0])[1] +
                         '_zone')


def find_next_zone(parameters, zone_prefix):
    # Find next available
    found = False
    counter = 1
    while not found:
        key = zone_prefix + '_' + str(counter)
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

    return [-wind_speed * math.sin(math.radians(wind_dir_degree)),
            -wind_speed * math.cos(math.radians(wind_dir_degree)),
            0.0]


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

    beta = math.asin(vec[1] / mag)
    alpha = math.acos(vec[0] / (mag * math.cos(beta)))
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
    rot[0] = math.cos(alpha) * math.cos(beta) * vec[0] + math.sin(beta) * \
        vec[1] + math.sin(alpha) * math.cos(beta) * vec[2]
    rot[1] = -math.cos(alpha) * math.sin(beta) * vec[0] + math.cos(beta) * \
        vec[1] - math.sin(alpha) * math.sin(beta) * vec[2]
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
    vector_orig = vector_orig / norm(vector_orig)
    vector_fin = vector_fin / norm(vector_fin)

    # The rotation axis (normalised).
    axis = cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

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
    R[0, 0] = 1.0 + (1.0 - ca) * (x**2 - 1.0)
    R[0, 1] = -z * sa + (1.0 - ca) * x * y
    R[0, 2] = y * sa + (1.0 - ca) * x * z
    R[1, 0] = z * sa + (1.0 - ca) * x * y
    R[1, 1] = 1.0 + (1.0 - ca) * (y**2 - 1.0)
    R[1, 2] = -x * sa + (1.0 - ca) * y * z
    R[2, 0] = -y * sa + (1.0 - ca) * x * z
    R[2, 1] = x * sa + (1.0 - ca) * y * z
    R[2, 2] = 1.0 + (1.0 - ca) * (z**2 - 1.0)


def vector_vector_rotate(vec, axis, origin, theta):
    # Rotate vector
    temp = [0.0, 0.0, 0.0]

    temp[0] = ((origin[0] * (axis[1] * axis[1] + axis[2] * axis[2]) -
                axis[0] * (origin[1] * axis[1] + origin[2] * axis[2] - dot(axis, vec))) *
               (1.0 - math.cos(theta)) +
               vec[0] * math.cos(theta) +
               (-origin[2] * axis[1] + origin[1] * axis[2] - axis[2] * vec[1] +
                axis[1] * vec[2]) *
               math.sin(theta))
    temp[1] = ((origin[1] * (axis[0] * axis[0] + axis[2] * axis[2]) -
                axis[1] * (origin[0] * axis[0] + origin[2] * axis[2] - dot(axis, vec))) *
               (1.0 - math.cos(theta)) +
               vec[1] * math.cos(theta) +
               (origin[2] * axis[0] - origin[0] * axis[2] + axis[2] * vec[0] -
                axis[0] * vec[2]) *
               math.sin(theta))
    temp[2] = ((origin[2] * (axis[0] * axis[0] + axis[1] * axis[1]) -
                axis[2] * (origin[0] * axis[0] + origin[1] * axis[1] - dot(axis, vec))) *
               (1.0 - math.cos(theta)) +
               vec[2] * math.cos(theta) +
               (-origin[1] * axis[0] + origin[0] * axis[1] - axis[1] * vec[0] +
                axis[0] * vec[1]) *
               math.sin(theta))

    return temp


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


def create_turbine_segments_simple(turbine_zone_dict, u_ref, density, turbine_name_dict={}, turbine_name="", number_of_segments=12):
    # turbine_zone_dict is a Python dictionary containing the fluid zone defintion for the turbine
    # u_ref is the reference wind speed in metres / second
    # density is is kg / cubic metre

    from mpi4py import MPI
    ri = turbine_zone_dict['inner radius']
    ro = turbine_zone_dict['outer radius']
    rotor_swept_area = (math.pi * ro * ro) - (math.pi * ri * ri)

    tc = np.interp(u_ref,
                   np.array(turbine_zone_dict[
                            'thrust coefficient curve']).T[0],
                   np.array(turbine_zone_dict['thrust coefficient curve']).T[1])

    ts = np.interp(u_ref,
                   np.array(turbine_zone_dict['tip speed ratio curve']).T[0],
                   np.array(turbine_zone_dict['tip speed ratio curve']).T[1])
    # Calculate rotation speed from tip speed ratio
    omega = ts * u_ref / ro

    power = np.interp(u_ref,
                      np.array(turbine_zone_dict['turbine power curve']).T[0],
                      np.array(turbine_zone_dict['turbine power curve']).T[1])

    total_thrust = 0.5 * density * rotor_swept_area * u_ref * u_ref * tc
    total_torque = 0.0
    if omega > 0.0:
        total_torque = power / omega

    # Divide into segments
    dtheta = math.radians(360.0 / number_of_segments)
    annulus = []
    theta = 0.0
    area_check = 0.0
    thrust_check = 0.0
    torque_check = 0.0
    for i in range(number_of_segments):
        dr = 0.0
        r = ri
        while r < ro:
            dr = dtheta * r / (1.0 - 0.5 * dtheta)
            max_r = r + dr
            if max_r > ro:
                dr = ro - r
            rp = r + 0.5 * dr
            dt = total_thrust * (dtheta * rp * dr / rotor_swept_area)
            dq = total_torque * (dtheta * rp * dr / rotor_swept_area) / rp
            annulus.append((dt, dq, r, dr, i * dtheta, dtheta))
            area_check = area_check + dtheta * rp * dr
            thrust_check = thrust_check + dt
            torque_check = torque_check + dq * rp
            # print str(i) + ' ' + str(rp) + ' ' + str(dt) + ' ' + str(dq)
            r = r + dr
    # print annulus

    if MPI.COMM_WORLD.Get_rank() == 0:
        # print 'WARNING: Torque applied evenly over disc area'
        # print 'wind speed = ' + str(u_ref) + ' m/s'
        # print 'rotor swept area = ' + str(rotor_swept_area) + ' m^2'
        # print 'thrust coefficient = ' + str(tc)
        # print 'tip speed ratio = ' + str(ts)
        # print 'turbine power = ' + str(power) + ' Watts'
        # print 'density = ' + str(density) + ' kg/m^3'
        # print 'number of segments = ' + str(number_of_segments)
        # print 'rotational speed = ' + str(omega) + ' rad/s'
        # print 'total thrust = ' + str(total_thrust) + ' Newtons'
        # print 'total torque = ' + str(total_torque) + ' Joules/rad'
        turbine_name_dict[turbine_name] = power

    return annulus


def create_turbine_segments_simple_nonuniform(turbine_zone_dict, u_ref, density, number_of_segments=12):
    # turbine_zone_dict is a Python dictionary containing the fluid zone defintion for the turbine
    # u_ref is the reference wind speed in metres / second
    # density is is kg / cubic metre

    from mpi4py import MPI
    ri = turbine_zone_dict['inner radius']
    ro = turbine_zone_dict['outer radius']
    rotor_swept_area = (math.pi * ro * ro) - (math.pi * ri * ri)

    tc = np.interp(u_ref,
                   np.array(turbine_zone_dict[
                            'thrust coefficient curve']).T[0],
                   np.array(turbine_zone_dict['thrust coefficient curve']).T[1])

    ts = np.interp(u_ref,
                   np.array(turbine_zone_dict['tip speed ratio curve']).T[0],
                   np.array(turbine_zone_dict['tip speed ratio curve']).T[1])
    # Calculate rotation speed from tip speed ratio
    omega = ts * u_ref / ro

    power = np.interp(u_ref,
                      np.array(turbine_zone_dict['turbine power curve']).T[0],
                      np.array(turbine_zone_dict['turbine power curve']).T[1])

    total_thrust = 0.5 * density * rotor_swept_area * u_ref * u_ref * tc
    total_torque = 0.0
    if omega > 0.0:
        total_torque = power / omega

    if MPI.COMM_WORLD.Get_rank() == 0:
        print 'WARNING: Torque applied evenly over disc area'
        print 'wind speed = ' + str(u_ref) + ' m/s'
        print 'rotor swept area = ' + str(rotor_swept_area) + ' m^2'
        print 'thrust coefficient = ' + str(tc)
        print 'tip speed ratio = ' + str(ts)
        print 'turbine power = ' + str(power) + ' Watts'
        print 'density = ' + str(density) + ' kg/m^3'
        print 'number of segments = ' + str(number_of_segments)
        print 'rotational speed = ' + str(omega) + ' rad/s'
        print 'total thrust = ' + str(total_thrust) + ' Newtons'
        print 'total torque = ' + str(total_torque) + ' Joules/rad'

    # Divide into segments
    dtheta = math.radians(360.0 / number_of_segments)
    annulus = []
    theta = 0.0
    area_check = 0.0
    thrust_check = 0.0
    for i in range(number_of_segments):
        dr = 0.0
        r = ri
        while r < ro:
            dr = dtheta * r / (1.0 - 0.5 * dtheta)
            max_r = r + dr
            if max_r > ro:
                dr = ro - r
            rp = r + 0.5 * dr
            dt = total_thrust * (dtheta * rp * dr / rotor_swept_area)
            dq = total_torque * (dtheta * rp * dr / rotor_swept_area)
            annulus.append((dt, dq, r, dr, i * dtheta, dtheta))
            area_check = area_check + dtheta * rp * dr
            thrust_check = thrust_check + dt
            # print str(i) + ' ' + str(rp) + ' ' + str(dt) + ' ' + str(dq)
            r = r + dr
    # print annulus
    return annulus


def create_turbine_segments(thrust_coefficient, blade_inner_location,
                            blade_radius,
                            tip_speed_ratio, u_inf, density,
                            turbine_name_dict={}, turbine_name="",
                            number_of_segments=12, disc_loading_profile=[]):

    # Local import to prevent mpi init call on import
    from mpi4py import MPI

    # Assuming constant circumferential variation of loading

    radius_inner = blade_inner_location
    radius_outer = blade_radius

    # Betz Optimal rotor
    # a = 1/3

    # No wake rotation Betz limit
    # Cp = 4 * a * ( 1 - a)^2
    # Ct = 4 * a * ( 1 - a)

    # With wake rotation

    # dt
    # dq

    if thrust_coefficient > 0.999:
        print 'Induction model expects thrust coefficient < 1.0'
        print 'For greater range use simple turbine model'
        print 'Specified thrust coeeficient = ' + str(thrust_coefficient)
        print 'Resetting thrust coefficient to 0.8'
        thrust_coefficient = 0.8

    area = (math.pi * radius_outer * radius_outer) - \
        math.pi * radius_inner * radius_inner
    total_thrust = 0.5 * area * density * u_inf * u_inf * thrust_coefficient

    induction_factor = (4.0 - math.sqrt(4.0 * 4.0 - 4.0 *
                                        4.0 * thrust_coefficient)) / (2.0 * 4.0)
    # Divide into segments
    dtheta = math.radians(360.0 / number_of_segments)

    # Calculate rotation speed from tip speed ratio
    omega = tip_speed_ratio * u_inf / blade_radius

    dr = 0.0
    r = radius_inner
    theta = 0.0
    annulus = []
    area_check = 0.0
    thrust_check = 0.0
    torque_check = 0.0
    for i in range(number_of_segments):
        dr = 0.0
        r = radius_inner
        while r < blade_radius:
            # Calculate delta
            dr = dtheta * r / (1.0 - 0.5 * dtheta)
            # Check if we have exceeded radius
            max_r = r + dr
            if max_r > blade_radius:
                dr = blade_radius - r
            # Midpoint radius
            rp = r + 0.5 * dr

            da = dtheta * rp * dr

            dt = 4.0 * induction_factor * \
                (1.0 - induction_factor) * 0.5 * \
                density * u_inf * u_inf * da

            lamda_r = rp * omega / u_inf
            angular_induction_factor = 0.0
            if lamda_r > 0.0:
                angular_induction_factor = -0.5 + \
                    math.sqrt(0.25 + induction_factor *
                              (1.0 - induction_factor) / lamda_r**2)
            # Tangential force = torque / r
            dq = 4.0 * angular_induction_factor * \
                (1.0 - induction_factor) * 0.5 * density * \
                u_inf * omega * rp * rp * da / rp

            annulus.append((dt, dq, r, dr, i * dtheta, dtheta))

            area_check = area_check + da
            thrust_check = thrust_check + dt
            torque_check = torque_check + dq * rp

            r = r + dr

    if MPI.COMM_WORLD.Get_rank() == 0:
        print 'Betz Limit Induction factor: ', str(induction_factor)
        print 'wind speed = ' + str(u_inf) + ' m/s'
        print 'rotor swept area = ' + str(area) + ' m^2'
        print 'thrust coefficient = ' + str(thrust_coefficient)
        print 'tip speed ratio = ' + str(tip_speed_ratio)
        print 'turbine power = ' + str(torque_check * omega) + ' Watts'
        print 'density = ' + str(density) + ' kg/m^3'
        print 'number of segments = ' + str(number_of_segments)
        print 'rotational speed = ' + str(omega) + ' rad/s'
        print 'total thrust = ' + str(thrust_check) + ' Newtons'
        print 'total torque = ' + str(torque_check) + ' Joules/rad'
        turbine_name_dict[turbine_name] = torque_check * omega

    return annulus


def project_to_plane(pt, plane_point, plane_normal):
    from numpy import dot
    # print pt,plane_point,plane_normal
    return pt - dot(pt - plane_point, plane_normal) * plane_normal


def clockwise_angle(up_vector, pt_vector, plane_normal):
    from numpy import zeros, array, dot, linalg, cross

    v_dot = dot(up_vector, pt_vector)
    v_det = dot(plane_normal, cross(up_vector, pt_vector))

    r = math.atan2(v_det, v_dot)

    if r < 0:
        r += 2.0 * math.pi

    return r


def convolution(disc, disc_centre, disc_radius, disc_normal, disc_up,
                cell_centre_list, cell_volume_list):
    from mpi4py import MPI
    import libconvolution as cv
    from numpy import zeros, array, dot, linalg, cross, asarray, ndarray

    cell_centre_list_np = asarray(cell_centre_list)
    cell_volume_list_np = asarray(cell_volume_list)
    kernel_3d = False

    weighted_sum = np.zeros(len(disc))

    weighted_sum = cv.convolution_2dkernel_weights(disc, disc_centre, disc_radius, disc_normal, disc_up,
                                                   cell_centre_list_np, cell_volume_list_np)
    # Need to reduce weighted sum over all processes
    totals = np.zeros_like(weighted_sum)

    MPI.COMM_WORLD.Allreduce(weighted_sum, totals, op=MPI.SUM)
    weighted_sum = totals

    thrust_check_total = 0

    cell_force = np.zeros(len(cell_centre_list_np) * 3)
    thrust_check = cv.convolution_2dkernel_force(disc, disc_centre, disc_radius, disc_normal, disc_up,
                                                 cell_centre_list_np, cell_volume_list_np, weighted_sum, cell_force)

    thrust_check_array = np.array([thrust_check])
    thrust_check_total_array = np.array([0.0])

    MPI.COMM_WORLD.Allreduce(
        thrust_check_array, thrust_check_total_array, op=MPI.SUM)
    thrust_check_total = thrust_check_total_array[0]
    thrust_check = thrust_check_total

    # if MPI.COMM_WORLD.Get_rank() == 0:
    #    print 'Convolved total thrust: ',thrust_check

    #thrust_check = 0.0
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
        thrust_factor = total_thrust / thrust_check
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #    print 'Scaling thrust: ', thrust_factor
        cell_force_scaled = []
        for cell in range(len(cell_force) / 3):
            cell_force_scaled.append((cell_force[cell * 3 + 0] * thrust_factor, cell_force[
                                     cell * 3 + 1] * thrust_factor, cell_force[cell * 3 + 2] * thrust_factor))
        return cell_force_scaled
    else:
        cell_force = ndarray.tolist(cell_force)
        cell_array = iter(cell_force)
        return zip(cell_array, cell_array, cell_array)

    cell_force = ndarray.tolist(cell_force)
    cell_array = iter(cell_force)
    return zip(cell_array, cell_array, cell_array)


def convolution2(disc, disc_centre, disc_radius, disc_normal, disc_up,
                 cell_centre_list, cell_volume_list):
    from mpi4py import MPI

    from numpy import zeros, array, dot, linalg, cross
    #from zutil import R_2vect

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

        disc_pt = array([disc_r * math.cos(disc_theta),
                         disc_r * math.sin(disc_theta),
                         0.0])
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

                dt_per_area = dt / area
                dq_per_area = dq / area

                disc_pt = disc_pt_list[idx]
                # print disc_pt

                distance_vec = array(cell_centre) - disc_pt
                distance = math.sqrt(dot(distance_vec, distance_vec))
                unit_distance_vec = distance_vec / distance

                epsilon = 2.0 * dr

                # 3D kernel
                eta = 1.0 / ((epsilon**2) * math.sqrt(math.pi) **
                             3) * math.exp(-(distance / epsilon)**2)
                #/math.fabs(dot(disc_normal,unit_distance_vec))

                weighted_sum[idx] += max(1.0e-16, eta *
                                         cell_volume_list[cell_idx])

                # 1D kernel
                #eta = 1.0/(epsilon*math.sqrt(math.pi)) * math.exp(-(distance/epsilon)**2)

                # print eta,cell_centre,disc_pt

                # Set thrust force
                # for i in range(3):
                #    cell_dt[i] += dt_per_area*disc_normal[i]*eta

                # Set torque force
                # Vector for centre to pt
                #disc_vec = disc_pt - array(disc_centre)
                #unit_disc_vec = disc_vec/linalg.norm(disc_vec)
                #torque_vector = cross(unit_disc_vec,disc_normal)

                # for i in range(3):
                #    cell_dq[i] += dq_per_area*torque_vector[i]*eta
        else:
            # Need for find nearest segment

            plane_pt = project_to_plane(
                array(cell_centre), array(disc_centre), array(disc_normal))

            # print 'Plane pt: ' + str(plane_pt)

            plane_pt_radius = linalg.norm(plane_pt - disc_centre)
            plane_pt_theta = 0.0
            if plane_pt_radius > 0.0:
                #plane_pt_theta  = math.acos(dot([0.0,0.0,1.0],(plane_pt-disc_centre)/plane_pt_radius))
                plane_pt_theta = clockwise_angle(
                    disc_up, (plane_pt - disc_centre) / plane_pt_radius, array(disc_normal))

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

                dt_per_area = dt / area
                dq_per_area = dq / area

                distance_vec = array(cell_centre) - plane_pt
                distance = math.sqrt(dot(distance_vec, distance_vec))

                #epsilon = 2.0*dr
                epsilon = 0.2 * disc_radius

                # 1D kernel
                eta = 1.0 / (epsilon * math.sqrt(math.pi)) * \
                    math.exp(-(distance / epsilon)**2)

                # Add max as eta may be zero due to underflow in the exponent
                # term
                weighted_sum[
                    min_index] += max(1.0e-16, eta * cell_volume_list[cell_idx])

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

                dt_per_area = dt / area
                dq_per_area = dq / area

                disc_pt = disc_pt_list[idx]
                # print disc_pt

                distance_vec = array(cell_centre) - disc_pt
                distance = math.sqrt(dot(distance_vec, distance_vec))
                unit_distance_vec = distance_vec / distance

                epsilon = 2.0 * dr

                # 3D kernel
                eta = 1.0 / ((epsilon**2) * math.sqrt(math.pi) **
                             3) * math.exp(-(distance / epsilon)**2)
                #/math.fabs(dot(disc_normal,unit_distance_vec))

                redistribution_weight = weighted_sum[idx]

                # 1D kernel
                #eta = 1.0/(epsilon*math.sqrt(math.pi)) * math.exp(-(distance/epsilon)**2)

                # print eta,cell_centre,disc_pt

                # Set thrust force
                for i in range(3):
                    cell_dt[i] += dt_per_area * area / \
                        redistribution_weight * disc_normal[i] * eta

                # Set torque force
                # Vector for centre to pt
                disc_vec = disc_pt - array(disc_centre)
                disc_vec_mag = linalg.norm(disc_vec)
                unit_disc_vec = disc_vec / disc_vec_mag
                torque_vector = cross(disc_normal, unit_disc_vec)

                # Note converting torque to a force
                for i in range(3):
                    cell_dq[i] += dq_per_area * area / redistribution_weight * \
                        torque_vector[i] * eta / disc_vec_mag
        else:
            # Need for find nearest segment

            plane_pt = project_to_plane(
                array(cell_centre), array(disc_centre), array(disc_normal))

            # print 'Plane pt: ' + str(plane_pt)

            plane_pt_radius = linalg.norm(plane_pt - disc_centre)

            plane_pt_theta = 0.0
            if plane_pt_radius > 0.0:
                #plane_pt_theta  = math.acos(dot([0.0,0.0,1.0],(plane_pt-disc_centre)/plane_pt_radius))
                plane_pt_theta = clockwise_angle(
                    disc_up, (plane_pt - disc_centre) / plane_pt_radius, array(disc_normal))

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

                dt_per_area = dt / area
                dq_per_area = dq / area

                distance_vec = array(cell_centre) - plane_pt
                distance = math.sqrt(dot(distance_vec, distance_vec))

                #epsilon = 2.0*dr
                epsilon = 0.2 * disc_radius

                # 1D kernel
                eta = 1.0 / (epsilon * math.sqrt(math.pi)) * \
                    math.exp(-(distance / epsilon)**2)

                redistribution_weight = weighted_sum[min_index]

                # print redistribution_weight

                # print dt,eta,cell_centre,plane_pt

                # Set thrust force
                for i in range(3):
                    cell_dt[i] += dt_per_area * area / \
                        redistribution_weight * disc_normal[i] * eta

                # Set torque force
                # Vector for centre to pt
                disc_vec = plane_pt - array(disc_centre)
                disc_vec_mag = linalg.norm(disc_vec)
                unit_disc_vec = disc_vec / disc_vec_mag
                torque_vector = cross(disc_normal, unit_disc_vec)

                # Note converting torque to force
                for i in range(3):
                    cell_dq[i] += dq_per_area * area / redistribution_weight * \
                        torque_vector[i] * eta / disc_vec_mag

        cell_force.append((cell_dt[0] + cell_dq[0],
                           cell_dt[1] + cell_dq[1],
                           cell_dt[2] + cell_dq[2]))

        thrust_check += dot(cell_force[-1],
                            disc_normal) * cell_volume_list[cell_idx]

    thrust_check_total = 0

    thrust_check_array = np.array([thrust_check])
    thrust_check_total_array = np.array([0.0])
    MPI.COMM_WORLD.Allreduce(
        thrust_check_array, thrust_check_total_array, op=MPI.SUM)
    thrust_check_total = thrust_check_total_array[0]
    thrust_check = thrust_check_total

    if MPI.COMM_WORLD.Get_rank() == 0:
        print 'Convolved total thrust: ', thrust_check

    #thrust_check = 0.0
    total_thrust = 0.0
    for idx, w in enumerate(weighted_sum):
        segment = disc[idx]
        # if w > 0.0:
        #    thrust_check += segment[0]
        total_thrust += segment[0]

    if MPI.COMM_WORLD.Get_rank() == 0:
        print 'Specified total thrust: ', total_thrust

    if thrust_check > 0.0:
        thrust_factor = total_thrust / thrust_check
        if MPI.COMM_WORLD.Get_rank() == 0:
            print 'Scaling thrust: ', thrust_factor
        cell_force_scaled = []
        for cell in cell_force:
            force = cell
            cell_force_scaled.append(
                (force[0] * thrust_factor, force[1] * thrust_factor, force[2] * thrust_factor))

        return cell_force_scaled
    else:
        return cell_force

    return cell_force


def test_convolution():
    a = create_turbine_segments(0.7, 0.1, 1.0, 6.0, 1, 1)
    b = convolution(a, (0, 0, 0), (1, 0, 0), [(0.0, 0.0, 1.5)], [1.0])
    b = convolution(a, (0, 0, 0), (1, 0, 0), [(0.0, 0.0, 0.99)], [1.0])
    b = convolution(a, (0, 0, 0), (1, 0, 0), [(2.0, 0.0, 0.5)], [1.0])

    print b
