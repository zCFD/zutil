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
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from past.builtins import execfile
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
import math
import sys
from os import path
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
    if path.exists(filename):
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
        print("Unexpected error:", sys.exc_info()[0])
        return None


def get_default_zone_info():
    import inspect
    _, filename, linenumber, _, _, _ = inspect.stack()[1]
    return get_zone_info(path.split(path.splitext(filename)[0])[1] +
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

# area of polygon specified as counter-clockwise vertices (x,y) where last = first
def polygon_area(x,y):
    a = 0.0
    for i in range(len(x)-1):
        a += 0.5*(x[i+1]+x[i])*(y[i+1]-y[i])
    return a

def trapezoid(x,y):
    a = 0
    for i in range(len(x)-1):
        a += 0.5*(y[i+1]+y[i])*(x[i+1]-x[i])
    return a

# Optimal power coefficient as a function of (a function of) tip speed ratio
# "A Compact, Closed-form Solution for the Optimum, Ideal Wind Turbine" (Peters, 2012)
def glauert_peters(y):
    p1 = 16.0*(1.0-2.0*y)/(27.0*(1.0+y/4.0))
    p2 = old_div((math.log(2.0*y)+(1.0-2.0*y)+0.5*(1.0-2.0*y)**2),((1.0-2.0*y)**3))
    p3 = 1.0 + (457.0/1280.0)*y + (51.0/640.0)*y**2 + y**3/160.0 + 3.0/2.0*y*p2
    power_coeff = p1*p3
    return power_coeff

# Golden section search: Given f with a single local max in [a,b], gss returns interval [c,d] with d-c <= tol.
def gss(f,a,b,tol=1e-5):
    invphi = old_div((math.sqrt(5) - 1), 2) # 1/phi
    invphi2 = old_div((3 - math.sqrt(5)), 2) # 1/phi^2
    (a,b)=(min(a,b),max(a,b))
    h = b - a
    if h <= tol: return (a,b)
    n = int(math.ceil(old_div(math.log(old_div(tol,h)),math.log(invphi))))
    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)[0]
    yd = f(d)[0]
    for k in range(n-1):
        if yc > yd:
            b = d
            d = c
            yd = yc
            h = invphi*h
            c = a + invphi2 * h
            yc = f(c)[0]
        else:
            a = c
            c = d
            yc = yd
            h = invphi*h
            d = a + invphi * h
            yd = f(d)[0]
    if yc > yd:
        return (a,d)
    else:
        return (c,b)

def create_annulus(turbine_zone_dict):
    from mpi4py import MPI

    if 'verbose' in turbine_zone_dict:
        verbose = turbine_zone_dict['verbose']
    else:
        verbose = False

    if 'number of segments' in turbine_zone_dict:
        number_of_segments = turbine_zone_dict['number of segments']
    else:
        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            print('NO NUMBER OF SEGMENTS SPECIFIED - SETTING TO DEFAULT 12')
        number_of_segments = 12

    if 'inner radius' in turbine_zone_dict:
        ri = turbine_zone_dict['inner radius']
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('NO INNER RADIUS SPECIFIED')

    if 'outer radius' in turbine_zone_dict:
        ro = turbine_zone_dict['outer radius']
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('NO OUTER RADIUS SPECIFIED')

    disc_centre = turbine_zone_dict['centre']
    disc_normal = turbine_zone_dict['normal']

    rotor_swept_area = (math.pi * ro * ro) - (math.pi * ri * ri)

    annulus = []
    dtheta = math.radians(360.0 / number_of_segments)
    theta = 0.0
    total_area = 0.0
    for i in range(number_of_segments):
        r = ri
        while r < ro:
            dr = dtheta * r / (1.0 - 0.5 * dtheta)
            max_r = r + dr
            if max_r > ro: dr = ro - r
            rp = r + 0.5 * dr
            da = dtheta * rp * dr
            disc_theta = i * dtheta + 0.5 * dtheta

            disc_pt = array([rp * math.cos(disc_theta),
                             rp * math.sin(disc_theta),
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

            #disc_pt_list.append(disc_pt)

            annulus.append((r, dr, i * dtheta, dtheta, disc_pt[0], disc_pt[1], disc_pt[2]))
            total_area += da
            r = r + dr

    return annulus

def create_turbine_segments(turbine_zone_dict, v0, v1, v2, density, turbine_name_dict={}, turbine_name="", annulusVel=None):
    # turbine_zone_dict is a Python dictionary containing the fluid zone definition for the turbine
    # vel is the reference wind velocity in metres / second
    # density is is kg / cubic metre
    from mpi4py import MPI

    if 'verbose' in turbine_zone_dict:
        verbose = turbine_zone_dict['verbose']
    else:
        verbose = False

    if 'number of segments' in turbine_zone_dict:
        number_of_segments = turbine_zone_dict['number of segments']
    else:
        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            print('NO NUMBER OF SEGMENTS SPECIFIED - SETTING TO DEFAULT 12')
        number_of_segments = 12

    if 'inner radius' in turbine_zone_dict:
        ri = turbine_zone_dict['inner radius']
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('NO INNER RADIUS SPECIFIED')

    if 'outer radius' in turbine_zone_dict:
        ro = turbine_zone_dict['outer radius']
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('NO OUTER RADIUS SPECIFIED')

    rotor_swept_area = (math.pi * ro * ro) - (math.pi * ri * ri)
    u_ref = math.sqrt(v0*v0 + v1*v1 + v2*v2)

    induction = False
    simple = False
    bet = False
    if 'model' in turbine_zone_dict:
        if 'induction' in turbine_zone_dict['model']:
            if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                print('INDUCTION MODEL')
            induction = True
        if 'simple' in turbine_zone_dict['model']:
            if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                print('SIMPLE MODEL')
            simple = True
        if 'blade element theory' in turbine_zone_dict['model']:
            if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
                print('BLADE ELEMENT THEORY MODEL')
            bet = True
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('No MODEL SPECIFIED - DEFAULT TO SIMPLE MODEL')
        simple = True

    if 'betz' in turbine_zone_dict:
        betz = turbine_zone_dict['betz']
    else:
        betz = False

    if 'glauert' in turbine_zone_dict:
        glauert = turbine_zone_dict['glauert']
        if glauert:
            if betz:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('OVER-RIDING BETZ WITH GLAUERT POWER MODEL')
                betz = False
    else:
        glauert = False

    if 'inertia' in turbine_zone_dict:
        inertia = turbine_zone_dict['inertia']
    else:
        inertia = False

    if inertia:
        if 'omega_old' in turbine_zone_dict:
            omega_old = turbine_zone_dict['omega old']
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('INERTIA NEEDS OMEGA HISTORY')
        if 'timestep' in turbine_zone_dict:
            timestep = turbine_zone_dict['timestep']
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('INERTIA NEEDS TIME STEP')
        if 'rotor moment of inertia' in turbine_zone_dict:
            rotor_moment = turbine_zone_dict['rotor moment of inertia']
        elif 'mean blade material density' in turbine_zone_dict:
            blade_material_density = turbine_zone_dict['mean blade material density']
            if 'aerofoil section area' in turbine_zone_dict:
                aerofoil_section_area = turbine_zone_dict['aerofoil section area']
            elif 'aerofoil profile' in turbine_zone_dict:
                if 'upper surface' in turbine_zone_dict['aerofoil profile']:
                    upper = turbine_zone_dict['aerofoil profile']['upper surface']
                else:
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print('AEROFOIL PROFILE NEEDS UPPER SURFACE')
                if 'lower surface' in turbine_zone_dict['aerofoil profile']:
                    lower = turbine_zone_dict['aerofoil profile']['lower surface']
                else:
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print('AEROFOIL PROFILE NEEDS LOWER SURFACE')
                x = np.concatenate((np.array(aerofoil['aerofoil profile']['lower surface']).T[0],
                    np.array(aerofoil['aerofoil profile']['upper surface']).T[0][::-1][1:]))
                y = np.concatenate((np.array(aerofoil['aerofoil profile']['lower surface']).T[1],
                    np.array(aerofoil['aerofoil profile']['upper surface']).T[1][::-1][1:]))
                aerofoil_section_area = polygon_area(x,y)
            else:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('CALCULATE ROTOR INERTIA NEEDS AEROFOIL SECTION AREA')
            if 'number of blades' in turbine_zone_dict:
                nblades = turbine_zone_dict['number of blades']
            else:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('CALCULATE ROTOR INERTIA NEEDS BLADE DEFINITION (NBLADES)')
            if 'blade chord' in turbine_zone_dict:
                blade_chord = np.array(turbine_zone_dict['blade chord'])
            else:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('BLADE ELEMENT THEORY NEEDS BLADE DEFINITION (CHORD)')
            rotor_moment = 0.0
            for r in np.linspace(ri, ro, 100):
                dr = old_div((ro-ri),100)
                c = np.interp(old_div(r,ro),blade_chord.T[0],blade_chord.T[1])*ro
                rotor_moment += r*r*c*c*dr
            rotor_moment = rotor_moment*blade_material_density*aerofoil_section_area*nblades
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('rotor moment of inertia = ' + str(rotor_moment))
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('INERTIA MODEL NEEDS MOMENT OF INERTIA')

    if bet:
        if 'number of blades' in turbine_zone_dict:
            nblades = turbine_zone_dict['number of blades']
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('BLADE ELEMENT THEORY NEEDS BLADE DEFINITION (NBLADES)')
        if 'aerofoil cl' in turbine_zone_dict:
            aerofoil_cl = np.array(turbine_zone_dict['aerofoil cl'])
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('BLADE ELEMENT THEORY NEEDS AEROFOIL DEFINITION (CL)')
        if 'aerofoil cd' in turbine_zone_dict:
            aerofoil_cd = np.array(turbine_zone_dict['aerofoil cd'])
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('BLADE ELEMENT THEORY NEEDS AEROFOIL DEFINITION (CD)')
        if 'blade chord' in turbine_zone_dict:
            blade_chord = np.array(turbine_zone_dict['blade chord'])
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('BLADE ELEMENT THEORY NEEDS BLADE DEFINITION (CHORD)')
        if 'blade twist' in turbine_zone_dict:
            blade_twist = np.array(turbine_zone_dict['blade twist'])
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('BLADE ELEMENT THEORY NEEDS BLADE DEFINITION (TWIST)')
        if 'blade pitch range' in turbine_zone_dict:
            blade_pitch_range = turbine_zone_dict['blade pitch range']
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('BLADE PITCH AUTO NEEDS RANGE')
            blade_pitch_range = [0.0,0.0]
        if 'blade pitch tol' in turbine_zone_dict:
            blade_pitch_tol = turbine_zone_dict['blade pitch tol']
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('BLADE PITCH NEEDS TOLERANCE')
            blade_pitch_tol = 1e-3

    if 'yaw' in turbine_zone_dict:
        yaw = turbine_zone_dict['yaw']
    else:
        yaw = 0.0

    if 'tilt' in turbine_zone_dict:
        tilt = turbine_zone_dict['tilt']
    else:
        tilt = 0.0

    if 'thrust coefficient curve' in turbine_zone_dict:
        tc = np.interp(u_ref,
                       np.array(turbine_zone_dict['thrust coefficient curve']).T[0],
                       np.array(turbine_zone_dict['thrust coefficient curve']).T[1])
    elif 'thrust coefficient' in turbine_zone_dict:
        tc = turbine_zone_dict['thrust coefficient']
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('NO THRUST COEFFICIENT SPECIFIED')

    if 'tip speed ratio curve' in turbine_zone_dict:
        ts = np.interp(u_ref,
                       np.array(turbine_zone_dict['tip speed ratio curve']).T[0],
                       np.array(turbine_zone_dict['tip speed ratio curve']).T[1])
    elif 'tip speed ratio' in turbine_zone_dict:
        ts = turbine_zone_dict['tip speed ratio']
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('NO TIP SPEED RATIO SPECIFIED')

    if induction and bet:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('CANNOT USE BLADE ELEMENT THEORY WITH INDUCTION MODEL')
        induction = False

    omega = old_div(ts*u_ref,ro)
    if verbose and (MPI.COMM_WORLD.Get_rank() == 0):
        print('tip speed ratio = ' + str(ts))
        print('rotational speed = ' + str(omega) + ' rad/s')
        print('wind speed = ' + str(u_ref) + ' m/s')
        print('rotor swept area = ' + str(rotor_swept_area) + ' m^2')
        print('density = ' + str(density) + ' kg/m^3')
        print('number of segments = ' + str(number_of_segments))

    if (induction):
        u_infty = u_ref
    else:
        u_infty = (3.0/2.0)*u_ref # Assuming 1D momentum theory and the Betz limit

    betz_power = 0.5*density*u_infty**3*rotor_swept_area*(16.0/27.0)
    b_vals = np.arange(0.3334,0.5,0.0001)
    peters_lr_vals = []
    for b in b_vals:
        peters_lr_vals.append(old_div(math.sqrt(1.0+b)*(1.0-2.0*b),math.sqrt(3.0*b-1.0)))
    b0 = np.interp(ts,peters_lr_vals[::-1],b_vals[::-1])
    y = (3.0*b0-1.0)
    glauert_power = 0.5*density*u_infty**3*rotor_swept_area*glauert_peters(y)

    if betz:
        power = betz_power
    elif glauert:
        power = glauert_power
    elif 'turbine power curve' in turbine_zone_dict:
        power = np.interp(u_ref,
                          np.array(turbine_zone_dict['turbine power curve']).T[0],
                          np.array(turbine_zone_dict['turbine power curve']).T[1])
    elif 'turbine power' in turbine_zone_dict:
        power = turbine_zone_dict['turbine power']
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('NO POWER MODEL SPECIFIED - USING BETZ LIMIT')
        power = betz_power

    if bet:
        def bet_kernel(beta_pitch):
            dtheta = math.radians(360.0 / number_of_segments)
            annulus = []
            theta = 0.0
            total_area = 0.0
            total_thrust = 0.0
            total_torque = 0.0
            u_ref_local = u_ref*math.cos(math.radians(yaw))*math.cos(math.radians(tilt))
            for i in range(number_of_segments):
                r = ri
                while r < ro:
                    dr = old_div(dtheta * r, (1.0 - 0.5 * dtheta))
                    max_r = r + dr
                    if max_r > ro: dr = ro - r
                    rp = r + 0.5 * dr
                    da = dtheta * rp * dr
                    #print 'Need to have the full velocity vector at every point on the actuator disk'
                    #print 'A reference velocity would also be useful, to provide baseline Betz / Glauert scaling'
                    #print 'This is also where yaw and tilt are ideally to be included in the BET model'
                    omega_air = 0.0 # -0.1*omega (this should be calculated locally, with a value of 5% typical)
                    omega_rel = omega - omega_air
                    urel = math.sqrt((rp*omega_rel)**2 + u_ref_local**2)
                    theta_rel = math.atan(old_div(u_ref_local,(rp*omega_rel)))
                    beta_twist = np.interp(old_div(rp,ro),blade_twist.T[0],blade_twist.T[1])
                    chord = np.interp(old_div(rp,ro),blade_chord.T[0],blade_chord.T[1])*ro
                    beta = math.radians(beta_pitch + beta_twist)
                    alpha = theta_rel - beta
                    if verbose:
                        if MPI.COMM_WORLD.Get_rank() == 0:
                            if (alpha < 0.0): print('WARNING - negative angle of attack ' + str(alpha) + ' ' + str(beta))
                    cl = np.interp(math.degrees(alpha),aerofoil_cl.T[0],aerofoil_cl.T[1])
                    cd = np.interp(math.degrees(alpha),aerofoil_cd.T[0],aerofoil_cd.T[1])
                    f_L = cl*0.5*density*urel**2*chord
                    f_D = cd*0.5*density*urel**2*chord
                    F_L = old_div(nblades,(2.0*math.pi*rp))*f_L
                    F_D = old_div(nblades,(2.0*math.pi*rp))*f_D
                    dt = (F_L*math.cos(theta_rel) + F_D*math.sin(theta_rel))*da
                    dq = (F_L*math.sin(theta_rel) - F_D*math.cos(theta_rel))*da
                    annulus.append((dt, dq, r, dr, i * dtheta, dtheta))
                    total_area += da
                    total_thrust += dt
                    total_torque += dq*rp
                    r = r + dr
            if ((MPI.COMM_WORLD.Get_rank() == 0) and verbose):
                print(beta_pitch, total_torque, total_area, total_thrust)
            return total_torque, total_area, total_thrust, annulus

        blade_pitch = np.mean(gss(bet_kernel, blade_pitch_range[0], blade_pitch_range[1], blade_pitch_tol))
        total_torque, total_area, total_thrust, annulus = bet_kernel(blade_pitch)

        if MPI.COMM_WORLD.Get_rank() == 0:
            if verbose:
                print('blade pitch = ' + str(blade_pitch) + ' degrees')
                tc = old_div(total_thrust,(0.5*density*u_ref**2*rotor_swept_area))
                print('thrust coefficient [calculated] = ' + str(tc))

    else:
        annulus = []
        # Induction assumes that u_ref is u_inf.  Direct (Simple) assumes that u_ref is at disk.
        if induction:
        # Momentum theory: Ct = 4 * a * ( 1 - a), Cp = 4 * a * ( 1 - a)^2, Betz Optimal rotor: a = 1/3
            if tc > 0.999:
                print('INDUCTION MODEL TC CANNOT EXCEED 1.0: ' + str(tc))
            ind_fac = old_div((4.0-math.sqrt(4.0*4.0-4.0*4.0*tc)),(2.0*4.0))
            if verbose and (MPI.COMM_WORLD.Get_rank() == 0):
                print('Induction factor: ', str(ind_fac))
        dtheta = math.radians(360.0 / number_of_segments)
        target_torque = old_div(power,omega)
        theta = 0.0
        total_area = 0.0
        total_thrust = 0.0
        total_torque = 0.0
        for i in range(number_of_segments):
            r = ri
            while r < ro:
                dr = old_div(dtheta * r, (1.0 - 0.5 * dtheta))
                max_r = r + dr
                if max_r > ro: dr = ro - r
                rp = r + 0.5 * dr
                da = dtheta * rp * dr
                if induction:
                    dt = 0.5*density*u_ref*u_ref*da*4.0*ind_fac*(1.0-ind_fac)
                    lambda_r = old_div(rp * omega, u_ref)
                    if lambda_r > 0.0:
                        ang_ind_fac = -0.5 + math.sqrt(0.25+old_div(ind_fac*(1.0-ind_fac),lambda_r**2))
                    else:
                        ang_ind_fac = 0.0
                    dq = 4.0*ang_ind_fac*(1.0-ind_fac)*0.5*density*u_ref*omega*rp*rp*da/rp
                else:
                    dt = 0.5*density*u_ref*u_ref*da*tc
                    dq = old_div((target_torque*da),(rotor_swept_area*rp))
                annulus.append((dt, dq, r, dr, i * dtheta, dtheta))
                total_area += da
                total_thrust += dt
                total_torque += dq*rp
                r = r + dr
        specified_thrust = 0.5*rotor_swept_area*density*u_ref*u_ref*tc

        if MPI.COMM_WORLD.Get_rank() == 0:
            if verbose:
                print('thrust coefficient [specified] = ' + str(tc))
                print('thrust [specified] = ' + str(specified_thrust))
                print('model specified power = ' + str(power) + ' Watts')
                print('target torque = ' + str(target_torque) + ' Joules/rad')

    if MPI.COMM_WORLD.Get_rank() == 0:
        total_power = total_torque * omega
        turbine_name_dict[turbine_name] = total_power
        if verbose:
            print('total area = ' + str(total_area) + ' m^2')
            print('turbine power = ' + str(total_power) + ' Watts')
            print('total thrust = ' + str(total_thrust) + ' Newtons')
            print('total torque = ' + str(total_torque) + ' Joules/rad')
            print('% of Betz limit power ' + str(old_div(100.0*total_power,betz_power)) + '%')
            print('% of Glauert optimal power ' + str(old_div(100.0*total_power,glauert_power)) + '%')
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

    MPI.COMM_WORLD.Allreduce(thrust_check_array, thrust_check_total_array, op=MPI.SUM)
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
        thrust_factor = old_div(total_thrust, thrust_check)
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #    print 'Scaling thrust: ', thrust_factor
        cell_force_scaled = []
        for cell in range(old_div(len(cell_force), 3)):
            cell_force_scaled.append((cell_force[cell * 3 + 0] * thrust_factor, cell_force[cell * 3 + 1] * thrust_factor, cell_force[cell * 3 + 2] * thrust_factor))
        return cell_force_scaled
    else:
        cell_force = ndarray.tolist(cell_force)
        cell_array = iter(cell_force)
        return list(zip(cell_array, cell_array, cell_array))

    cell_force = ndarray.tolist(cell_force)
    cell_array = iter(cell_force)
    return list(zip(cell_array, cell_array, cell_array))


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

                dt_per_area = old_div(dt, area)
                dq_per_area = old_div(dq, area)

                disc_pt = disc_pt_list[idx]
                # print disc_pt

                distance_vec = array(cell_centre) - disc_pt
                distance = math.sqrt(dot(distance_vec, distance_vec))
                unit_distance_vec = old_div(distance_vec, distance)

                epsilon = 2.0 * dr

                # 3D kernel
                eta = 1.0 / ((epsilon**2) * math.sqrt(math.pi) **
                             3) * math.exp(-(old_div(distance, epsilon))**2)
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
                    disc_up, old_div((plane_pt - disc_centre), plane_pt_radius), array(disc_normal))

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

                #epsilon = 2.0*dr
                epsilon = 0.2 * disc_radius

                # 1D kernel
                eta = 1.0 / (epsilon * math.sqrt(math.pi)) * \
                    math.exp(-(old_div(distance, epsilon))**2)

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

                dt_per_area = old_div(dt, area)
                dq_per_area = old_div(dq, area)

                disc_pt = disc_pt_list[idx]
                # print disc_pt

                distance_vec = array(cell_centre) - disc_pt
                distance = math.sqrt(dot(distance_vec, distance_vec))
                unit_distance_vec = old_div(distance_vec, distance)

                epsilon = 2.0 * dr

                # 3D kernel
                eta = 1.0 / ((epsilon**2) * math.sqrt(math.pi) **
                             3) * math.exp(-(old_div(distance, epsilon))**2)
                #/math.fabs(dot(disc_normal,unit_distance_vec))

                redistribution_weight = weighted_sum[idx]

                # 1D kernel
                #eta = 1.0/(epsilon*math.sqrt(math.pi)) * math.exp(-(distance/epsilon)**2)

                # print eta,cell_centre,disc_pt

                # Set thrust force
                for i in range(3):
                    cell_dt[i] += old_div(dt_per_area * area, redistribution_weight) * disc_normal[i] * eta

                # Set torque force
                # Vector for centre to pt
                disc_vec = disc_pt - array(disc_centre)
                disc_vec_mag = linalg.norm(disc_vec)
                unit_disc_vec = old_div(disc_vec, disc_vec_mag)
                torque_vector = cross(disc_normal, unit_disc_vec)

                # Note converting torque to a force
                for i in range(3):
                    cell_dq[i] += old_div(dq_per_area * area, redistribution_weight) * \
                        old_div(torque_vector[i] * eta / disc_vec_mag)
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
                    disc_up, old_div((plane_pt - disc_centre), plane_pt_radius), array(disc_normal))

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

                #epsilon = 2.0*dr
                epsilon = 0.2 * disc_radius

                # 1D kernel
                eta = 1.0 / (epsilon * math.sqrt(math.pi)) * \
                    math.exp(-(old_div(distance, epsilon))**2)

                redistribution_weight = weighted_sum[min_index]

                # print redistribution_weight

                # print dt,eta,cell_centre,plane_pt

                # Set thrust force
                for i in range(3):
                    cell_dt[i] += old_div(dt_per_area * area, \
                        redistribution_weight) * disc_normal[i] * eta

                # Set torque force
                # Vector for centre to pt
                disc_vec = plane_pt - array(disc_centre)
                disc_vec_mag = linalg.norm(disc_vec)
                unit_disc_vec = old_div(disc_vec, disc_vec_mag)
                torque_vector = cross(disc_normal, unit_disc_vec)

                # Note converting torque to force
                for i in range(3):
                    cell_dq[i] += old_div(dq_per_area * area, redistribution_weight) * \
                        old_div(torque_vector[i] * eta / disc_vec_mag)

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
        print('Convolved total thrust: ', thrust_check)

    #thrust_check = 0.0
    total_thrust = 0.0
    for idx, w in enumerate(weighted_sum):
        segment = disc[idx]
        # if w > 0.0:
        #    thrust_check += segment[0]
        total_thrust += segment[0]

    if MPI.COMM_WORLD.Get_rank() == 0:
        print('Specified total thrust: ', total_thrust)

    if thrust_check > 0.0:
        thrust_factor = old_div(total_thrust, thrust_check)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('Scaling thrust: ', thrust_factor)
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

    print(b)
