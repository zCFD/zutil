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

from builtins import str
from builtins import range
import math
import ast
import sys
import paraview.simple as pvs
import numpy as np
from zutil import post
import zutil
import os
from zutil import ABL
import vtk
from string import Template


# Usage
# import zutil.farm as farm
# farm.create_mesh_sources('London_Array_ZCFD.py',(396000,5721000))

# farm.create_mesh_sources('Horns_Rev_ZCFD.py',(427000,6149500))
# f.create_zcfd_input('Horns_Rev_ZCFD.py',(427000,6149500))

# http://jthatch.com/Terrain2STL/


def create_mesh_sources(array_data_file, farm_centre, turbine_only=False):
    # Read file

    array_data = {}

    with open(array_data_file, "r") as f:
        s = f.read()
        array_data = ast.literal_eval(s)

    # Cases
    cases = array_data["Cases"]

    for key, value in list(cases.items()):
        # print key

        # Wind direction
        wind_direction = key

        # Turbines
        turbines = array_data["Turbines"]

        # List of tuples
        mesh_source_location = []

        for key, value in list(turbines.items()):
            # print key
            name = key
            if isinstance(key, int):
                name = "A" + str(key)

            # Location as a tuple
            turbine_location = (value["X"], value["Y"], float(value["Z"]))

            # Convert to local coordinates
            turbine_location = convert_to_local_coordinates(
                turbine_location, farm_centre
            )

            # Compute new location
            turbine_location = get_turbine_location(turbine_location, wind_direction)

            turbine_diameter = float(value["RotorDiameter"])
            # Create line source using turbine diameter
            source = create_source(turbine_location, turbine_diameter)
            mesh_source_location.append(source)

            generate_turbine(name, turbine_location, turbine_diameter, wind_direction)
            if not turbine_only:
                generate_turbine_region(
                    name, turbine_location, turbine_diameter, wind_direction
                )

        if not turbine_only:
            # Write Solar .bac file
            write_solar_bac(
                "wind-" + str(wind_direction) + ".bac",
                wind_direction,
                mesh_source_location,
            )
            # Write Solar .ctl file
            write_control_file("wind-" + str(wind_direction) + ".ctl")


def create_turbines(array_data_file, wall_file, volume_file, turbine_only=False):
    array_data = {}

    with open(array_data_file, "r") as f:
        s = f.read()
        array_data = ast.literal_eval(s)

    # Read terrain
    terrain = pvs.PVDReader(FileName=wall_file)
    terrain = pvs.CleantoGrid(Input=terrain)
    bounds = terrain.GetDataInformation().GetBounds()
    # Elevation
    elevation = pvs.Elevation(Input=terrain)
    elevation.LowPoint = [0, 0, bounds[4]]
    elevation.HighPoint = [0, 0, bounds[5]]
    elevation.ScalarRange = [bounds[4], bounds[5]]
    # Flatten
    transform = pvs.Transform(Input=elevation)
    transform.Transform = "Transform"
    transform.Transform.Scale = [1.0, 1.0, 0.0]
    transform.UpdatePipeline()

    # create a new 'Probe Location'
    probeLocation = pvs.ProbeLocation(
        Input=transform, ProbeType="Fixed Radius Point Source"
    )
    probeLocation.Tolerance = 2.22044604925031e-16

    # Read volume
    volume = pvs.PVDReader(FileName=volume_file)
    volume = pvs.CleantoGrid(Input=volume)
    volume.UpdatePipeline()
    hubProbe = pvs.ProbeLocation(Input=volume, ProbeType="Fixed Radius Point Source")
    hubProbe.Tolerance = 2.22044604925031e-16

    # Cases
    cases = array_data["Cases"]

    for key, value in list(cases.items()):
        # print key

        # Wind direction
        wind_direction = key

        # Turbines
        turbines = array_data["Turbines"]

        # List of tuples
        location = []

        for key, value in list(turbines.items()):
            # print key
            name = key
            if isinstance(key, int):
                name = "A" + str(key)

            # Location as a tuple
            turbine_location = (value["X"], value["Y"], float(value["Z"]))

            # Find terrain elevation at X and Y
            probeLocation.ProbeType.Center = [
                turbine_location[0],
                turbine_location[1],
                0.0,
            ]
            probeLocation.UpdatePipeline()

            ground = probeLocation.GetPointData().GetArray("Elevation").GetValue(0)

            turbine_location = (value["X"], value["Y"], ground + float(value["Z"]))

            turbine_diameter = float(value["RotorDiameter"])

            # Get wind direction at hub
            hubProbe.ProbeType.Center = [
                turbine_location[0],
                turbine_location[1],
                turbine_location[2],
            ]
            hubProbe.UpdatePipeline()
            (u, v) = hubProbe.GetPointData().GetArray("V").GetValue(0)

            # Thrust Coefficient
            try:
                thrust_coefficient = float(value["ThrustCoEfficient"][-1])
            except:
                thrust_coefficient = float(value["ThrustCoEfficient"])

            turbine_diameter = float(value["RotorDiameter"])

            # Point turbine into the wind
            turbine_normal = [-u, -v, 0.0]
            mag = math.sqrt(sum(x**2 for x in turbine_normal))
            turbine_normal = [-u / mag, -v / mag, 0.0]

            generate_turbine(
                name, turbine_location, turbine_diameter, wind_direction, True
            )
            if not turbine_only:
                generate_turbine_region(
                    name, turbine_location, turbine_diameter, wind_direction, True
                )

            location.append(
                (
                    name,
                    wind_direction,
                    turbine_location,
                    turbine_diameter,
                    thrust_coefficient,
                    turbine_normal,
                )
            )

        # Write zone definition
        write_zcfd_zones("wind-" + str(wind_direction) + "_zone.py", location)
    pass


def create_zcfd_input(array_data_file, farm_centre):
    # Read file

    array_data = {}

    with open(array_data_file, "r") as f:
        s = f.read()
        array_data = ast.literal_eval(s)

    # Cases
    cases = array_data["Cases"]

    for key, value in list(cases.items()):
        # print key

        # Wind direction
        wind_direction = key

        # Turbines
        turbines = array_data["Turbines"]

        # List of tuples
        location = []

        for key, value in list(turbines.items()):
            # print key
            name = key
            if isinstance(key, int):
                name = "A" + str(key)
            # Location as a tuple
            turbine_location = (value["X"], value["Y"], float(value["Z"]))

            # Convert to local coordinates
            turbine_location = convert_to_local_coordinates(
                turbine_location, farm_centre
            )

            # Compute new location
            turbine_location = get_turbine_location(turbine_location, wind_direction)

            # Thrust Coefficient
            try:
                thrust_coefficient = float(value["ThrustCoEfficient"][-1])
            except:
                thrust_coefficient = float(value["ThrustCoEfficient"])

            turbine_diameter = float(value["RotorDiameter"])

            location.append(
                (
                    name,
                    wind_direction,
                    turbine_location,
                    turbine_diameter,
                    thrust_coefficient,
                )
            )

        # Write zone definition
        write_zcfd_zones("wind-" + str(wind_direction) + "_zone.py", location)

    pass


def write_zcfd_zones(zcfd_file_name, location):
    with open(zcfd_file_name, "w") as f:
        for idx, val in enumerate(location):
            f.write("'FZ_" + str(idx + 1) + "':{\n")
            f.write("'type':'disc',\n")
            f.write("'def':'" + str(val[0]) + "-" + str(val[1]) + ".vtp',\n")
            f.write("'thrust coefficient':" + str(val[4]) + ",\n")
            f.write("'tip speed ratio':" + str(6.0) + ",\n")
            f.write(
                "'centre':["
                + str(val[2][0])
                + ","
                + str(val[2][1])
                + ","
                + str(val[2][2])
                + "],\n"
            )
            f.write("'up':[0.0,0.0,1.0],\n")
            if len(val) > 5:
                f.write(
                    "'normal':["
                    + str(val[5][0])
                    + ","
                    + str(val[5][1])
                    + ","
                    + str(val[5][2])
                    + "],\n"
                )
            else:
                f.write("'normal':[-1.0,0.0,0.0],\n")
            f.write("'inner radius':" + str(0.05 * val[3] / 2.0) + ",\n")
            f.write("'outer radius':" + str(val[3] / 2.0) + ",\n")
            f.write("'reference plane': True,\n")
            f.write(
                "'reference point':["
                + str(val[2][0])
                + ","
                + str(val[2][1])
                + ","
                + str(val[2][2])
                + "],\n"
            )
            f.write("'update frequency': 10,\n")

            f.write("},\n")
            # out_dict = {
            #     "type": "disc",
            #     "controller": {
            #         "type": "tsr",
            #         "omega":
            #     }

            # }
    pass


def generate_turbine_region(
    turbine_name,
    turbine_location,
    turbine_diameter,
    wind_direction,
    turbine_factor=2.0,
    rotate=False,
):
    # cylinder = Cylinder()
    # cylinder.Radius = 0.5 * turbine_diameter
    # cylinder.Resolution = 128
    # cylinder.Height = turbine_factor * turbine_diameter

    line = pvs.Line()
    line.Point1 = [0.0, -0.5 * turbine_factor * turbine_diameter, 0.0]
    line.Point2 = [0.0, 0.5 * turbine_factor * turbine_diameter, 0.0]
    line.Resolution = 10

    tube = pvs.Tube(Input=line)
    tube.NumberofSides = 128
    tube.Radius = 0.5 * turbine_diameter

    transform = pvs.Transform(Input=tube)
    transform.Transform = "Transform"
    transform.Transform.Rotate = [0.0, 0.0, 90.0]
    if rotate:
        transform.Transform.Rotate = [0.0, 0.0, -wind_direction]
    transform.Transform.Translate = [
        turbine_location[0],
        turbine_location[1],
        turbine_location[2],
    ]

    writer = pvs.CreateWriter(turbine_name + "-" + str(wind_direction) + ".vtp")
    writer.Input = transform
    writer.UpdatePipeline()


def generate_turbine(
    turbine_name, turbine_location, turbine_diameter, wind_direction, rotate=False
):
    disk = pvs.Disk()
    disk.InnerRadius = 0.05 * 0.5 * turbine_diameter
    disk.OuterRadius = 0.5 * turbine_diameter
    disk.CircumferentialResolution = 128
    disk.RadialResolution = 12

    transform = pvs.Transform()
    transform.Transform = "Transform"
    transform.Transform.Rotate = [0.0, 90.0, 0.0]
    if rotate:
        transform.Transform.Rotate = [0.0, 90.0, 90.0 - wind_direction]
    transform.Transform.Translate = [
        turbine_location[0],
        turbine_location[1],
        turbine_location[2],
    ]

    writer = pvs.CreateWriter(turbine_name + "-" + str(wind_direction) + "-disk.vtp")
    writer.Input = transform
    writer.UpdatePipeline()


def write_control_file(control_file_name):
    with open(control_file_name, "w") as f:
        f.write("domain:  -50000 50000 -50000 50000 0 1000.0" + "\n")
        f.write("initial: 5000.0" + "\n")
        f.write("generateCartOnly: true" + "\n")
        f.write("generateLayer: false" + "\n")


def create_source(turbine_location, diameter):
    upstream_factor = 1.0
    downstream_factor = 4.0
    radial_factor = 1.0
    diameter_mesh_pts = 50.0
    radius_factor = 2.0

    # Upstream
    pt_1 = turbine_location[0] - upstream_factor * diameter
    # Downstream
    pt_2 = turbine_location[0] + downstream_factor * diameter
    # Radius
    radius = 0.5 * diameter * radial_factor
    # Mesh size
    mesh_size = diameter / diameter_mesh_pts

    return (
        (
            pt_1,
            turbine_location[1],
            turbine_location[2],
            mesh_size,
            radius,
            radius * radius_factor,
        ),
        (
            pt_2,
            turbine_location[1],
            turbine_location[2],
            mesh_size,
            radius,
            radius * radius_factor,
        ),
    )


def write_solar_bac(bac_file_name, wind_direction, mesh_source):
    farfield_mesh_size = 5000.0

    with open(bac_file_name, "w") as f:
        f.write("zCFD Farmer - wind direction: " + str(wind_direction) + "\n")
        f.write(" 8 6" + "\n")
        f.write("  1  1.0e+6 -1.0e+6 -1.0e+6" + "\n")
        f.write("  1.0 0.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 1.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 0.0 1.0 " + str(farfield_mesh_size) + "\n")
        f.write("  2  1.0e+6  1.0e+6 -1.0e+6" + "\n")
        f.write("  1.0 0.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 1.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 0.0 1.0 " + str(farfield_mesh_size) + "\n")
        f.write("  3 -1.0e+6 -1.0e+6 -1.0e+6" + "\n")
        f.write("  1.0 0.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 1.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 0.0 1.0 " + str(farfield_mesh_size) + "\n")
        f.write("  4 -1.0e+6  1.0e+6 -1.0e+6" + "\n")
        f.write("  1.0 0.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 1.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 0.0 1.0 " + str(farfield_mesh_size) + "\n")
        f.write("  5  1.0e+6 -1.0e+6  1.0e+6" + "\n")
        f.write("  1.0 0.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 1.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 0.0 1.0 " + str(farfield_mesh_size) + "\n")
        f.write("  6  1.0e+6  1.0e+6  1.0e+6" + "\n")
        f.write("  1.0 0.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 1.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 0.0 1.0 " + str(farfield_mesh_size) + "\n")
        f.write("  7 -1.0e+6 -1.0e+6  1.0e+6" + "\n")
        f.write("  1.0 0.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 1.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 0.0 1.0 " + str(farfield_mesh_size) + "\n")
        f.write("  8 -1.0e+6  1.0e+6  1.0e+6" + "\n")
        f.write("  1.0 0.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 1.0 0.0 " + str(farfield_mesh_size) + "\n")
        f.write("  0.0 0.0 1.0 " + str(farfield_mesh_size) + "\n")
        f.write("  1  1  2  4  8" + "\n")
        f.write("  2  1  2  8  6" + "\n")
        f.write("  3  1  6  8  5" + "\n")
        f.write("  4  2  3  4  7" + "\n")
        f.write("  5  2  7  4  8" + "\n")
        f.write("  6  2  7  8  6" + "\n")
        f.write(" background sources..." + "\n")
        f.write(" 0 " + str(len(mesh_source)) + " 0" + "\n")
        f.write(" The points" + "\n")
        f.write(" The lines" + "\n")
        for s in mesh_source:
            pt_1 = s[0]
            pt_2 = s[1]
            f.write("Line Source :" + "\n")
            f.write(" ".join(str(elem) for elem in pt_1) + "\n")
            f.write(" ".join(str(elem) for elem in pt_2) + "\n")
        f.write(" The triangles" + "\n")


def convert_to_local_coordinates(turbine_location, farm_centre):
    # Converts location from Eastings and Northings into local coordinates
    #

    local_coordinates = (
        turbine_location[0] - farm_centre[0],
        turbine_location[1] - farm_centre[1],
        turbine_location[2],
    )
    return local_coordinates


def get_turbine_location(current_location, wind_direction):
    # Assume that the CFD solver expects freestream flow in x-direction
    # Rotate farm about its centre to account for the actual wind direction
    # i.e. if the wind is coming from the West (270 degrees), then there is no rotation.
    # In general: rotate wind farm in standard co-ordinate geometry sense
    # about centre by (wind direction + 90 degrees)
    rotation_angle = wind_direction + 90.0
    rotation_angle = math.radians(rotation_angle)

    x_temp = current_location[0]
    y_temp = current_location[1]
    new_x_temp = x_temp * math.cos(rotation_angle) + y_temp * math.sin(rotation_angle)
    new_y_temp = -x_temp * math.sin(rotation_angle) + y_temp * math.cos(rotation_angle)

    return (new_x_temp, new_y_temp, current_location[2])


def report_data_reader(name, arr1, arr2):
    var = -999.0
    idx = np.where(arr1 == name)
    if len(idx[0].flat) == 0:
        print(
            "farm.py : report_data_reader : no data found in results file for " + name
        )
    elif len(idx[0].flat) == 1:
        var = float(arr2[idx[0]][0])
    else:
        print(
            "farm.py : report_data_reader : multiple columns in data results file for "
            + name
            + " "
            + len(idx[0].flat)
        )
    return var


min_dist = 1.0e16
closest_point = [min_dist, min_dist, min_dist]


def closest_point_func(dataset, pointset, s=[0, 0, 0], **kwargs):
    global min_dist, closest_point
    points = pointset.GetPoints()
    for p in points:
        dx = s[0] - p[0]
        dy = s[1] - p[1]
        dist = math.sqrt(dx * dx + dy * dy)  # + dz*dz)
        if dist < min_dist:
            closest_point = p
            min_dist = dist


def write_windfarmer_data(case_name, num_processes, up):
    # case_name = 'windfarm' (for example - note that there is no .py suffix)
    # ground_zone = 1 (for example, an integer)
    # num_processes = 24 (for example, an integer)
    # up = [0,1,0] (the vector pointing vertically upwards)
    global min_dist, closest_point
    # Step 1: Read the case file parameters
    __import__(case_name)
    case_data = getattr(sys.modules[case_name], "parameters")

    # Step 2: Determine the (conventional) wind direction from the case inflow
    # parameters
    v = case_data["IC_1"]["V"]["vector"]
    print("farm.py : write_windfarmer_data : v = " + str(v))
    import numpy as np

    angle = 270.0 - np.angle(complex(v[0], v[1]), deg=True)
    if angle < 0.0:
        angle += 360.0
    if angle > 360.0:
        angle -= 360.0
    print("farm.py : write_windfarmer_data : angle = " + str(angle))

    # Step 3: Import the result file data incuding the probe data
    windfarmer_filename = case_name + "_" + str(angle) + ".out"
    print(
        "farm.py : write_windfarmer_data : windfarmer_filename = " + windfarmer_filename
    )
    report_file_name = case_name + "_report.csv"
    report_array = np.genfromtxt(report_file_name, dtype=None)

    # Step 4: Calculate the ground heights at the probe locations by
    # subtracting the local height of the wall
    reader = pvs.OpenDataFile(
        "./"
        + case_name
        + "_P"
        + str(num_processes)
        + "_OUTPUT/"
        + case_name
        + "_wall.pvd"
    )
    local_surface = pvs.servermanager.Fetch(reader)

    # Step 5: Loop over the probe locations plus the results to create the
    # Windfarmer file.
    with open(windfarmer_filename, "w") as f:
        f.write(
            '"Point","X[m]","Y [m]","Z ground [m]","Z hub [m]","H [m]","D [m]","Theta[Deg]","TI hub","TI upper",'
            + '"TI lower","TI15 hub","TI15 upper","TI15 lower","Vxy hub [m/s]","Vxy upper [m/s]","Vxy lower [m/s]","Windshear [-]",'
            + '"Theta left [Deg]","Theta hub [Deg]","Theta right [Deg]","Veer [Deg]","Local Elevation Angle [Deg]","Simple Power [kW]",'
            + '"V/VT01_sample_vxy [-]","Power (Sector) [kW]","AEP (Sector) [kWh]","NEC [kWh]" \n'
        )
        for probe in case_data["report"]["monitor"]:
            point = case_data["report"]["monitor"][probe]["point"]
            name = case_data["report"]["monitor"][probe]["name"]

            # Step 5.1: Find the report data for the windfarmer probe if it
            # exists.
            V_x = report_data_reader(
                name + "_V_x", report_array[0], report_array[len(report_array) - 1]
            )
            V_y = report_data_reader(
                name + "_V_y", report_array[0], report_array[len(report_array) - 1]
            )
            V_z = report_data_reader(
                name + "_V_z", report_array[0], report_array[len(report_array) - 1]
            )
            TI_hub = report_data_reader(
                name + "_ti", report_array[0], report_array[len(report_array) - 1]
            )
            VXY_hub = math.sqrt(V_x * V_x + V_y * V_y)
            Theta_hub = 270.0 - np.angle(complex(V_x, V_y), deg=True)
            if Theta_hub < 0.0:
                Theta_hub += 360.0
            if Theta_hub > 360.0:
                Theta_hub -= 360.0
            Local_Elevation_Angle = np.angle(complex(VXY_hub, V_z), deg=True)

            # Step 5.2: Swap the axes if necessary to match the Windfarmer
            # default (z-axis is up)
            if up[0] == 0:
                x = point[0]
                if up[1] == 0:
                    y = point[1]
                    z = point[2]
                elif up[2] == 0:
                    y = point[2]
                    z = point[1]
            else:
                x = point[1]
                y = point[2]
                z = point[0]

            # Step 5.3: Find the closest ground point to the probe to work out
            # elevation
            min_dist = 1.0e16
            closest_point = [min_dist, min_dist, min_dist]
            post.for_each(local_surface, closest_point_func, s=[x, y, z])
            zground = (
                up[0] * closest_point[0]
                + up[1] * closest_point[1]
                + up[2] * closest_point[2]
            )
            zhub = up[0] * x + up[1] * y + up[2] * z

            # Step 5.4: Output the Windfarmer data
            f.write(
                name
                + ","
                + str(x)
                + ","
                + str(y)
                + ","
                + str(zground)
                + ","
                + str(zhub)
                + ","
                + str(zhub - zground)
                + ","
                + ","
                + str(angle)
                + ","
                + str(TI_hub)
                + ",,,,,,"
                + str(VXY_hub)
                + ",,,,,"
                + str(Theta_hub)
                + ",,,"
                + str(Local_Elevation_Angle)
                + ",,,,, \n"
            )
        print("farm.py : write_windfarmer_data : DONE")


def create_trbx_zcfd_input(
    case_name="windfarm",
    wind_direction=0.0,
    reference_wind_speed=10.0,
    terrain_file=None,  # any file for ParaView reader (STL, PVD, PVTU, etc)
    report_frequency=200,
    update_frequency=50,
    reference_point_offset=1.0,
    turbine_zone_length_factor=1.0,
    model="simple",  # options are (induction, simple, blade element theory)
    turbine_files=[
        ["xyz_location_file1.txt", "turbine_type1.trbx"],
        ["xyz_location_file2.txt", "turbine_type2.trbx"],
    ],
    calibration_offset=0.0,
    **kwargs,
):
    # Ensure turbine folder exists
    directory = "./turbine_vtp/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Make sure that the turbine zone contains the reference point
    if turbine_zone_length_factor < 2.5 * reference_point_offset:
        print(
            "WARNING: Increasing Turbine Zone Length Factor from "
            + str(turbine_zone_length_factor)
            + " to "
            + str(2.5 * reference_point_offset)
        )
        turbine_zone_length_factor = 2.5 * reference_point_offset
    # Issue a warning if the turbine zone length factor is less than 1.0
    if turbine_zone_length_factor < 1.0:
        print(
            "WARNING: Turbine Zone Length Factor less than 1.0: "
            + str(turbine_zone_length_factor)
        )
    global min_dist, closest_point
    from xml.etree import ElementTree as ET

    local_surface = None
    if terrain_file is not None:
        reader = pvs.OpenDataFile(terrain_file)
        local_surface = pvs.servermanager.Fetch(reader)
        print("terrain file = " + terrain_file)
        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(local_surface)
        pointLocator.BuildLocator()

    # Step 1: Read in the location data (.txt) and turbine information (.trbx)
    # for each turbine type
    idx = 0

    tz = {}
    tp = {}
    for turbine_type in turbine_files:
        location_file_name = turbine_type[0]
        if model in ("simple", "induction"):
            trbx_file_name = turbine_type[1]
            print("trbx file name = " + trbx_file_name)
            trbx = ET.ElementTree(file=trbx_file_name)
            root = trbx.getroot()
            turbine_dict = {}
            for elem in root:
                turbine_dict[elem.tag] = elem.text
            for elem in trbx.find("Turbine3DModel"):
                turbine_dict[elem.tag] = elem.text
            for elem in trbx.find(
                "PerformanceTableList/PerformanceTable/PowerCurveInfo"
            ):
                turbine_dict[elem.tag] = elem.text
            for elem in trbx.find(
                "PerformanceTableList/PerformanceTable/PowerCurveInfo/StartStopStrategy"
            ):
                turbine_dict[elem.tag] = elem.text
            turbine_dict["DataTable"] = {}
            wp = 0
            for elem in trbx.find("PerformanceTableList/PerformanceTable/DataTable"):
                turbine_dict["DataTable"][wp] = {}
                for child in elem:
                    turbine_dict["DataTable"][wp][child.tag] = child.text
                wp += 1
        elif model in ("blade element theory"):
            turbine_dict = turbine_type[1]
        else:
            print("Model not identified (simple, induction, blade element theory)")

        print("location file name = " + location_file_name)
        location_array = np.genfromtxt(
            location_file_name, delimiter=" ", dtype=("<U100", float, float)
        )
        print(location_array)
        # catch the case where only one turbine location is specified
        if location_array.ndim < 1:
            location_array = np.reshape(location_array, (1,))
        for location in location_array:
            idx += 1
            name = str(location[0])
            if len(name) > 99:
                print("WARNING: farm.py: turbine name may be truncated " + str(name))
            easting = location[1]
            northing = location[2]

            # Step 2: Work out the local elevation
            if model in ("simple", "induction"):
                hub_height = turbine_dict["SelectedHeight"]
                rd = float(turbine_dict["RotorDiameter"])
            elif model in ("blade element theory"):
                hub_height = turbine_dict["hub height"]
                rd = float(turbine_dict["outer radius"]) * 2.0
            else:
                print("Model not identified (simple, induction, blade element theory)")

            min_dist = 1.0e16
            closest_point = [min_dist, min_dist, min_dist]
            if local_surface is not None:
                pid = pointLocator.FindClosestPoint([easting, northing, 0.0])
                closest_point = local_surface.GetPoint(pid)
                height = closest_point[2]
                hub_z = height + float(hub_height)
            else:
                hub_z = float(hub_height)

            # Step 3: Generate the turbine region files
            # (./turbine_vtp/*.vtp)
            generate_turbine_region(
                directory + name,
                [easting, northing, hub_z],
                float(rd),
                wind_direction,
                turbine_zone_length_factor,
                True,
            )
            generate_turbine(
                directory + name,
                [easting, northing, hub_z],
                float(rd),
                wind_direction,
                True,
            )

            # Step 4: Generate the turbine zone definition
            # (./turbine_zone.py)
            wv = zutil.vector_from_wind_dir(wind_direction)
            if model in ("simple", "induction"):
                zID = "FZ_{}".format(idx)

                pref = [
                    easting - reference_point_offset * rd * wv[0],
                    northing - reference_point_offset * rd * wv[1],
                    hub_z - reference_point_offset * rd * wv[2],
                ]

                tz[zID] = {
                    "type": "disc",
                    "name": name,
                    "def": directory + name + "-" + str(wind_direction) + ".vtp",
                    "reference plane": kwargs.get("reference_plane", True),
                    "reference point": pref,
                    "update frequency": update_frequency,
                    "verbose": turbine_dict.get("verbose", False),
                }

                tz[zID]["discretisation"] = {
                    "type": "disc",
                    "number of elements": kwargs.get("number_of_segments", 12),
                }

                tz[zID]["geometry"] = {
                    "centre": [easting, northing, hub_z],
                    "normal": [-wv[0], -wv[1], -wv[2]],
                    "up": [0.0, 0.0, 1.0],
                    "inner radius": float(turbine_dict["DiskDiameter"]) / 2.0,
                    "outer radius": float(turbine_dict["RotorDiameter"]) / 2.0,
                }

                if model in ("simple", "induction"):
                    if len(list(turbine_dict["DataTable"].keys())) == 0:
                        print(
                            "WARNING: Windspeed DataTable empty - using Reference Wind Speed = "
                            + str(reference_wind_speed)
                        )
                    wsc = np.zeros((4, len(list(turbine_dict["DataTable"].keys()))))
                    tcc = []  # Thrust coefficient curve
                    tsc = []  # Tip speed ratio curve
                    tpc = []  # Turbine Power Curve
                    for wp in list(turbine_dict["DataTable"].keys()):
                        # Allow velocities to be shifted by user specified calibration
                        wsc[0][wp] = (
                            float(turbine_dict["DataTable"][wp]["WindSpeed"])
                            - calibration_offset
                        )
                        wsc[1][wp] = turbine_dict["DataTable"][wp]["ThrustCoEfficient"]
                        wsc[2][wp] = turbine_dict["DataTable"][wp]["RotorSpeed"]
                        wsc[3][wp] = turbine_dict["DataTable"][wp]["PowerOutput"]
                        tcc.append([wsc[0][wp], wsc[1][wp]])
                        tsc.append(
                            [
                                wsc[0][wp],
                                (wsc[2][wp] * np.pi / 30 * rd / 2.0)
                                / max(wsc[0][wp], 1.0),
                            ]
                        )
                        tpc.append([wsc[0][wp], wsc[3][wp]])
                    # print wsc
                    # If there is a single value for thrust coefficient use the
                    # reference wind speed
                    tc = np.interp(reference_wind_speed, wsc[0], wsc[1])
                    rs = np.interp(reference_wind_speed, wsc[0], wsc[2])
                    tsr = ((rs * math.pi / 30.0) * rd / 2.0) / reference_wind_speed
                    tpow = np.interp(reference_wind_speed, wsc[0], wsc[3])

                    tz[zID]["controller"] = {
                        "type": "tsr curve",
                        "tip speed ratio": tsr,
                        "tip speed ratio curve": tsc,
                    }

                    tz[zID]["model"] = {
                        "thrust coefficient": tc,
                        "thrust coefficient curve": tcc,
                        "turbine power": tpow,
                        "power curve": tpc,
                        "type": model,
                        "power model": "curve",
                    }

            elif model in ("blade element theory"):
                tz[zID]["status"] = "on"
                tz[zID]["blade material density"] = turbine_dict[
                    "blade material density"
                ]
                tz[zID]["auto yaw"] = True
                tz[zID]["tip speed limit"] = turbine_dict["tip speed limit"]
                tz[zID]["rpm ramp"] = turbine_dict["rpm ramp"]
                tz[zID]["rated power"] = turbine_dict["rated power"]
                tz[zID]["dt"] = turbine_dict["dt"]
                tz[zID]["inertia"] = True
                tz[zID]["damage ti"] = turbine_dict["damage ti"]
                tz[zID]["damage speed"] = turbine_dict["damage speed"]
                tz[zID]["friction loss"] = turbine_dict["friction loss"]
                tz[zID]["cut in speed"] = turbine_dict["cut in speed"]
                tz[zID]["cut out speed"] = turbine_dict["cut out speed"]
                tz[zID]["rotation direction"] = "clockwise"

                if "thrust factor" in turbine_dict:
                    tz[zID]["thrust factor"] = turbine_dict["thrust factor"]

                tz[zID]["model"] = {
                    "number of blades": turbine_dict["number of blades"],
                    "type": "BET",
                    "tilt": turbine_dict["tilt"],
                    "yaw": turbine_dict["yaw"],
                }

                if "tip loss correction" in turbine_dict:
                    tz[zID]["model"]["tip loss correction"] = turbine_dict[
                        "tip loss correction"
                    ]

                # controller bits

                tz[zID]["controller"] = {
                    "type": "fixed",
                    "pitch": 0.0,
                    "omega": 0.0,
                    "blade pitch tol": turbine_dict["blade pitch tol"],
                    "blade pitch range": turbine_dict["blade pitch range"],
                }

                if "blade pitch step" in turbine_dict:
                    tz[zID]["controller"]["blade pitch step"] = turbine_dict[
                        "blade pitch step"
                    ]

                if "blade pitch" in turbine_dict:
                    tz[zID]["pitch"] = turbine_dict["blade pitch"]

                tz[zID]["model"]["aerofoils"] = (
                    {
                        "aerofoil1": {
                            "cl": turbine_dict["aerofoil cl"],
                            "cd": turbine_dict["aerofoil cd"],
                        },
                    },
                )

                tz[zID]["model"]["aerofoil positions"] = [
                    [0.0, "aerofoil1"],
                    [1.0, "aerofoil1"],
                ]
                tz[zID]["geometry"]["blade chord"] = turbine_dict["blade chord"]
                tz[zID]["geometry"]["blade twist"] = turbine_dict["blade twist"]

            else:
                pass

            # Step 5: Generate the turbine monitor probes (./turbine_probe.py)
            # Turbines:    label@MHH@## (## = hub height of the turbine relative to the ground in meters)
            # Anemometers: label@AN@##  (## = height of the anemometer
            # above the ground in meters)
            pID = "MR_{}".format(idx)
            tp[pID] = {
                "variables": ["V", "ti"],
                "name": "probe{}@MHH@{}".format(idx, hub_height),
                "point": [easting, northing, hub_z],
            }

    # write tz
    with open(case_name + "_zones.py", "w") as zone_file:
        zone_file.write("turb_zone = {}".format(str(tz)))

    with open(case_name + "_probes.py", "w") as probe_file:
        probe_file.write("turb_probe = {}".format(str(tp)))
    # write tp


def extract_probe_data(
    case_name="windfarm",
    wind_direction_start=0,
    wind_direction_end=360,
    wind_direction_step=10,
    num_processes=16,
    probe_location_file="name_x_y_z.txt",
    offset=0.0,
    **kwargs,
):
    import vtk
    from vtk.util import numpy_support as VN

    probe_location_array = np.genfromtxt(probe_location_file, dtype=None)
    probe = vtk.vtkProbeFilter()
    point = vtk.vtkPointSource()
    for wd in range(wind_direction_start, wind_direction_end, wind_direction_step):
        directory = (
            case_name + "_" + str(int(wd)) + "_P" + str(num_processes) + "_OUTPUT"
        )
        filename = case_name + "_" + str(int(wd)) + ".pvd"
        reader = pvs.OpenDataFile("./" + directory + "/" + filename)
        local_volume = pvs.servermanager.Fetch(reader)
        for location in probe_location_array:
            name = location[0]
            easting = location[1]
            northing = location[2]
            height = location[3] + offset
            point.SetNumberOfPoints(1)
            point.SetCenter([easting, northing, height])
            probe.SetInputConnection(point.GetOutputPort())
            probe.SetSourceData(local_volume)
            probe.Update()
            V = VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray("V"))
            ti = VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray("ti"))
            print(
                str(wd)
                + " "
                + name
                + "_zoffset_"
                + str(offset)
                + "_V_x "
                + str(V[0][0])
            )
            print(
                str(wd)
                + " "
                + name
                + "_zoffset_"
                + str(offset)
                + "_V_y "
                + str(V[0][1])
            )
            print(
                str(wd)
                + " "
                + name
                + "_zoffset_"
                + str(offset)
                + "_V_z "
                + str(V[0][2])
            )
            print(
                str(wd)
                + " "
                + name
                + "_zoffset_"
                + str(offset)
                + "_ti "
                + str(ti[0] + 0.1)
            )


def generate_mesh_pts():
    start = 0.001
    max_height = 20000
    growth_rate = 1.01

    pts = []
    # 1.3 growth rate to layer height
    current_height = start
    current_pos = start
    pts.append(current_pos)
    while current_pos < max_height:
        current_height = min(100.0, current_height * growth_rate)
        current_pos = current_pos + current_height
        pts.append(current_pos)
        # print current_height

    return np.array(pts)


def create_profile(
    profile_name,
    hub_height,
    hub_height_vel,
    direction,
    roughness,
    min_ti,
    min_mut,
    scale_k=False,
    plot=False,
    kappa=0.41,
    rho=1.225,
    cmu=0.03,
    mu=1.789e-5,
    latitude=55.0,
):
    # Using RH Law compute utau using hub values
    utau = ABL.friction_velocity(hub_height_vel, hub_height, roughness, kappa)

    print("Friction Velocity: " + str(utau))

    # Ref http://orbit.dtu.dk/files/3737714/ris-r-1688.pdf
    coriolis_parameter = ABL.coriolis_parameter(latitude)
    geostrophic_plane = ABL.ekman_layer_height(utau, coriolis_parameter)

    print("Ekman Layer top: " + str(geostrophic_plane))
    print("This is top of ABL for neutral conditions")
    print("Wall Stress: " + str(rho * utau**2))

    pts = generate_mesh_pts()

    vel = ABL.wind_speed_array(pts, utau, roughness, kappa)

    if scale_k:
        k_scale = (
            np.ones(len(pts)) - np.minimum(np.ones(len(pts)), (pts / geostrophic_plane))
        ) ** 2
    else:
        k_scale = np.ones(len(pts))

    k = k_scale * (utau**2) / math.sqrt(cmu)
    eps = np.ones(len(pts)) * (utau**3) / (kappa * (pts + roughness))
    # Note this mut/mu
    mut = np.maximum(rho * cmu * k**2 / (eps * mu), np.ones(len(pts)) * min_mut)
    TI = np.maximum((2 * k / 3) ** 0.5 / vel, np.ones(len(pts)) * min_ti)

    points = vtk.vtkPoints()
    for x in pts:
        points.InsertNextPoint([0.0, 0.0, x])

    vel_vec = vtk.vtkFloatArray()
    vel_vec.SetNumberOfComponents(3)
    vel_vec.SetName("Velocity")
    for v in vel:
        vel_vec.InsertNextTuple(zutil.vector_from_wind_dir(direction, v))

    ti_vec = vtk.vtkFloatArray()
    ti_vec.SetNumberOfComponents(1)
    ti_vec.SetName("TI")
    for t in TI:
        ti_vec.InsertNextTuple([t])

    mut_vec = vtk.vtkFloatArray()
    mut_vec.SetNumberOfComponents(1)
    mut_vec.SetName("EddyViscosity")
    for m in mut:
        mut_vec.InsertNextTuple([m])

    # Create poly data
    linesPolyData = vtk.vtkPolyData()
    linesPolyData.SetPoints(points)
    linesPolyData.GetPointData().AddArray(vel_vec)
    linesPolyData.GetPointData().AddArray(ti_vec)
    linesPolyData.GetPointData().AddArray(mut_vec)

    # Write
    print("Writing: " + profile_name + ".vtp")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(profile_name + ".vtp")
    writer.SetInputData(linesPolyData)
    writer.Write()

    # This is always going to be avoided

    # if plot:
    #     fig = get_figure(plt)
    #     ax = fig.add_subplot(111)
    #     ax.grid(True)
    #     x_label(ax, "Velocity")
    #     y_label(ax, "Height")
    #     set_ticks(ax)
    #     ax.semilogy(vel, pts)
    #     if scale_k:
    #         ax.set_ylim(None, geostrophic_plane)

    #     fig = get_figure(plt)
    #     ax = fig.add_subplot(111)
    #     ax.grid(True)
    #     x_label(ax, "TI")
    #     y_label(ax, "Height")
    #     set_ticks(ax)
    #     ax.semilogy(TI, pts)
    #     if scale_k:
    #         ax.set_ylim(None, geostrophic_plane)

    #     fig = get_figure(plt)
    #     ax = fig.add_subplot(111)
    #     ax.grid(True)
    #     x_label(ax, "Length scale")
    #     y_label(ax, "Height")
    #     set_ticks(ax)
    #     if scale_k:
    #         ax.set_ylim(0.0, geostrophic_plane)
    #     ax.plot(lengthscale, pts)

    #     fig = get_figure(plt)
    #     ax = fig.add_subplot(111)
    #     ax.grid(True)
    #     x_label(ax, "mut/mu")
    #     y_label(ax, "Height")
    #     set_ticks(ax)
    #     if scale_k:
    #         ax.set_ylim(0.0, geostrophic_plane)
    #     ax.plot(mut, pts)

    #     fig = get_figure(plt)
    #     ax = fig.add_subplot(111)
    #     ax.grid(True)
    #     x_label(ax, "stress")
    #     y_label(ax, "Height")
    #     set_ticks(ax)
    #     if scale_k:
    #         ax.set_ylim(0.0, geostrophic_plane)


def get_case_name(base_case, wind_direction, wind_speed):
    wind_direction_str = "{0:.2f}".format(wind_direction).replace(".", "p")
    wind_speed_str = "{0:.2f}".format(wind_speed).replace(".", "p")
    return base_case + "_" + wind_direction_str + "_" + wind_speed_str


def get_profile_name(base_case, wind_direction, wind_speed):
    wind_direction_str = "{0:.2f}".format(wind_direction).replace(".", "p")
    wind_speed_str = "{0:.2f}".format(wind_speed).replace(".", "p")
    return "profile_" + wind_direction_str + "_" + wind_speed_str


def generate_inputs(
    base_case,
    wind_direction,
    wind_speed,
    wind_height,
    roughness_length,
    turbine_info,
    terrain_file,
    min_ti,
    min_mut,
    scale_k=True,
):
    # Generate new name for this case
    case_name = get_case_name(base_case, wind_direction, wind_speed)
    profile_name = get_profile_name(base_case, wind_direction, wind_speed)

    # Generate turbines
    create_trbx_zcfd_input(
        case_name=case_name,
        wind_direction=wind_direction,
        reference_wind_speed=wind_speed,
        terrain_file=terrain_file,
        report_frequency=10,
        update_frequency=1,
        reference_point_offset=0.0,
        turbine_zone_length_factor=0.2,
        model="simple",
        turbine_files=turbine_info,
        calibration_offset=0.0,
    )

    # Generate profile
    create_profile(
        profile_name,
        wind_height,
        wind_speed,
        wind_direction,
        roughness_length,
        min_ti,
        min_mut,
        scale_k,
    )
    case_file = """
import zutil
base_case = '$basecasename'
case = '$casename'

parameters = zutil.get_parameters_from_file(base_case)

turbine_zones = zutil.get_zone_info(case+'_zones')

for key,value in turbine_zones.turb_zone.items():
    valid_key = zutil.find_next_zone(parameters,'FZ')
    parameters[valid_key]=value

turbine_probes = zutil.get_zone_info(case+'_probes')
for key,value in turbine_probes.turb_probe.items():
    if not 'report' in parameters:
        parameters['report'] = {}
    if not 'monitor' in parameters['report']:
        parameters['report']['monitor'] = {}
    valid_key = zutil.find_next_zone(parameters['report']['monitor'],'MR')
    parameters['report']['monitor'][valid_key]=value

# Set reference speed
parameters['IC_1']['V']['vector'] = zutil.vector_from_wind_dir($wind_direction,$wind_speed)

# Set profile
if not 'profile' in parameters['IC_1']:
    parameters['IC_1']['profile'] = {}
parameters['IC_1']['profile']['field'] = '$profile_name.vtp'
"""

    case_file_str = Template(case_file).substitute(
        basecasename=base_case,
        casename=case_name,
        wind_direction=wind_direction,
        wind_speed=wind_speed,
        profile_name=profile_name,
    )
    print("Writing: " + case_name + ".py")
    with open(case_name + ".py", "w") as f:
        f.write(case_file_str)
