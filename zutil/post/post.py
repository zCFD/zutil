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


Helper functions for accessing zCFD Paraview functionality
"""

from tqdm import tqdm
from zutil import analysis
import json

import paraview.simple as pvs
from builtins import object
from builtins import range
from builtins import str
from future import standard_library
from typing import Union, Tuple, Optional
import os
from zutil.fileutils import clean_name
from zutil.fileutils import get_csv_data
import pandas as pd
from pathlib import Path
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

standard_library.install_aliases()
# from paraview.vtk.util import numpy_support
try:
    from paraview.vtk.dataset_adapter import Table
    from paraview.vtk.dataset_adapter import DataSet
    from paraview.vtk.dataset_adapter import PointSet
except ImportError:
    from paraview.vtk.numpy_interface.dataset_adapter import Table
    from paraview.vtk.numpy_interface.dataset_adapter import DataSet
    from paraview.vtk.numpy_interface.dataset_adapter import PointSet


def get_data_array_and_zone(input, array_name, cell_data):
    """"""
    if cell_data:
        p = input.GetCellData().GetArray(array_name)
        z = input.GetCellData().GetArray("zone")
        n = input.GetNumberOfCells()
    else:
        # check if requested data array excists in point data
        p = input.GetPointData().GetArray(array_name)
        z = input.GetPointData().GetArray("zone")
        n = input.GetNumberOfPoints()

    return p, z, n


def _call_extract(source, proxy, array_name, ignore_zone, filter):
    if array_name in source.CellData.keys():
        extract = sum_and_zone_filter(
            proxy, array_name, ignore_zone, filter, cell_data=True
        )
    elif array_name in source.PointData.keys():
        extract = sum_and_zone_filter(
            proxy, array_name, ignore_zone, filter, cell_data=False
        )
    else:
        print("Error: {} variable not found in input".format(array_name))
        extract = [0.0, 0.0, 0.0]
    return extract


def sum_and_zone_filter_array(
    input: any, array_name: str, ignore_zone: list, filter: any = None, cell_data=True
) -> list:
    """Take an array data set and sum the values in the valid regions as specified through the filters"""

    sum = [0.0, 0.0, 0.0]
    # check if requested data array exists in cell data:
    p, z, num_locations = get_data_array_and_zone(input, array_name, cell_data)

    for x in range(num_locations):
        if len(ignore_zone) == 0:
            v = p.GetTuple(x)
            for i in range(0, 3):
                sum[i] += v[i]
        else:
            zone = z.GetValue(x)
            if zone not in ignore_zone:
                v = p.GetTuple(x)
                if filter is None or filter.test(input, x):
                    # print 'Zone: %i'%(zone)
                    for i in range(0, 3):
                        sum[i] += v[i]
    return sum


def sum_and_zone_filter(
    input: any, array_name: str, ignore_zone: list, filter: any = None, cell_data=True
) -> list:
    """Break down vtkMultiBlockDataSet object and sum desired arrays according to filters"""
    sum = [0.0, 0.0, 0.0]
    if input.IsA("vtkMultiBlockDataSet"):
        iter = input.NewIterator()
        iter.UnRegister(None)
        iter.InitTraversal()
        while not iter.IsDoneWithTraversal():
            cur_input = iter.GetCurrentDataObject()
            v = sum_and_zone_filter_array(
                cur_input, array_name, ignore_zone, filter, cell_data
            )
            for i in range(0, 3):
                sum[i] += v[i]
            iter.GoToNextItem()
    else:
        sum = sum_and_zone_filter_array(
            input, array_name, ignore_zone, filter, cell_data
        )

    return sum


class GeomFilterLT(object):
    '''Geometric filter, which tests whether an input (x,y,z)[idx] is less than a float "val"'''

    def __init__(self, val: int, idx: int) -> None:
        self.val = val
        self.idx = idx

    def test(self, input: any, x: int) -> bool:
        """Returns True if the (x,y,z)[idx] value is less than the filter threshold value"""
        centre = input.GetCellData().GetArray("centre").GetTuple(x)
        if centre[self.idx] < self.val:
            return True
        else:
            return False


class GeomFilterGT(object):
    '''Geometric filter, which tests whether an input (x,y,z)[idx] is greater than a float "val"'''

    def __init__(self, val: float, idx: int) -> None:
        #
        self.val = val
        self.idx = idx

    def test(self, input: any, x: int) -> bool:
        """Returns True if the (x,y,z)[idx] value is greater than the filter threshold value"""
        centre = input.GetCellData().GetArray("centre").GetTuple(x)
        if centre[self.idx] >= self.val:
            return True
        else:
            return False


def clean_vtk(
    vtk_object: object, cellDataToPointData: bool = True, mergeBlocks: bool = True
):
    """Takes a VTK object, and performs basic cleanup operations:
    cleanToGrid
    mergeBlocks [optional]
    cellDataToPointData [optional]

    Args:
        vtk_object: A loaded vtk object
    Optional:
        mergeBlocks: default=True-Whether to perform a mergeBlocks operation on the data
        cellDataToPointData: defult=True- Whether to perform a cellDataToPointData mapping on the object
    """
    data = pvs.CleantoGrid(Input=vtk_object)

    if mergeBlocks:
        data = pvs.MergeBlocks(Input=data)
        pvs.UpdatePipeline()

    if cellDataToPointData:
        data = pvs.CellDatatoPointData(Input=data)
        data.ProcessAllArrays = 1
        data.PassCellData = 1
        pvs.UpdatePipeline()

    return data


def calc_force_from_file(
    file_name: str,
    ignore_zone: list,
    half_model: bool = False,
    filter: any = None,
    **kwargs,
) -> Tuple[float, float]:
    """Calculate the pressure and friction force

    This function requires that the VTK file contains three cell data arrays
    called pressureforce, frictionforce and zone

    Args:
        file_name (str): the VTK file name including path
        ignore_zone (list): List of zones to be ignored

    Kwargs:
        half_nodel (bool): Does the data represent only half of the model
        filter (function): GeomFilter object

    Returns:
        float, float. pressure force and friction force
    """
    wall = pvs.PVDReader(FileName=file_name)
    wall.UpdatePipeline()

    return calc_force(wall, ignore_zone, half_model, filter, kwargs)


def calc_force_wall(
    file_root: str,
    ignore_zone: list,
    half_model: bool = False,
    filter: any = None,
    **kwargs,
) -> Tuple[float, float]:
    """Calculate pressure and friction forces at wall"""

    wall = pvs.PVDReader(FileName=file_root + "_wall.pvd")
    wall.UpdatePipeline()

    force = calc_force(wall, ignore_zone, half_model, filter, **kwargs)
    pvs.Delete(wall)
    del wall
    return force


def _get_logo_path(strapline=False):
    zcfd_home = Path(__file__).parents[5]
    if strapline:
        logo = "ZCFD_Mark_CMYK.png"
    else:
        logo = "ZCFD_Mark_CMYK_No_Strapline_trans.png"
    file_loc = zcfd_home / "share" / "assets" / logo
    return str(file_loc)


def vtk_logo_stamp(
    input: any, location="Upper Left Corner", logo_file=None, strapline=False
):
    """stamps a render view with a zcfd logo
    Inputs:
    input: paraview render view object
    Optional:
    location: str (default="Upper Left Corner") location of the logo
    logo_file: str (default=None) optional string to alternative logo file
    strapline: bool (default=False) whether to include the strapline in the logo"""
    logo1 = pvs.Logo()

    # get the logo file
    if logo_file:
        logo_path = logo_file
    else:
        logo_path = _get_logo_path(strapline)

    logo_texture = pvs.CreateTexture(logo_path)
    logo1.Texture = logo_texture
    logoDisplay = pvs.Show(logo1, input)
    logoDisplay.WindowLocation = location
    # TODO: Add the ability to adjust the logo size with paraview 13

    return logoDisplay


def vtk_text_stamp(
    input: any,
    text_str: str,
    fontsize=40,
    color=[1.0, 1.0, 1.0],
    location="Lower Left Corner",
    bold=False,
    justification=None,
    font="Courier",
):
    """stamps a renderview object with a text string
    Inputs:
    input: paraview render view object
    text_str: str text to be displayed
    Optional:
    fontsize: int (default=40) font size
    color: list (default=[1.0,1.0,1.0]) RGB color
    location: str (default="Lower Left Corner") location of the text
    bold: bool (default=False) whether the text is bold
    justification: str (default=None) justification of the text- by default it is set by the location
    font: str (default="Courier") font family
    """
    text = pvs.Text()
    text.Text = text_str
    textDisplay = pvs.Show(text, input)
    textDisplay.FontSize = fontsize
    textDisplay.Color = color
    textDisplay.Bold = int(bold)
    textDisplay.WindowLocation = location
    textDisplay.FontFamily = font
    if justification:
        textDisplay.Justification = justification
    else:
        textDisplay.Justification = location.split(" ")[1]
    return textDisplay


def plt_logo_stamp(ax, location=(0.9, 0.95), logo_file=None, strapline=False):
    """stamps a matplotlib plot with a zCFD logo
    Inputs:
    ax: matplotlib axis object
    Optional:
    location: tuple (default=(0.9,0.95)) location of the logo as a % of x and y axis position
    logo_file: str (default=None) optional string to alternative logo file
    strapline: bool (default=False) whether to include the strapline in the logo
    """
    if logo_file:
        logo_path = logo_file
    else:
        logo_path = _get_logo_path(strapline)
    logo = image.imread(logo_path)

    imagebox = OffsetImage(logo, zoom=0.25)
    ab = AnnotationBbox(imagebox, xy=location, xycoords="axes fraction", frameon=False)
    ax.add_artist(ab)

    return ax


def calc_force(
    surface_data: any,
    ignore_zone: list = [],
    half_model: bool = False,
    filter: any = None,
    **kwargs,
) -> Tuple[float, float]:
    """Calculate forces from a surface file

    Inputs:
    surface_data: VTKPointData object

    Optional:
    ignore_zones: list (default = [])
    half_model: bool (default = False)
    filter: callable (default = None)
    """

    sum_client = pvs.servermanager.Fetch(surface_data)

    pforce = _call_extract(
        surface_data, sum_client, "pressureforce", ignore_zone, filter
    )
    fforce = _call_extract(
        surface_data, sum_client, "frictionforce", ignore_zone, filter
    )

    if half_model:
        for i in range(0, 3):
            pforce[i] *= 2.0
            fforce[i] *= 2.0

    del sum_client

    return pforce, fforce


def calc_moment_wall(
    file_root: str,
    ignore_zone: list,
    half_model: bool = False,
    filter: any = None,
    **kwargs,
) -> Tuple[float, float]:
    """Calculate the pressure and friction moment

    This function requires that the _wall.pvd file contains three cell data arrays
    called pressuremomentx, frictionmomentx and zone

    Args:
        file_name (str): the VTK file name including path
        ignore_zone (list): List of zones to be ignored

    Kwargs:
        half_nodel (bool): Does the data represent only half of the model
        filter (function): GeomFilter object

    Returns:
        float, float. pressure force and friction force
    """
    wall = pvs.PVDReader(FileName=file_root + "_wall.pvd")
    wall.UpdatePipeline()

    moment = calc_moment(wall, ignore_zone, half_model, filter, **kwargs)
    pvs.Delete(wall)
    del wall
    return moment


def calc_moment(
    surface_data: any,
    ignore_zone: list = [],
    half_model: bool = False,
    filter: any = None,
    **kwargs,
) -> Tuple[float, float]:
    """Akin to zutil.post.calc_force: Function called by .calc_moment_wall"""

    if "ref_pt" in kwargs:
        p_metric = "pressuremomentx"
        f_metric = "frictionmomentx"
    else:
        p_metric = "pressuremoment"
        f_metric = "frictionmoment"

    sum_client = pvs.servermanager.Fetch(surface_data)

    pmoment = _call_extract(surface_data, sum_client, p_metric, ignore_zone, filter)
    fmoment = _call_extract(surface_data, sum_client, f_metric, ignore_zone, filter)

    if half_model:
        # This is only valid for X-Z plane reflection
        pmoment[0] += -pmoment[0]
        pmoment[1] += pmoment[1]
        pmoment[2] += -pmoment[2]

        fmoment[0] += -fmoment[0]
        fmoment[1] += fmoment[1]
        fmoment[2] += -fmoment[2]

    return pmoment, fmoment


def calc_lift_centre_of_action(
    force: Union[list, tuple], moment: Union[list, tuple], ref_point: Union[list, tuple]
) -> Tuple[tuple, float]:
    """Calculate centre of action of lift given a force, moement and reference point"""
    # longitudinal centre xs0 at zs0
    # spanwise centre ys0 at zs0
    # residual Mz moment (Mx=My=0) mzs0

    xs0 = ref_point[0] - moment[1] / force[2]
    ys0 = ref_point[1] + moment[0] / force[2]

    zs0 = ref_point[2]
    mzs0 = moment[2] - force[1] * (xs0 - ref_point[0]) + force[0] * (ys0 - ref_point[1])

    return (xs0, ys0, zs0), mzs0


def calc_drag_centre_of_action(
    force: Union[list, tuple], moment: Union[list, tuple], ref_point: Union[list, tuple]
) -> Tuple[tuple, float]:
    """Calculate centre of action of drag given a force, moement and reference point"""

    # longitudinal centre xs0 at zs0
    # spanwise centre ys0 at zs0
    # residual Mz moment (Mx=My=0) mzs0

    zs0 = ref_point[2] + moment[1] / force[0]
    ys0 = ref_point[1] - moment[2] / force[0]

    xs0 = ref_point[0]
    # moment[2] - force[1]*(xs0-ref_point[0]) + force[0]*(ys0-ref_point[1])
    mzs0 = 0.0

    return (xs0, ys0, zs0), mzs0


def get_span(wall: any) -> Tuple[float, float]:
    """Returns the min and max y ordinate

    Args:
        wall (vtkMultiBlockDataSet): The input surface

    Returns:
        (float,float). Min y, Max y
    """
    bounds = wall.GetDataInformation().GetBounds()
    y_min = bounds[2]
    y_max = bounds[3]

    return y_min, y_max


def get_chord(
    slice: any,
) -> Tuple[float, float]:
    """Returns the min and max x ordinate

    Args:
        wall (vtkMultiBlockDataSet): The input surface

    Returns:
        (float,float). Min x, Max x
    """
    bounds = slice.GetDataInformation().GetBounds()
    x_min = bounds[0]
    x_max = bounds[1]

    return x_min, x_max


def get_monitor_data(file: str, monitor_name: str, var_name: str) -> tuple:
    """Return the _report file data corresponding to a monitor point and variable name"""
    df = get_csv_data(file, True)
    names = list(df.keys())
    variable = str(monitor_name) + "_" + str(var_name)
    if variable in names:
        return (df[names[0]].tolist(), df[variable].tolist())
    else:
        print(
            "POST.PY: MONITOR POINT: "
            + str(monitor_name)
            + "_"
            + str(var_name)
            + " NOT FOUND in "
            + str(names)
        )


def for_each(surface: any, func: any, **kwargs) -> any:
    """Applies a function "func" to each "surface" vtkMultiBlockDataSet"""
    if surface.IsA("vtkMultiBlockDataSet"):
        iter = surface.NewIterator()
        iter.UnRegister(None)
        iter.InitTraversal()
        while not iter.IsDoneWithTraversal():
            cur_input = iter.GetCurrentDataObject()
            # numCells = cur_input.GetNumberOfCells()
            numPts = cur_input.GetNumberOfPoints()
            if numPts > 0:
                calc = DataSet(cur_input)
                pts = PointSet(cur_input)
                func(calc, pts, **kwargs)
            iter.GoToNextItem()
    else:
        calc = DataSet(surface)
        pts = PointSet(surface)
        func(calc, pts, **kwargs)


def cp_profile_wall_from_file(
    file_root: str, slice_normal: tuple, slice_origin: tuple, **kwargs
) -> dict:
    """Return chordwise cp profile from wall file"""

    wall = pvs.PVDReader(FileName=file_root)
    clean = clean_vtk(wall)

    pvs.Delete(wall)
    del wall
    profile = cp_profile(clean, slice_normal, slice_origin, **kwargs)
    pvs.Delete(clean)
    del clean

    return profile


def slice_metric(
    surface: any,
    slice_normal: tuple,
    slice_origin: tuple,
    metric_name: str,
    time_average: bool = False,
    **kwargs,
) -> dict:
    """Calculate metric profile from a VTK surface data slice

    Inputs:
    surface: VTKPointData object to take cp cut from
    slice_normal: [nx, ny, nz] normal vector to the cut
    slice_origin: [x, y, z] location of the slice
    metric_name: str name of the metric to be plotted

    Optional:
    time_average: whether to take the time averaged solution for unsteady simulations (default=False)

    Returns:
    slice_dict: dict = {multiblockID: {"chord", "metric"}}
    """

    slice = pvs.Slice(Input=surface, SliceType="Plane")

    slice.SliceType.Normal = slice_normal
    slice.SliceType.Origin = slice_origin

    slice.UpdatePipeline()

    if time_average:
        temporal = pvs.TemporalStatistics(Input=slice)
        temporal.ComputeMaximum = 0
        temporal.ComputeStandardDeviation = 0
        temporal.ComputeMinimum = 0
        temporal.UpdatePipeline()
        pvs.Delete(slice)
        del slice
        slice = temporal

    offset = get_chord(slice)

    chord_calc = pvs.Calculator(Input=slice)
    chord_calc.AttributeType = "Point Data"
    chord_calc.Function = (
        "(coords.iHat - " + str(offset[0]) + ")/" + str(offset[1] - offset[0])
    )
    chord_calc.ResultArrayName = "chord"

    sorted_line = pvs.PlotOnSortedLines(Input=chord_calc)
    sorted_line.UpdatePipeline()

    rawData = pvs.servermanager.Fetch(sorted_line)
    slice_dict = {}
    for block_index in range(rawData.GetNumberOfBlocks()):
        data = DataSet(rawData.GetBlock(block_index))
        slice_dict[block_index] = {
            metric_name: data.PointData[metric_name],
            "chord": data.PointData["chord"],
        }

    # clean up
    pvs.Delete(chord_calc)
    del chord_calc
    pvs.Delete(slice)
    del slice
    pvs.Delete(sorted_line)
    del sorted_line

    return slice_dict


def cp_profile(surface, normal, slice_loc) -> dict:
    """Calculate cp profile from a VTK surface data slice

    Inputs:
    surface: VTKPointData object to take cp cut from
    slice_normal: [nx, ny, nz] normal vector to the cut
    slice_origin: [x, y, z] location of the slice
    metric_name: str name of the metric to be plotted

    Optional:
    time_average: whether to take the time averaged solution for unsteady simulations (default=False)

    Returns:
    slice_dict: dict = {multiblockID: {"chord", "cp"}}
    """
    return slice_metric(surface, normal, slice_loc, "cp")


def cf_profile_wall_from_file(
    file_root: tuple, slice_normal: tuple, slice_origin: tuple, **kwargs
) -> dict:
    """Force coefficient calculation at slice loaction for a file string"""
    wall = pvs.PVDReader(FileName=file_root + "_wall.pvd")
    clean = pvs.CleantoGrid(Input=wall)
    clean.UpdatePipeline()
    inp = pvs.servermanager.Fetch(clean)
    if inp.IsA("vtkMultiBlockDataSet"):
        inp = pvs.MergeBlocks(Input=clean)
    else:
        inp = clean

    pvs.Delete(wall)
    del wall
    out = cf_profile(inp, slice_normal, slice_origin, **kwargs)
    pvs.Delete(clean)
    del clean
    pvs.Delete(inp)
    del inp

    return out


def cf_profile(surface: tuple, slice_normal: tuple, slice_loc: tuple, **kwargs) -> None:
    '''Calculate the force coefficient profile for a slice defined with "slice_normal" and "slice_origin"'''

    cf_calc = pvs.Calculator(Input=surface)

    cf_calc.AttributeType = "Point Data"
    cf_calc.Function = "mag(cf)"
    cf_calc.ResultArrayName = "cfmag"

    return slice_metric(cf_calc, slice_normal, slice_loc, "cfmag")


def screenshot() -> None:
    """TODO finalise..."""
    # position camera
    view = pvs.GetActiveView()
    if not view:
        # When using the ParaView UI, the View will be present, not otherwise.
        view = pvs.CreateRenderView()
    view.CameraViewUp = [0, 0, 1]
    view.CameraFocalPoint = [0, 0, 0]
    view.CameraViewAngle = 45
    view.CameraPosition = [5, 0, 0]

    # draw the object
    pvs.Show()

    # set the background color
    view.Background = [1, 1, 1]  # white

    # set image size
    view.ViewSize = [200, 300]  # [width, height]

    dp = pvs.GetDisplayProperties()

    # set point color
    dp.AmbientColor = [1, 0, 0]  # red

    # set surface color
    dp.DiffuseColor = [0, 1, 0]  # blue

    # set point size
    dp.PointSize = 2

    # set representation
    dp.Representation = "Surface"

    pvs.Render()

    # save screenshot
    pvs.WriteImage("test.png")


def sum_array(input: any, array_name: str) -> list:
    '''Sum of desired array over every cell in "input"'''
    sum = [0.0, 0.0, 0.0]
    p = input.GetCellData().GetArray(array_name)
    numCells = input.GetNumberOfCells()
    for x in range(numCells):
        v = p.GetTuple(x)
        for i in range(0, 3):
            sum[i] += v[i]
    return sum


def get_case_parameters_str(case_name: str, **kwargs) -> Optional[str]:
    "Returns a string of the input dictionary for a zCFD case"
    _data_dir = analysis.data.data_dir
    if "data_dir" in kwargs:
        _data_dir = kwargs["data_dir"]

    try:
        # Get contents of local file
        with open(_data_dir + "/" + case_name + ".py") as f:
            case_file_str = f.read()

            if case_file_str is not None:
                # print status_file_str
                return case_file_str
            else:
                print("WARNING: " + case_name + ".py file not found")
                return None
    except:
        print("WARNING: " + case_name + ".py file not found")
        return None


def get_case_parameters(case_name: str, **kwargs) -> dict:
    """Returns the zCFD input parameter dictionary"""
    case_file_str = get_case_parameters_str(case_name, **kwargs)
    namespace = {}
    exec(case_file_str, namespace)
    return namespace["parameters"]


def get_status_dict(case_name: str, **kwargs) -> Optional[str]:
    """Returns a string of the status of a zCFD run"""
    _data_dir = analysis.data.data_dir
    if "data_dir" in kwargs:
        _data_dir = kwargs["data_dir"]

    try:
        # Get contents of local file
        with open(_data_dir + "/" + case_name + "_status.txt") as f:
            status_file_str = f.read()

            if status_file_str is not None:
                # print status_file_str
                return json.loads(status_file_str)
            else:
                print("WARNING: " + case_name + "_status.txt file not found")
                return None
    except Exception as e:
        print("WARNING: " + case_name + "_status.txt file not found")
        print("Caught exception " + str(e))
        return None


def get_num_procs(case_name: str, **kwargs) -> Optional[int]:
    """Returns the number of processes used in a zCFD run"""
    # remote_host,remote_dir,case_name):
    status = get_status_dict(case_name, **kwargs)
    if status is not None:
        if "num processor" in status:
            return status["num processor"]
        else:
            return None
    else:
        print("status file not found")


def get_case_root(case_name: str, num_procs: Optional[str]) -> str:
    """Returns the output folder path for a specified zCFD run"""
    if num_procs is None:
        num_procs = get_num_procs(case_name)
    return case_name + "_P" + str(num_procs) + "_OUTPUT/" + case_name


def get_case_report(case: str) -> str:
    """Returns the path to the zCFD report file for a specific run"""
    return case + "_report.csv"


def get_case_success(case: str) -> bool:
    """Scans the zCFD log file to identify if a run has successfully completed"""
    # remove .py file exension if its provided
    case_clean = clean_name(case)
    try:
        with open(case_clean + ".log", "r") as f:
            for line in f.readlines():
                if "Solver loop finished" in line:
                    return True
    except FileExistsError as e:
        print(e)
        return False
    return False


class ProgressBar(object):
    """Class to display a progress bar when performing rendering opteraions"""

    def __init__(self) -> None:
        self.pbar = tqdm(total=100)

    def __iadd__(self, v):
        self.pbar.update(v)
        return self

    def complete(self):
        self.pbar.close()

    def update(self, i):
        self.pbar.update(i)
