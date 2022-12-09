""" Helper functions for accessing Paraview functionality
.. moduleauthor:: Zenotech Ltd
"""


from tqdm import tqdm
from IPython.display import HTML, Javascript, display
import uuid
import csv
from zutil import analysis
import time
import math
from zutil import mag
import json
from zutil import rotate_vector
from paraview.simple import *
from builtins import object
from past.utils import old_div
from builtins import range
from builtins import str
from future import standard_library

standard_library.install_aliases()
# from paraview.vtk.util import numpy_support
try:
    from paraview.vtk.dataset_adapter import numpyTovtkDataArray
    from paraview.vtk.dataset_adapter import Table
    from paraview.vtk.dataset_adapter import PolyData
    from paraview.vtk.dataset_adapter import DataSetAttributes
    from paraview.vtk.dataset_adapter import DataSet
    from paraview.vtk.dataset_adapter import CompositeDataSet
    from paraview.vtk.dataset_adapter import PointSet
except:
    from paraview.vtk.numpy_interface.dataset_adapter import numpyTovtkDataArray
    from paraview.vtk.numpy_interface.dataset_adapter import Table
    from paraview.vtk.numpy_interface.dataset_adapter import PolyData
    from paraview.vtk.numpy_interface.dataset_adapter import DataSetAttributes
    from paraview.vtk.numpy_interface.dataset_adapter import DataSet
    from paraview.vtk.numpy_interface.dataset_adapter import CompositeDataSet
    from paraview.vtk.numpy_interface.dataset_adapter import PointSet


def sum_and_zone_filter_array(input, array_name, ignore_zone, filter=None):
    sum = [0.0, 0.0, 0.0]
    p = input.GetCellData().GetArray(array_name)
    z = input.GetCellData().GetArray("zone")
    numCells = input.GetNumberOfCells()
    for x in range(numCells):
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


def sum_and_zone_filter(input, array_name, ignore_zone, filter=None):
    sum = [0.0, 0.0, 0.0]
    if input.IsA("vtkMultiBlockDataSet"):
        iter = input.NewIterator()
        iter.UnRegister(None)
        iter.InitTraversal()
        while not iter.IsDoneWithTraversal():
            cur_input = iter.GetCurrentDataObject()
            v = sum_and_zone_filter_array(cur_input, array_name, ignore_zone, filter)
            for i in range(0, 3):
                sum[i] += v[i]
            iter.GoToNextItem()
    else:
        sum = sum_and_zone_filter_array(input, array_name, ignore_zone, filter)

    return sum


class GeomFilterLT(object):
    def __init__(self, val, idx):
        #
        self.val = val
        self.idx = idx

    def test(self, input, x):
        centre = input.GetCellData().GetArray("centre").GetTuple(x)
        if centre[self.idx] < self.val:
            return True
        else:
            return False


class GeomFilterGT(object):
    def __init__(self, val, idx):
        #
        self.val = val
        self.idx = idx

    def test(self, input, x):
        centre = input.GetCellData().GetArray("centre").GetTuple(x)
        if centre[self.idx] >= self.val:
            return True
        else:
            return False


def calc_force_from_file(
    file_name, ignore_zone, half_model=False, filter=None, **kwargs
):
    """ Calculates the pressure and friction force

    This function requires that the VTK file contains three cell data arrays
    called pressureforce, frictionforce and zone

    Args:
        file_name (str): the VTK file name including path
        ignore_zone (list): List of zones to be ignored

    Kwargs:
        half_nodel (bool): Does the data represent only half of the model
        filter (function):

    Returns:
        float, float. pressure force and friction force
    """
    wall = PVDReader(FileName=file_name)
    wall.UpdatePipeline()

    return calc_force(wall, ignore_zone, half_model, filter, kwargs)


def calc_force_wall(file_root, ignore_zone, half_model=False, filter=None, **kwargs):

    wall = PVDReader(FileName=file_root + "_wall.pvd")
    wall.UpdatePipeline()

    force = calc_force(wall, ignore_zone, half_model, filter, **kwargs)
    Delete(wall)
    del wall
    return force


def calc_force(surface_data, ignore_zone, half_model=False, filter=None, **kwargs):

    alpha = 0.0
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    beta = 0.0
    if "beta" in kwargs:
        beta = kwargs["beta"]

    sum_client = servermanager.Fetch(surface_data)

    pforce = sum_and_zone_filter(sum_client, "pressureforce", ignore_zone, filter)
    fforce = sum_and_zone_filter(sum_client, "frictionforce", ignore_zone, filter)

    pforce = rotate_vector(pforce, alpha, beta)
    fforce = rotate_vector(fforce, alpha, beta)

    if half_model:
        for i in range(0, 3):
            pforce[i] *= 2.0
            fforce[i] *= 2.0

    del sum_client

    return pforce, fforce


def calc_moment(surface_data, ignore_zone, half_model=False, filter=None, **kwargs):

    alpha = 0.0
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    beta = 0.0
    if "beta" in kwargs:
        beta = kwargs["beta"]

    if "ref_pt" in kwargs:
        sum_client = servermanager.Fetch(surface_data)
        if sum_client.GetCellData().GetArray("pressuremomentx"):
            pmoment = sum_and_zone_filter(
                sum_client, "pressuremomentx", ignore_zone, filter
            )
            fmoment = sum_and_zone_filter(
                sum_client, "frictionmomentx", ignore_zone, filter
            )

            pmoment = rotate_vector(pmoment, alpha, beta)
            fmoment = rotate_vector(fmoment, alpha, beta)
            # fforce = rotate_vector(fforce,alpha,beta)

            if half_model:
                # This is only valid for X-Z plane reflection
                pmoment[0] += -pmoment[0]
                pmoment[1] += pmoment[1]
                pmoment[2] += -pmoment[2]

                fmoment[0] += -fmoment[0]
                fmoment[1] += fmoment[1]
                fmoment[2] += -fmoment[2]

            return pmoment, fmoment

    else:
        sum_client = servermanager.Fetch(surface_data)
        pmoment = sum_and_zone_filter(sum_client, "pressuremoment", ignore_zone, filter)
        fmoment = sum_and_zone_filter(sum_client, "frictionmoment", ignore_zone, filter)

        pmoment = rotate_vector(pmoment, alpha, beta)
        fmoment = rotate_vector(fmoment, alpha, beta)
        # fforce = rotate_vector(fforce,alpha,beta)

        if half_model:
            # This is only valid for X-Z plane reflection
            pmoment[0] += -pmoment[0]
            pmoment[1] += pmoment[1]
            pmoment[2] += -pmoment[2]

            fmoment[0] += -fmoment[0]
            fmoment[1] += fmoment[1]
            fmoment[2] += -fmoment[2]

        return pmoment, fmoment


def calc_lift_centre_of_action(force, moment, ref_point):
    # longitudinal centre xs0 at zs0
    # spanwise centre ys0 at zs0
    # residual Mz moment (Mx=My=0) mzs0

    xs0 = ref_point[0] - moment[1] / force[2]
    ys0 = ref_point[1] + moment[0] / force[2]

    zs0 = ref_point[2]
    mzs0 = moment[2] - force[1] * (xs0 - ref_point[0]) + force[0] * (ys0 - ref_point[1])

    return (xs0, ys0, zs0), mzs0


def calc_drag_centre_of_action(force, moment, ref_point):
    # longitudinal centre xs0 at zs0
    # spanwise centre ys0 at zs0
    # residual Mz moment (Mx=My=0) mzs0

    zs0 = ref_point[2] + moment[1] / force[0]
    ys0 = ref_point[1] - moment[2] / force[0]

    xs0 = ref_point[0]
    # moment[2] - force[1]*(xs0-ref_point[0]) + force[0]*(ys0-ref_point[1])
    mzs0 = 0.0

    return (xs0, ys0, zs0), mzs0


def move_moment_ref_point(moment, ref_point, new_ref_point):
    pass


def get_span(wall):
    """ Returns the min and max y ordinate

    Args:
        wall (vtkMultiBlockDataSet): The input surface

    Returns:
        (float,float). Min y, Max y
    """
    Calculator1 = Calculator(Input=wall)

    Calculator1.AttributeType = "Point Data"
    Calculator1.Function = "coords.jHat"
    Calculator1.ResultArrayName = "ypos"
    Calculator1.UpdatePipeline()

    ymin = MinMax(Input=Calculator1)
    ymin.Operation = "MIN"
    ymin.UpdatePipeline()

    ymin_client = servermanager.Fetch(ymin)

    min_pos = ymin_client.GetPointData().GetArray("ypos").GetValue(0)

    ymax = MinMax(Input=Calculator1)
    ymax.Operation = "MAX"
    ymax.UpdatePipeline()

    ymax_client = servermanager.Fetch(ymax)

    max_pos = ymax_client.GetPointData().GetArray("ypos").GetValue(0)

    Delete(ymin)
    Delete(ymax)
    Delete(Calculator1)

    return [min_pos, max_pos]


def get_chord(slice, rotate_geometry=[0.0, 0.0, 0.0]):
    """ Returns the min and max x ordinate

    Args:
        wall (vtkMultiBlockDataSet): The input surface

    Returns:
        (float,float). Min x, Max x
    """

    transform = Transform(Input=slice, Transform="Transform")
    transform.Transform.Scale = [1.0, 1.0, 1.0]
    transform.Transform.Translate = [0.0, 0.0, 0.0]
    transform.Transform.Rotate = rotate_geometry
    transform.UpdatePipeline()

    Calculator1 = Calculator(Input=transform)

    Calculator1.AttributeType = "Point Data"
    Calculator1.Function = "coords.iHat"
    Calculator1.ResultArrayName = "xpos"
    Calculator1.UpdatePipeline()

    xmin = MinMax(Input=Calculator1)
    xmin.Operation = "MIN"
    xmin.UpdatePipeline()

    xmin_client = servermanager.Fetch(xmin)

    min_pos = xmin_client.GetPointData().GetArray("xpos").GetValue(0)

    xmax = MinMax(Input=Calculator1)
    xmax.Operation = "MAX"
    xmax.UpdatePipeline()

    xmax_client = servermanager.Fetch(xmax)

    max_pos = xmax_client.GetPointData().GetArray("xpos").GetValue(0)

    Delete(xmin)
    Delete(xmax)
    Delete(Calculator1)
    Delete(transform)

    return [min_pos, max_pos]


def get_chord_spanwise(slice):

    Calculator1 = Calculator(Input=slice)

    Calculator1.AttributeType = "Point Data"
    Calculator1.Function = "coords.jHat"
    Calculator1.ResultArrayName = "ypos"
    Calculator1.UpdatePipeline()

    ymin = MinMax(Input=Calculator1)
    ymin.Operation = "MIN"
    ymin.UpdatePipeline()

    ymin_client = servermanager.Fetch(ymin)

    min_pos = ymin_client.GetPointData().GetArray("ypos").GetValue(0)

    ymax = MinMax(Input=Calculator1)
    ymax.Operation = "MAX"
    ymax.UpdatePipeline()

    ymax_client = servermanager.Fetch(ymax)

    max_pos = ymax_client.GetPointData().GetArray("ypos").GetValue(0)

    Delete(ymin)
    Delete(ymax)
    Delete(Calculator1)

    return [min_pos, max_pos]


def get_monitor_data(file, monitor_name, var_name):
    """ Return the _report file data corresponding to a monitor point and variable name
        """
    monitor = CSVReader(FileName=[file])
    monitor.HaveHeaders = 1
    monitor.MergeConsecutiveDelimiters = 1
    monitor.UseStringDelimiter = 0
    monitor.DetectNumericColumns = 1
    monitor.FieldDelimiterCharacters = " "
    monitor.UpdatePipeline()
    monitor_client = servermanager.Fetch(monitor)
    table = Table(monitor_client)
    data = table.RowData
    names = list(data.keys())
    num_var = len(names) - 2
    if str(monitor_name) + "_" + str(var_name) in names:
        index = names.index(str(monitor_name) + "_" + str(var_name))
        return (data[names[0]], data[names[index]])
    else:
        print(
            "POST.PY: MONITOR POINT: "
            + str(monitor_name)
            + "_"
            + str(var_name)
            + " NOT FOUND"
        )


def residual_plot(file, pl, ncol=3):
    """ Plot the _report file
    """
    from matplotlib.ticker import FormatStrFormatter
    l2norm = CSVReader(FileName=[file])
    l2norm.HaveHeaders = 1
    l2norm.MergeConsecutiveDelimiters = 1
    l2norm.UseStringDelimiter = 0
    l2norm.DetectNumericColumns = 1
    l2norm.FieldDelimiterCharacters = " "
    l2norm.UpdatePipeline()

    l2norm_client = servermanager.Fetch(l2norm)

    table = Table(l2norm_client)

    data = table.RowData

    names = list(data.keys())

    num_var = len(names) - 2
    num_rows = (old_div((num_var - 1), ncol)) + 1

    fig = pl.figure(figsize=(3 * ncol, 3 * num_rows), dpi=100, facecolor="w", edgecolor="k")

    #fig.suptitle(file, fontweight="bold")

    i = 1
    for var_name in names:
        if var_name != "Cycle" and var_name != "RealTimeStep":
            ax = fig.add_subplot(num_rows, ncol, i)
            i = i + 1
            if "rho" in var_name:
                ax.set_yscale("log")
                ax.set_ylabel("l2norm " + var_name, multialignment="center")
            else:
                ax.set_ylabel(var_name, multialignment="center")

            ax.grid(True)
            ax.set_xlabel("Cycles")
            ax.tick_params(axis='both', labelsize='small')
            #ax.ticklabel_format(axis='both', style='sci')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

            ax.plot(data["Cycle"], data[var_name], color="r", label=var_name)
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    pl.tight_layout()

def for_each(surface, func, **kwargs):
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


def cp_profile_wall_from_file(file_root, slice_normal, slice_origin, **kwargs):

    wall = PVDReader(FileName=file_root + "_wall.pvd")
    clean = CleantoGrid(Input=wall)
    clean.UpdatePipeline()
    inp = servermanager.Fetch(clean)
    if inp.IsA("vtkMultiBlockDataSet"):
        inp = MergeBlocks(Input=clean)
    else:
        inp = clean

    Delete(wall)
    del wall
    profile = cp_profile(inp, slice_normal, slice_origin, **kwargs)
    Delete(clean)
    del clean
    Delete(inp)
    del inp

    return profile


def cp_profile_wall_from_file_span(file_root, slice_normal, slice_origin, **kwargs):

    wall = PVDReader(FileName=file_root + "_wall.pvd")
    if 'time' in kwargs:
        wall.UpdatePipeline(kwargs['time'])
    clean = CleantoGrid(Input=wall)
    clean.UpdatePipeline()
    inp = servermanager.Fetch(clean)
    if inp.IsA("vtkMultiBlockDataSet"):
        inp = MergeBlocks(Input=clean)
    else:
        inp = clean

    Delete(wall)
    del wall
    profile = cp_profile_span(inp, slice_normal, slice_origin, **kwargs)
    Delete(clean)
    del clean
    Delete(inp)
    del inp

    return profile


def cp_profile(surface, slice_normal, slice_origin, **kwargs):

    alpha = 0.0
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    beta = 0.0
    if "beta" in kwargs:
        beta = kwargs["beta"]

    time_average = False
    if "time_average" in kwargs:
        time_average = kwargs["time_average"]

    rotate_geometry = [0.0, 0.0, 0.0]
    if "rotate_geometry" in kwargs:
        rotate_geometry = kwargs["rotate_geometry"]

    clean = CleantoGrid(Input=surface)
    clean.UpdatePipeline()

    point_data = CellDatatoPointData(Input=clean)
    point_data.PassCellData = 1
    Delete(clean)
    del clean

    if "filter" in kwargs:
        filter_zones = kwargs["filter"]
        calc_str = "".join("(zone={:d})|".format(i) for i in filter_zones)
        filter_data = Calculator(Input=point_data)
        filter_data.AttributeType = "Cell Data"
        filter_data.Function = "if (" + calc_str[:-1] + ", 1, 0)"
        filter_data.ResultArrayName = "zonefilter"
        filter_data.UpdatePipeline()
        Delete(point_data)
        del point_data
        point_data = Threshold(Input=filter_data)
        point_data.Scalars = ["CELLS", "zonefilter"]
        point_data.ThresholdRange = [1.0, 1.0]
        point_data.UpdatePipeline()
        Delete(filter_data)
        del filter_data

    surf = ExtractSurface(Input=point_data)
    surf_normals = GenerateSurfaceNormals(Input=surf)
    surf_normals.UpdatePipeline()

    Delete(surf)
    del surf
    Delete(point_data)
    del point_data

    point_data = surf_normals

    slice = Slice(Input=point_data, SliceType="Plane")

    slice.SliceType.Normal = slice_normal
    slice.SliceType.Origin = slice_origin

    slice.UpdatePipeline()
    Delete(point_data)
    del point_data

    if time_average:
        temporal = TemporalStatistics(Input=slice)
        temporal.ComputeMaximum = 0
        temporal.ComputeStandardDeviation = 0
        temporal.ComputeMinimum = 0
        temporal.UpdatePipeline()
        Delete(slice)
        del slice
        slice = temporal

    offset = get_chord(slice, rotate_geometry)

    transform = Transform(Input=slice, Transform="Transform")
    transform.Transform.Scale = [1.0, 1.0, 1.0]
    transform.Transform.Translate = [0.0, 0.0, 0.0]
    transform.Transform.Rotate = rotate_geometry
    transform.UpdatePipeline()

    if "chord_func" in kwargs:
        pass
    else:
        chord_calc = Calculator(Input=transform)
        chord_calc.AttributeType = "Point Data"
        chord_calc.Function = (
            "(coords.iHat - " + str(offset[0]) + ")/" + str(offset[1] - offset[0])
        )
        chord_calc.ResultArrayName = "chord"

    # Attempt to calculate forces
    pforce = [0.0, 0.0, 0.0]
    fforce = [0.0, 0.0, 0.0]
    pmoment = [0.0, 0.0, 0.0]
    fmoment = [0.0, 0.0, 0.0]
    pmomentx = [0.0, 0.0, 0.0]
    fmomentx = [0.0, 0.0, 0.0]
    pmomenty = [0.0, 0.0, 0.0]
    fmomenty = [0.0, 0.0, 0.0]
    pmomentz = [0.0, 0.0, 0.0]
    fmomentz = [0.0, 0.0, 0.0]

    sum = MinMax(Input=slice)
    sum.Operation = "SUM"
    sum.UpdatePipeline()

    sum_client = servermanager.Fetch(sum)
    if sum_client.GetCellData().GetArray("pressureforce"):
        pforce = sum_client.GetCellData().GetArray("pressureforce").GetTuple(0)
        pforce = rotate_vector(pforce, alpha, beta)

    if sum_client.GetCellData().GetArray("frictionforce"):
        fforce = sum_client.GetCellData().GetArray("frictionforce").GetTuple(0)
        fforce = rotate_vector(fforce, alpha, beta)
        """
        # Add sectional force integration
        sorted_line = PlotOnSortedLines(Input=chord_calc)
        sorted_line.UpdatePipeline()
        sorted_line = servermanager.Fetch(sorted_line)
        cp_array = sorted_line.GetCellData().GetArray("cp")

        for i in range(0,len(cp_array)):
            sorted_line.GetPointData().GetArray("X")
            pass
        """
    if sum_client.GetCellData().GetArray("pressuremoment"):
        pmoment = sum_client.GetCellData().GetArray("pressuremoment").GetTuple(0)
        pmoment = rotate_vector(pmoment, alpha, beta)

    if sum_client.GetCellData().GetArray("frictionmoment"):
        fmoment = sum_client.GetCellData().GetArray("frictionmoment").GetTuple(0)
        fmoment = rotate_vector(fmoment, alpha, beta)

    if sum_client.GetCellData().GetArray("pressuremomentx"):
        pmomentx = sum_client.GetCellData().GetArray("pressuremomentx").GetTuple(0)
        pmomentx = rotate_vector(pmomentx, alpha, beta)

    if sum_client.GetCellData().GetArray("frictionmomentx"):
        fmomentx = sum_client.GetCellData().GetArray("frictionmomentx").GetTuple(0)
        fmomentx = rotate_vector(fmomentx, alpha, beta)

    if "func" in kwargs:
        sorted_line = PlotOnSortedLines(Input=chord_calc)
        sorted_line.UpdatePipeline()
        extract_client = servermanager.Fetch(sorted_line)
        for_each(extract_client, **kwargs)

    Delete(chord_calc)
    del chord_calc
    Delete(sum)
    del sum
    del sum_client
    Delete(slice)
    del slice
    Delete(sorted_line)
    del sorted_line
    del extract_client

    return {
        "pressure force": pforce,
        "friction force": fforce,
        "pressure moment": pmoment,
        "friction moment": fmoment,
    }


def cp_profile_span(surface, slice_normal, slice_origin, **kwargs):

    alpha = 0.0
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    beta = 0.0
    if "beta" in kwargs:
        beta = kwargs["beta"]

    point_data = CellDatatoPointData(Input=surface)
    point_data.PassCellData = 1
    clip = Clip(Input=point_data, ClipType="Plane")
    clip.ClipType.Normal = [0.0, 1.0, 0.0]
    clip.ClipType.Origin = [0.0, 0.0, 0.0]
    clip.UpdatePipeline()

    slice = Slice(Input=clip, SliceType="Plane")

    slice.SliceType.Normal = slice_normal
    slice.SliceType.Origin = slice_origin

    slice.UpdatePipeline()

    offset = get_chord_spanwise(slice)
    # define the cuts and make sure the is the one one you want
    # make the
    chord_calc = Calculator(Input=slice)

    chord_calc.AttributeType = "Point Data"
    chord_calc.Function = (
        "(coords.jHat - " + str(offset[0]) + ")/" + str(offset[1] - offset[0])
    )
    chord_calc.ResultArrayName = "chord"

    sum = MinMax(Input=slice)
    sum.Operation = "SUM"
    sum.UpdatePipeline()

    sum_client = servermanager.Fetch(sum)
    pforce = sum_client.GetCellData().GetArray("pressureforce").GetTuple(0)
    fforce = sum_client.GetCellData().GetArray("frictionforce").GetTuple(0)

    pforce = rotate_vector(pforce, alpha, beta)
    fforce = rotate_vector(fforce, alpha, beta)

    if "func" in kwargs:
        sorted_line = PlotOnSortedLines(Input=chord_calc)
        sorted_line.UpdatePipeline()
        extract_client = servermanager.Fetch(sorted_line)
        for_each(extract_client, **kwargs)

    return {"pressure force": pforce, "friction force": fforce}


def cf_profile_wall_from_file(file_root, slice_normal, slice_origin, **kwargs):

    wall = PVDReader(FileName=file_root + "_wall.pvd")
    clean = CleantoGrid(Input=wall)
    clean.UpdatePipeline()
    inp = servermanager.Fetch(clean)
    if inp.IsA("vtkMultiBlockDataSet"):
        inp = MergeBlocks(Input=clean)
    else:
        inp = clean

    Delete(wall)
    del wall
    profile = cf_profile(inp, slice_normal, slice_origin, **kwargs)
    Delete(clean)
    del clean
    Delete(inp)
    del inp

    return profile


def cf_profile(surface, slice_normal, slice_origin, **kwargs):

    alpha = 0.0
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    beta = 0.0
    if "beta" in kwargs:
        beta = kwargs["beta"]

    point_data = CellDatatoPointData(Input=surface)
    point_data.PassCellData = 1

    slice = Slice(Input=point_data, SliceType="Plane")

    slice.SliceType.Normal = slice_normal
    slice.SliceType.Origin = slice_origin

    slice.UpdatePipeline()

    offset = get_chord(slice)

    chord_calc = Calculator(Input=slice)

    chord_calc.AttributeType = "Point Data"
    chord_calc.Function = (
        "(coords.iHat - " + str(offset[0]) + ")/" + str(offset[1] - offset[0])
    )
    chord_calc.ResultArrayName = "chord"

    cf_calc = Calculator(Input=chord_calc)

    cf_calc.AttributeType = "Point Data"
    cf_calc.Function = "mag(cf)"
    cf_calc.ResultArrayName = "cfmag"

    sum = MinMax(Input=slice)
    sum.Operation = "SUM"
    sum.UpdatePipeline()

    sum_client = servermanager.Fetch(sum)
    pforce = sum_client.GetCellData().GetArray("pressureforce").GetTuple(0)
    fforce = sum_client.GetCellData().GetArray("frictionforce").GetTuple(0)

    pforce = rotate_vector(pforce, alpha, beta)
    fforce = rotate_vector(fforce, alpha, beta)

    if "func" in kwargs:
        sorted_line = PlotOnSortedLines(Input=cf_calc)
        sorted_line.UpdatePipeline()
        extract_client = servermanager.Fetch(sorted_line)
        for_each(extract_client, **kwargs)

    return {"pressure force": pforce, "friction force": fforce}


def get_csv_data(filename, header=False, remote=False, delim=" "):
    """ Get csv data
    """
    if remote:
        theory = CSVReader(FileName=[filename])
        theory.HaveHeaders = 0
        if header:
            theory.HaveHeaders = 1
        theory.MergeConsecutiveDelimiters = 1
        theory.UseStringDelimiter = 0
        theory.DetectNumericColumns = 1
        theory.FieldDelimiterCharacters = delim
        theory.UpdatePipeline()
        theory_client = servermanager.Fetch(theory)
        table = Table(theory_client)
        data = table.RowData
    else:
        import pandas as pd

        if not header:
            data = pd.read_csv(filename, sep=delim, header=None)
        else:
            data = pd.read_csv(filename, sep=delim)
    return data


def get_fw_csv_data(filename, widths, header=False, remote=False, **kwargs):

    if remote:
        theory = CSVReader(FileName=[filename])
        theory.HaveHeaders = 0
        theory.MergeConsecutiveDelimiters = 1
        theory.UseStringDelimiter = 0
        theory.DetectNumericColumns = 1
        theory.FieldDelimiterCharacters = " "
        theory.UpdatePipeline()

        theory_client = servermanager.Fetch(theory)

        table = Table(theory_client)

        data = table.RowData

    else:
        import pandas as pd

        if not header:
            data = pd.read_fwf(filename, sep=" ", header=None, widths=widths, **kwargs)
        else:
            data = pd.read_fwf(filename, sep=" ", width=widths, **kwargs)

    return data


def screenshot(wall):
    # position camera
    view = GetActiveView()
    if not view:
        # When using the ParaView UI, the View will be present, not otherwise.
        view = CreateRenderView()
    view.CameraViewUp = [0, 0, 1]
    view.CameraFocalPoint = [0, 0, 0]
    view.CameraViewAngle = 45
    view.CameraPosition = [5, 0, 0]

    # draw the object
    Show()

    # set the background color
    view.Background = [1, 1, 1]  # white

    # set image size
    view.ViewSize = [200, 300]  # [width, height]

    dp = GetDisplayProperties()

    # set point color
    dp.AmbientColor = [1, 0, 0]  # red

    # set surface color
    dp.DiffuseColor = [0, 1, 0]  # blue

    # set point size
    dp.PointSize = 2

    # set representation
    dp.Representation = "Surface"

    Render()

    # save screenshot
    WriteImage("test.png")


def sum_array(input, array_name):
    sum = [0.0, 0.0, 0.0]
    p = input.GetCellData().GetArray(array_name)
    numCells = input.GetNumberOfCells()
    for x in range(numCells):
        v = p.GetTuple(x)
        for i in range(0, 3):
            sum[i] += v[i]
    return sum


def get_case_file():
    with cd(remote_dir):
        get(case_name + ".py", "%(path)s")


def cat_case_file(remote_dir, case_name):
    with cd(remote_dir):
        with hide("output", "running", "warnings"), settings(warn_only=True):
            # cmd = 'cat '+case_name+'.py'
            import io

            contents = io.StringIO()
            get(case_name + ".py", contents)
            # operate on 'contents' like a file object here, e.g. 'print
            return contents.getvalue()


def cat_status_file(remote_dir, case_name):

    with cd(remote_dir), hide("output", "running", "warnings"), settings(
        warn_only=True
    ):
        # cmd = 'cat '+case_name+'_status.txt'
        import io

        contents = io.StringIO()
        result = get(case_name + "_status.txt", contents)
        if result.succeeded:
            # operate on 'contents' like a file object here, e.g. 'print
            return contents.getvalue()
        else:
            return None


def get_case_parameters_str(case_name, **kwargs):
    # global remote_data, data_dir, data_host, remote_server_auto, paraview_cmd
    _remote_dir = analysis.data.data_dir
    if "data_dir" in kwargs:
        _remote_dir = kwargs["data_dir"]
    _remote_host = analysis.data.data_host
    if "data_host" in kwargs:
        _remote_host = kwargs["data_host"]

    _remote_data = analysis.data.remote_data
    if "remote_data" in kwargs:
        _remote_data = kwargs["remote_data"]

    if _remote_data:
        env.use_ssh_config = True
        env.host_string = _remote_host
        case_file_str = cat_case_file(_remote_dir, case_name)
        return case_file_str
    else:
        try:
            # Get contents of local file
            with open(_remote_dir + "/" + case_name + ".py") as f:
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


def get_case_parameters(case_name, **kwargs):
    case_file_str = get_case_parameters_str(case_name, **kwargs)
    namespace = {}
    exec(case_file_str, namespace)
    return namespace["parameters"]


def get_status_dict(case_name, **kwargs):
    # global remote_data, data_dir, data_host, remote_server_auto, paraview_cmd

    _remote_data = analysis.data.remote_data
    if "remote_data" in kwargs:
        _remote_data = kwargs["remote_data"]

    _remote_dir = analysis.data.data_dir
    if "data_dir" in kwargs:
        _remote_dir = kwargs["data_dir"]

    if _remote_data:
        _remote_host = analysis.data.data_host
        if "data_host" in kwargs:
            _remote_host = kwargs["data_host"]

        env.use_ssh_config = True
        env.host_string = _remote_host
        status_file_str = cat_status_file(_remote_dir, case_name)

        if status_file_str is not None:
            # print status_file_str
            return json.loads(status_file_str)
        else:
            print("WARNING: " + case_name + "_status.txt file not found")
            return None
    else:
        try:
            # Get contents of local file
            with open(_remote_dir + "/" + case_name + "_status.txt") as f:
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


def get_num_procs(case_name, **kwargs):
    # remote_host,remote_dir,case_name):
    status = get_status_dict(case_name, **kwargs)
    if status is not None:
        if "num processor" in status:
            return status["num processor"]
        else:
            return None
    else:
        print("status file not found")


def get_case_root(case_name, num_procs=None):
    if num_procs is None:
        num_procs = get_num_procs(case_name)
    return case_name + "_P" + str(num_procs) + "_OUTPUT/" + case_name


def get_case_report(case):
    return case + "_report.csv"


def print_html_parameters(parameters):

    reference = parameters["reference"]
    # material = parameters['material']
    conditions = parameters[reference]

    mach = 0.0
    speed = 0.0

    if "Mach" in conditions["V"]:
        mach = conditions["V"]["Mach"]
        speed = 0.0
    else:
        speed = mag(conditions["V"]["vector"])
        mach = 0.0

    if "Reynolds No" in conditions:
        reynolds = conditions["Reynolds No"]
    else:
        reynolds = "undefined"

    if "Reference Length" in conditions:
        reflength = conditions["Reference Length"]
    else:
        reflength = "undefined"

    import string

    html_template = """<table>
<tr><td>pressure</td><td>$pressure</td></tr>
<tr><td>temperature</td><td>$temperature</td></tr>
<tr><td>Reynolds No</td><td>$reynolds</td></tr>
<tr><td>Ref length</td><td>$reflength</td></tr>
<tr><td>Speed</td><td>$speed</td></tr>
<tr><td>Mach No</td><td>$mach</td></tr>
</table>"""
    html_output = string.Template(html_template)

    return html_output.substitute(
        {
            "pressure": conditions["pressure"],
            "temperature": conditions["temperature"],
            "reynolds": reynolds,
            "reflength": reflength,
            "speed": speed,
            "mach": mach,
        }
    )


class ProgressBar(object):
    def __init__(self):
        self.pbar = tqdm(total=100)

    def __iadd__(self, v):
        self.pbar.update(v)
        return self

    def complete(self):
        self.pbar.close()

    def update(self, i):
        self.pbar.update(i)


# remote_data = True
# data_host = 'user@server'
# data_dir = 'data'
# remote_server_auto = True
# paraview_cmd = 'mpiexec pvserver'
# paraview_home = '/usr/local/bin/'
# job_queue = 'default'
# job_tasks = 1
# job_ntaskpernode = 1
# job_project = 'default'


def data_location_form_html(**kwargs):
    global remote_data, data_dir, data_host, remote_server_auto, paraview_cmd
    global job_queue, job_tasks, job_ntaskpernode, job_project

    if "data_dir" in kwargs:
        data_dir = kwargs["data_dir"]
    if "paraview_cmd" in kwargs:
        paraview_cmd = kwargs["paraview_cmd"]
    if "data_host" in kwargs:
        data_host = kwargs["data_host"]

    remote_data_checked = ""
    if remote_data:
        remote_data_checked = 'checked="checked"'
    remote_server_auto_checked = ""
    if remote_server_auto:
        remote_server_auto_checked = 'checked="checked"'

    remote_cluster_checked = ""
    job_queue = "default"
    job_tasks = 1
    job_ntaskpernode = 1
    job_project = "default"

    input_form = """
<div style="background-color:gainsboro; border:solid black; width:640px; padding:20px;">
<label style="width:22%;display:inline-block">Remote Data</label>
<input type="checkbox" id="remote_data" value="remote" {remote_data_checked}><br>
<label style="width:22%;display:inline-block">Data Directory</label>
<input style="width:75%;" type="text" id="data_dir" value="{data_dir}"><br>
<label style="width:22%;display:inline-block">Data Host</label>
<input style="width:75%;" type="text" id="data_host" value="{data_host}"><br>
<label style="width:22%;display:inline-block">Remote Server Auto</label>
<input type="checkbox" id="remote_server_auto" value="remote_auto" {remote_server_auto_checked}><br>
<label style="width:22%;display:inline-block">Paraview Cmd </label>
<input style="width:75%;" type="text" id="paraview_cmd" value="{paraview_cmd}"><br>
<label style="width:22%;display:inline-block">Remote Cluster</label>
<input type="checkbox" id="remote_cluster" value="remote_cluster" {remote_cluster_checked}><br>
<label style="width:22%;display:inline-block">Job Queue </label>
<input style="width:75%;" type="text" id="job_queue" value="{job_queue}"><br>
<label style="width:22%;display:inline-block">Job Tasks </label>
<input style="width:75%;" type="text" id="job_tasks" value="{job_tasks}"><br>
<label style="width:22%;display:inline-block">Job Tasks per Node </label>
<input style="width:75%;" type="text" id="job_ntaskpernode" value="{job_ntaskpernode}"><br>
<label style="width:22%;display:inline-block">Job Project </label>
<input style="width:75%;" type="text" id="job_project" value="{job_project}"><br>
<button onclick="apply()">Apply</button>
</div>
"""

    javascript = """
    <script type="text/Javascript">
        function apply(){
            var remote_data = ($('input#remote_data').is(':checked') ? 'True' : 'False');
            var data_dir = $('input#data_dir').val();
            var data_host = $('input#data_host').val();
            var remote_server_auto = ($('input#remote_server_auto').is(':checked') ? 'True' : 'False');
            var paraview_cmd = $('input#paraview_cmd').val();
            var remote_cluster = ($('input#remote_cluster').is(':checked') ? 'True' : 'False');


            var kernel = IPython.notebook.kernel;

            // Send data dir to ipython
            var  command = "from zutil import post; post.data_dir = '" + data_dir + "'";
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Send data host to ipython
            var  command = "from zutil import post; post.data_host = '" + data_host + "'";
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Send remote server flag to ipython
            var  command = "from zutil import post; post.remote_server_auto = " + remote_server_auto;
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Send paraview command to ipython
            var  command = "from zutil import post; post.paraview_cmd = '" + paraview_cmd + "'";
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Send remote data flag to ipython
            var  command = "from zutil import post; post.remote_data = " + remote_data ;
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Set paraview command to none if not using remote server
            var command = "from zutil import post; if not post.remote_server_auto: post.paraview_cmd=None"
            console.log("Executing Command: " + command);
            kernel.execute(command);

            // Set data to local host for local data
            var command = "from zutil import post; if not post.post.remote_data: post.data_host='localhost'; post.paraview_cmd=None"
            console.log("Executing Command: " + command);
            kernel.execute(command);

            if(remote_cluster == 'True'){
                // Set cluster job info
                //var command = "from zutil import post; post.jo";
            }

        }
    </script>
    """

    return HTML(
        input_form.format(
            data_dir=data_dir,
            data_host=data_host,
            paraview_cmd=paraview_cmd,
            remote_data_checked=remote_data_checked,
            remote_server_auto_checked=remote_server_auto_checked,
            remote_cluster_checked=remote_cluster_checked,
            job_queue=job_queue,
            job_tasks=job_tasks,
            job_ntaskpernode=job_ntaskpernode,
            job_project=job_project,
        )
        + javascript
    )
