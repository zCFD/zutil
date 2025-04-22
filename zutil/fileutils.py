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

File I/O utilities
"""

import importlib
import os
from pathlib import Path
import json
import glob
import pandas as pd
from typing import Optional, Union
import re


class zCFD_Report(object):
    """zCFD Base Report object- will handle reading and data IO of report file"""

    def __init__(self, report_file: Optional[str] = None) -> None:
        self.header_list = None

        if report_file:
            if Path(report_file).exists():
                self.read_data(report_file)

    def read_data(self, report_file: str) -> None:
        """Read in csv data from zCFD report file"""
        self.data = get_csv_data(report_file, header=True).dropna(axis=1, how="all")

        # Check for restart by
        restart_file = report_file.rsplit(".csv", 1)[0] + ".restart.csv"
        if os.path.isfile(restart_file):
            self.restart_data = get_csv_data(restart_file, header=True).dropna(
                axis=1, how="all"
            )
            # Get first entry in new data
            restart_cycle = self.data["Cycle"].iloc[0]
            self.restart_data = self.restart_data[
                self.restart_data["Cycle"] < restart_cycle
            ]
            # Merge restart data with data
            self.data = pd.concat([self.restart_data, self.data], ignore_index=True)

        self.header_list: list[str] = list(self.data)
        self.residual_list = []
        for h in self.header_list:
            if h.startswith("rho") or h == "p" or h.startswith("turbEqn"):
                append_str = ""
                # if append_index > 0:
                #    append_str = '_'+str(append_index)
                self.residual_list.append(h + append_str)


class zCFD_Result:
    """helper class for zCFD result files"""

    def __init__(
        self,
        control_file_path: Optional[str] = None,
        verbose: bool = False,
        absolute=True,
    ) -> None:
        # paths
        self._control_file_path = Path()
        self._report_file_path = Path()
        self._output_directory_path = Path()
        self._volume_file_path = Path()
        self._status_file_path = Path()
        self._control_file_stem = Path()
        self._data_dir = Path()
        self._log_file_path = Path()
        self._checkpoint_path = Path()
        self._mesh_path = Path()

        # boundary file paths- resolve to strings
        self._wall_boundary_path = Path()
        self._symmetry_boundary_path = Path()
        self._farfield_boundary_path = Path()
        self._periodic_boundary_path = Path()
        self._immersed_boundary_path = Path()
        self._inflow_boundary_path = Path()
        self._outflow_boundary_path = Path()
        self._overset_boundary_path = Path()
        self._fwh_wall_data = Path()

        # strings
        # paths
        self.control_file_path: str
        self.report_file_path: str
        self.output_directory_path: str
        self.volume_file_path: str
        self.status_file_path: str
        self.control_file_stem: str
        self.data_dir: str
        self.log_file_path: str
        self.checkpoint_path: str
        self.mesh_path: str

        # boundary file paths- resolve to strings
        self.wall_boundary_path: str
        self.symmetry_boundary_path: str
        self.farfield_boundary_path: str
        self.periodic_boundary_path: str
        self.immersed_boundary_path: str
        self.inflow_boundary_path: str
        self.outflow_boundary_path: str
        self.overset_boundary_path: str
        self.fwh_wall_data: str

        # bools
        self.zcfd_started: bool
        self.overset: bool = False
        self.zcfd_completed: bool
        self.verbose: bool = verbose
        self.steady: bool
        self.absolute = True
        self.rotating = False
        self.translating = False

        # ints
        self.num_procs: int

        # lists
        self.boundary_files: list
        self.immersed_wall_paths: list = []
        self.fwh_permeable_paths: list = []
        self.rank_report_paths: list = []

        # data objects
        self.report: zCFD_Report
        self.parameters: dict
        self.status: dict

        # wizardary to make outside references to path variables resolve to strings

        self._all_paths = [
            attr.lstrip("_")
            for attr, value in vars(self).items()
            if isinstance(value, Path)
        ]

        for path in self._all_paths:
            setattr(
                self.__class__,
                path,
                property(lambda self, p=path: str(getattr(self, f"_{p}"))),
            )

        if control_file_path:
            # check if .py file extension has been removed or not
            self.read_control_file(control_file_path)

    def read_control_file(self, control_file_path_str):
        control_file_path, valid = check_valid_control_file(control_file_path_str)

        if self.absolute:
            control_file_path = control_file_path.absolute()

        if valid:
            self._control_file_path = control_file_path
            self._control_file_stem = Path(control_file_path).stem
            self._data_dir = self._control_file_path.parent

            self.parameters = get_parameters_from_file(control_file_path)

            self._check_steady_state()
            self._check_rotating()
            self._check_translating()

            self._status_file_path = Path(
                self._data_dir, self._control_file_stem + "_status.txt"
            )

            self.zcfd_started = self._check_status()

            if self.zcfd_started:
                self._init_paths()

    def _check_steady_state(self):
        if self.parameters.get("time marching").get("unsteady").get(
            "total time"
        ) == self.parameters.get("time marching").get("unsteady").get("time step"):
            self.steady = True
        else:
            self.steady = False

    def _check_rotating(self):
        """sets true if the fluid domain is rotating"""
        fz_keys = [key for key in self.parameters.keys() if "FZ_" in key]
        if len(fz_keys) == 0:
            self.rotating = False
        else:
            for fz in fz_keys:
                if (self.parameters[fz]["type"] == "rotating") and (
                    0 in self.parameters[fz]["zone"]
                ):
                    self.rotating = True
                    return

    def _check_translating(self):
        """sets true if the fluid domain is translating"""
        fz_keys = [key for key in self.parameters.keys() if "FZ_" in key]
        if len(fz_keys) == 0:
            self.translating = False
        else:
            for fz in fz_keys:
                if (self.parameters[fz]["type"] == "translating") and (
                    0 in self.parameters[fz]["zone"]
                ):
                    self.translating = True
                    return

    def _init_paths(self):
        """initialise paths to the various zCFD files"""
        self._print("initialising paths")

        self._report_file_path = Path(
            self._data_dir, self._control_file_stem + "_report.csv"
        )
        self.report = zCFD_Report(str(self._report_file_path))

        self._status_file_path = Path(
            self._data_dir, self._control_file_stem + "_status.txt"
        )
        self._checkpoint_path = Path(
            self._data_dir, self._control_file_stem + "_results.h5"
        )

        self._mesh_path = Path(self.status["mesh"] + ".h5")

        self.num_procs = self.status["num processor"]

        self._output_directory_path = Path(
            self._data_dir,
            self._control_file_stem + "_P{}_OUTPUT/".format(self.num_procs),
        )

        self._volume_file_path = Path(
            self._output_directory_path, self._control_file_stem + ".pvd"
        )

        self._log_file_path = Path(self._data_dir, self._control_file_stem + ".log")
        self._get_rank_log_paths()

        self._init_boundary_paths()

    def _get_rank_log_paths(self):
        """get paths to rank log files"""
        self._print("getting rank log paths")
        rank_log_dir = Path(self._output_directory_path, "LOGGING")
        for r in range(self.num_procs):
            self.rank_report_paths.append(
                Path(rank_log_dir, self._control_file_stem + "." + str(r) + ".log")
            )

    def _init_boundary_paths(self):
        self._print("getting boundary files")
        self.boundary_files = list(
            self._output_directory_path.glob(self._control_file_stem + "_*.pvd")
        )

        for boundary_file in self.boundary_files:
            # big ugly case switch depending on file extension
            if boundary_file.match("*_symmetry.pvd"):
                self._symmetry_boundary_path = boundary_file
            elif boundary_file.match("*_wall.pvd"):
                self._wall_boundary_path = boundary_file
            elif boundary_file.match("*_immersed wall.pvd"):
                self._immersed_boundary_path = boundary_file
            elif boundary_file.match("*_farfield.pvd"):
                self._farfield_boundary_path = boundary_file
            elif boundary_file.match("*_periodic.pvd"):
                self._periodic_boundary_path = boundary_file
            elif boundary_file.match("*_overset.pvd"):
                self._overset_path = boundary_file
            elif boundary_file.match("*_inflow.pvd"):
                self._inflow_boundary_path = boundary_file
            elif boundary_file.match("*_outflow.pvd"):
                self._outflow_boundary_path = boundary_file
            else:
                print(
                    "error, unrecognised boundary file type: {}".format(boundary_file)
                )

        # if immersed, check for mapped solutions
        if self._immersed_boundary_path:
            immersed_list = self.parameters.get("write output").get(
                "immersed boundary surface"
            )
            if immersed_list:
                for immersed_zone in immersed_list:
                    stl_name = Path(immersed_zone["stl"]).stem
                    # perform regex pattern match of output directory to find all unsteady immersed wall files
                    immersed_files = glob.glob(
                        str(self.output_directory_path) + "/" + stl_name + "_[0-9]*.vtu"
                    )
                    # sort the files by cycle number (glob will put 11 before 9 etc...)
                    immersed_files.sort(key=self._sort_unsteady_key)

                    self.immersed_wall_paths.append(immersed_files)

        # set FWH wall data path
        if self.parameters.get("write output").get("fwh wall data"):
            self._fwh_wall_data = (
                self._output_directory_path
                / "ACOUSTIC_DATA"
                / (str(self._control_file_stem) + "_wall_FWHData.h5")
            )

        # check for permeable FWH surfaces
        if self.parameters.get("write output").get("fwh interpolate"):
            fwh_interpolate_list = self.parameters.get("write output").get(
                "fwh interpolate"
            )
            for fwh_interpolate in fwh_interpolate_list:
                stl_name = Path(fwh_interpolate).stem
                fwh_name = stl_name + "_FWHData.h5"
                self.fwh_permeable_paths.append(
                    self._output_directory_path / "ACOUSTIC_DATA" / fwh_name
                )

    def _check_status(self):
        # Get the status dictionary - this bit is copied from zutil/zutil/post/post.py on Github
        if self._status_file_path.exists():
            with open(self._status_file_path, "r") as f:
                self.status = json.load(f)

            if "problem" in self.status.keys():
                self.status["mesh"] = self.status["problem"]

            return True
        else:
            self._print(
                "Status file path ( {}) does not exist".format(self.status_file_path)
            )
            return False

    def _print(self, message):
        if self.verbose:
            print(message)

    @staticmethod
    def _sort_unsteady_key(filename):
        # Extracts the integer (cycle number) from the filename
        match = re.search(r"_(\d+)\.vtu$", filename)
        if match:
            return int(match.group(1))  # Return the integer for sorting
        return 0  # Default return value if the pattern is not found


class zCFD_Overset_Result(object):
    """Class with helper functions for overset result paths"""

    def __init__(
        self,
        override_file_path: Optional[str] = None,
        verbose: bool = False,
        absolute=True,
    ) -> None:
        # paths
        self._override_file_path = Path()
        self._override_file_stem = Path()
        self._log_file_path = Path()
        self._data_dir = Path()

        # strings
        # paths
        self.override_file_path: str
        self.override_file_stem: str
        self.log_file_path: str
        self.data_dir: str

        # bools
        self.verbose: bool = verbose
        self.overset = True
        self.zcfd_started = False

        # ints

        # lists
        self.overset_cases: list[zCFD_Result]

        # data objects
        self.override: dict

        # wizardary to make outside references to path variables resolve to strings

        self._all_paths = [
            attr.lstrip("_")
            for attr, value in vars(self).items()
            if isinstance(value, Path)
        ]

        for path in self._all_paths:
            setattr(
                self.__class__,
                path,
                property(lambda self, p=path: str(getattr(self, f"_{p}"))),
            )

        if override_file_path:
            # check if .py file extension has been removed or not
            self.read_override_file(override_file_path)

    def read_override_file(self, override_file_path) -> None:
        override_file_path, valid = check_valid_control_file(override_file_path)
        if valid:
            self._override_file_path = override_file_path
            self._override_file_stem = Path(override_file_path).stem
            self._data_dir = self._override_file_path.parent

            self.override = get_parameters_from_file(
                override_file_path, parameters_key="override"
            )

            self._log_file_path = Path(self.data_dir, self._override_file_stem + ".log")

            self._init_overset_cases()

    def _init_overset_cases(self) -> None:
        """initialise paths to the various zCFD files"""
        self._print("initialising paths")

        self.overset_cases = []
        for pair in self.override["mesh_case_pair"]:
            control_path = Path(pair[1])
            if control_path.is_absolute():
                overset_case = control_path
            else:
                overset_case = Path(self._data_dir, control_path)

            self.overset_cases.append(
                zCFD_Result(str(overset_case), verbose=self.verbose)
            )
            # recursively set overset flag to True
            self.overset_cases[-1].overset = True

        # get zcfd_started from background case
        self.zcfd_started = self.overset_cases[0].zcfd_started

    def _print(self, message):
        if self.verbose:
            print(message)


def get_zcfd_result(
    control_file_path: str, verbose: bool = False, absolute=True
) -> Union[zCFD_Result, zCFD_Overset_Result]:
    """function to read the supplied parameters file, identify if it is an overset case and return the appropriate object"""
    control_file_path, valid = check_valid_control_file(control_file_path)

    # cache cwd
    cwd = os.getcwd()

    # move to file directory for relative imports
    file_dir = Path(control_file_path).parent
    os.chdir(file_dir)
    control_file_path = control_file_path.name

    if valid:
        # check if overset
        namespace = _get_namespace_from_file(control_file_path)
        if namespace.get("override"):
            result = zCFD_Overset_Result(
                control_file_path, verbose=verbose, absolute=absolute
            )
            os.chdir(cwd)
            return result
        elif namespace.get("parameters"):
            result = zCFD_Result(control_file_path, verbose=verbose, absolute=absolute)
            os.chdir(cwd)
            return result
        else:
            raise ValueError(
                "Control file {} does not contain a 'parameters' or 'override' dictionary".format(
                    control_file_path
                )
            )
    else:
        raise FileNotFoundError(
            "Control file {} could not be found".format(control_file_path)
        )


def get_parameters_from_file(
    control_file_path: Path,
    parameters_key: str = "parameters",
    namespace: Optional[dict] = None,
    working_dir: str = ".",
) -> dict:
    """
    Loads and executes a Python module file, extracting the 'parameters' attribute
    as a dictionary.

    Args:
        control_file_path (Path or str): Path to the Python file containing the
                                         `parameters` dictionary.
    Optional Args:
        parameters_key (str): The key to extract from the imported module.
        namespace (dict): The namespace to execute the file in- used to pass variables to the control file.

    Returns:
        dict: The 'parameters' dictionary from the imported module.

    Raises:
        ValueError: If the provided file path does not exist.
        KeyError: If the 'parameters' attribute is not found in the file.
    """
    namespace = namespace or {}

    namespace = _get_namespace_from_file(control_file_path, namespace)

    # check if case is an overset control file
    # Retrieve and return the 'parameters' dictionary from the namespace
    try:
        control_dict = namespace[parameters_key]
    except KeyError:
        raise KeyError(
            "The file does not contain a '{}' attribute".format(parameters_key)
        )

    return control_dict


def _get_namespace_from_file(
    control_file_path: Path, namespace: Optional[dict] = None
) -> dict:
    """
    Loads and executes a Python module file, returning the resulting namespace.

    Args:
        control_file_path (Path): Path to the Python file to be executed"""

    # Get the current __file__ attribute from the namespace - needed to provide __file__ to the control file
    if namespace:
        old_file = namespace.get("__file__")
    else:
        namespace = {}
        old_file = None

    # Convert string input to Path and ensure file has a .py extension
    if isinstance(control_file_path, str):
        if not control_file_path.endswith(".py"):
            control_file_path += ".py"
        control_file_path = Path(control_file_path)
    namespace["__file__"] = str(control_file_path)
    # Check if the file exists before proceeding
    if not control_file_path.exists():
        raise ValueError(
            "Control file path ({}) does not exist".format(control_file_path)
        )

    # Read the file and execute its content in a temporary namespace
    with open(control_file_path, "r") as f:
        control_string = f.read()
    exec(control_string, namespace)

    # reset __file__ attribute to its original value
    if old_file:
        namespace["__file__"] = old_file
    return namespace


def include(filename: str, _globals: Optional[dict] = None) -> None:
    """
    include a file by executing it. This imports everything including
    variables into the calling module
    """
    _globals = _globals or {}
    if os.path.exists(filename):
        exec(compile(open(filename, "rb").read(), filename, "exec"), _globals)


def clean_name(name: str) -> str:
    """Returns the blank name for a zcfd case- removes .py extension if used"""
    return os.path.splitext(name)[0]


def get_zone_info(zone_dict_name: str) -> dict:
    """
    Loads a zone control dictionary by importing it as a Python module.

    Args:
        zone_dict_name (str): The name of the module containing the zone dictionary.

    Returns:
        dict: The loaded zone control dictionary.

    Raises:
        ImportError: If the module cannot be imported or does not contain a dictionary.
        ValueError: If the imported module does not contain a dictionary.
    """

    try:
        zone_module = importlib.import_module(zone_dict_name)
        return zone_module
    except ImportError as e:
        raise ImportError(
            f"Failed to import zone control dictionary from '{zone_dict_name}': {e}"
        ) from e


def check_valid_control_file(control_file_path_str: str) -> tuple[Path, bool]:
    """Util to check the validity of a file path supplied as a zCFD control file"""
    control_file_path = Path(control_file_path_str)
    # check file extension
    if not control_file_path.suffix == ".py":
        # try appending .py first
        control_file_path = control_file_path.with_suffix(".py")
    if not control_file_path.exists():
        print(
            "Error: zCFD control file {} could not be found".format(control_file_path)
        )
        if not control_file_path.is_absolute():
            print("Relative path supplied, cwd is: {}".format(os.getcwd()))
        return control_file_path, False
    return control_file_path, True


def get_csv_data(
    filename: str, header: bool = False, remote: bool = False, delim: str = " "
) -> pd.DataFrame:
    if not header:
        data = pd.read_csv(filename, sep=delim, header=None)
    else:
        data = pd.read_csv(filename, sep=delim)
    return data


def get_fw_csv_data(
    filename: str, widths: int, header: bool = False, remote: bool = False, **kwargs
) -> any:
    """Reads csv and returns specified rows"""
    if not header:
        data = pd.read_fwf(filename, sep=" ", header=None, widths=widths, **kwargs)
    else:
        data = pd.read_fwf(filename, sep=" ", width=widths, **kwargs)
    return data
