"""
Copyright (c) 2012-2026, Zenotech Ltd
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
import pandas as pd
from typing import Optional, Union
import re
from importlib.resources import files
import numpy as np

from zutil.deck_dependencies import (
    record_executed_file,
    record_imported_siblings,
)


def report_stems(report_block: dict, key: str) -> dict:
    """Report-column stem of every entry under ``report[key]``, keyed by its label.

    The solver writes an entry's columns under its ``name`` when the deck
    sets one and under its label, lower case with spaces as underscores,
    otherwise (``Report.h``).
    """
    from zutil.validate.validation_utils import normalise_key

    block = report_block.get(key)
    if not isinstance(block, dict):
        return {}
    return {
        label: (entry.get("name") or normalise_key(label))
        for label, entry in block.items()
        if isinstance(entry, dict)
    }


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
            if (
                not self.data.empty
                and "Cycle" in self.data.columns
                and "Cycle" in self.restart_data.columns
            ):
                # Get first entry in new data
                restart_cycle = self.data["Cycle"].iloc[0]
                self.restart_data = self.restart_data[
                    self.restart_data["Cycle"] < restart_cycle
                ]
                # Merge restart data with data
                self.data = pd.concat([self.restart_data, self.data], ignore_index=True)
            elif self.data.empty:
                self.data = self.restart_data

        self.header_list: list[str] = list(self.data)
        self.residual_list = []
        for h in self.header_list:
            if h.startswith("rho") or h == "p" or h.startswith("turbEqn"):
                append_str = ""
                # if append_index > 0:
                #    append_str = '_'+str(append_index)
                self.residual_list.append(h + append_str)

    def get_array(self, array_name: str, DTS_filter: bool = True) -> list[float]:
        """Get an array from a zCFD report file

        Args:
            array_name (str): Name of the array to retrieve
            DTS_filter (bool = True): If True, return only the last value per psuedo time step
        """

        # check if array exists
        if array_name not in self.header_list:
            raise ValueError(
                "Array {} not found in report file {}".format(
                    array_name, self.header_list
                )
            )
        else:
            # get the array from the report data
            array = self.data[array_name].to_numpy()

            if DTS_filter:
                real_time_step = self.data["RealTimeStep"].to_numpy()
                indices = np.where(real_time_step[1:] != real_time_step[:-1])[0]
                indices = np.append(indices, len(real_time_step) - 1)
                filtered_array = array[indices]
                return filtered_array.tolist()
            else:
                return array.tolist()
        raise Exception(
            "An unknown error occurred while retrieving array {}".format(array_name)
        )


class zCFD_Result_Base(object):
    """Base class for zCFD result files"""

    def __init__(self) -> None:
        self.verbose: bool = False
        self.log_file_path: str
        self.report: zCFD_Report

    def _print(self, message):
        if self.verbose:
            print(message)

    def get_average_solve_time(self) -> float:
        """Get average solve time for all reports"""
        return _get_average_time_from_file(self.log_file_path, "solving")

    def get_average_report_time(self) -> float:
        """Get average report time for all reports"""
        return _get_average_time_from_file(self.log_file_path, "reporting")


class zCFD_Result(zCFD_Result_Base):
    """helper class for zCFD result files

    ``parameters`` holds the control-file dict exactly as the deck wrote it
    (typically space-separated key names). ``normalized_parameters`` holds the
    solver-facing view (spaces -> underscores, schema keys lowercased) — the
    same normalisation the launcher applies before the C++ sees the dict.
    """

    def __init__(
        self,
        control_file_path: Optional[str] = None,
        verbose: bool = False,
        absolute=True,
        model: Optional[str] = None,
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
        self._inflow_boundary_path = Path()
        self._outflow_boundary_path = Path()
        self._overset_boundary_path = Path()
        self._fwh_wall_data = Path()
        self._acoustic_data_path = Path()

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
        self.fwh_permeable_paths: list = []
        self.rank_report_paths: list = []

        # data objects
        self.report: zCFD_Report
        self.parameters: dict
        self.normalized_parameters: dict
        self.status: dict

        # Per-model scoping: if set, default paths point to this model
        self._target_model: Optional[str] = model

        # Per-model data (populated by _init_paths when model subdirs detected)
        self.model_names: list[str] = []
        self.model_dirs: dict[str, Path] = {}
        self.model_reports: dict[str, zCFD_Report] = {}
        self.model_volume_paths: dict[str, Path] = {}
        self.model_checkpoint_paths: dict[str, Path] = {}
        self.model_boundary_files: dict[str, list[Path]] = {}

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

            # Local import: zutil.validate pulls in pydantic, which not every
            # fileutils consumer needs at import time.
            from zutil.validate.validation_utils import replace_spaces_in_keys
            from zutil.validate.zcfd_schema import resolve_mesh_settings

            self.normalized_parameters = resolve_mesh_settings(
                replace_spaces_in_keys(self.parameters)
            )

            self._check_steady_state()
            kinds = self._motion_kinds()
            self.rotating = "rotating" in kinds
            self.translating = "translating" in kinds

            self._status_file_path = Path(
                self._data_dir, self._control_file_stem + "_status.yaml"
            )

            self.zcfd_started = self._check_status()

            if self.zcfd_started:
                self._init_paths()

    def _check_steady_state(self):
        if self.normalized_parameters.get("config_version") != 2:
            raise ValueError(
                "Legacy configuration detected (config_version != 2). "
                "Please update to config_version: 2."
            )
        time_settings = self.normalized_parameters["solver"]["time_settings"]
        self.steady = time_settings.get("type") == "steady"

    def _motion_kinds(self) -> set:
        model_block = self.normalized_parameters.get("model", {})
        if self._target_model and self._target_model in model_block:
            model_block = {self._target_model: model_block[self._target_model]}
        kinds = set()
        for model_dict in model_block.values():
            if not isinstance(model_dict, dict):
                continue
            motion = (model_dict.get("transforms") or {}).get("motion")
            if isinstance(motion, dict) and motion.get("type") in (
                "rotating",
                "translating",
            ):
                kinds.add(motion["type"])
        return kinds

    def _init_paths(self):
        """initialise paths to the various zCFD files"""
        self._print("initialising paths")

        self._status_file_path = Path(
            self._data_dir, self._control_file_stem + "_status.yaml"
        )

        # Extract mesh path from the model block (first model)
        model_block = self.normalized_parameters.get("model", {})
        mesh_name = ""
        for v in model_block.values():
            if isinstance(v, dict) and "mesh" in v:
                mesh_name = v["mesh"]
                break
        if not mesh_name.endswith(".h5"):
            mesh_name += ".h5"
        self._mesh_path = Path(mesh_name)

        self.num_procs = self.status["num processor"]

        self._output_directory_path = Path(
            self._data_dir,
            self._control_file_stem + "_OUTPUT/",
        )

        # Detect per-model output directories.
        # Layout: model subdirectories inside <case>_OUTPUT/
        # e.g., <case>_OUTPUT/background/, <case>_OUTPUT/cylinder_1/
        self.model_names = []
        self.model_dirs = {}
        self.model_reports = {}
        self.model_volume_paths = {}
        self.model_checkpoint_paths = {}
        self.model_boundary_files = {}

        if self._output_directory_path.exists():
            # The models are declared by the case: model_ordering from the
            # status YAML (DAG order set by the solver), falling back to
            # control file dict order.  A model is present if the solver
            # created its output subdirectory.
            dag_order = self.status.get("model_ordering", [])
            if dag_order:
                candidate_names = dag_order
            else:
                candidate_names = list(self.normalized_parameters.get("model", {}))

            for name in candidate_names:
                model_dir = self._output_directory_path / name
                if model_dir.is_dir():
                    self.model_names.append(name)
                    self.model_dirs[name] = model_dir

        # Set overset flag if multiple models detected
        if len(self.model_names) > 1:
            self.overset = True

        # Populate per-model data
        # Files within model dirs are named with the model name.
        for mname in self.model_names:
            mdir = self.model_dirs[mname]
            report_path = Path(mdir, mname + "_report.csv")
            self.model_reports[mname] = zCFD_Report(str(report_path))
            self.model_volume_paths[mname] = Path(mdir, mname + ".vtkhdf")
            self.model_checkpoint_paths[mname] = Path(mdir, mname + "_results.h5")
            self.model_boundary_files[mname] = list(mdir.glob(mname + "_*.vtkhdf"))

        # Pick the model for default paths
        target_model = self._target_model
        if target_model and target_model not in self.model_dirs:
            self._print(f"Warning: target model '{target_model}' not found in output")
            target_model = None

        if self.model_names:
            primary = target_model or self.model_names[0]
            primary_dir = self.model_dirs[primary]

            self._report_file_path = Path(primary_dir, primary + "_report.csv")
            self._checkpoint_path = Path(primary_dir, primary + "_results.h5")
            self._volume_file_path = Path(primary_dir, primary + ".vtkhdf")
        else:
            # Single-model case: paths are directly in the output directory
            self._report_file_path = Path(
                self._data_dir, self._control_file_stem + "_report.csv"
            )
            self._checkpoint_path = Path(
                self._data_dir, self._control_file_stem + "_results.h5"
            )
            self._volume_file_path = Path(
                self._output_directory_path, self._control_file_stem + ".vtkhdf"
            )

        self.report = zCFD_Report(str(self._report_file_path))

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

        # Determine where to search and what file stub to use
        if self.model_names:
            target = (
                self._target_model
                if self._target_model in self.model_dirs
                else self.model_names[0]
            )
            search_dir = self.model_dirs[target]
            file_stub = target
        else:
            search_dir = self._output_directory_path
            file_stub = str(self._control_file_stem)

        self.boundary_files = list(search_dir.glob(file_stub + "_*.vtkhdf"))

        for boundary_file in self.boundary_files:
            # big ugly case switch depending on file extension
            if boundary_file.match("*_symmetry.vtkhdf"):
                self._symmetry_boundary_path = boundary_file
            elif boundary_file.match("*_wall.vtkhdf"):
                self._wall_boundary_path = boundary_file
            elif boundary_file.match("*_farfield.vtkhdf"):
                self._farfield_boundary_path = boundary_file
            elif boundary_file.match("*_periodic.vtkhdf"):
                self._periodic_boundary_path = boundary_file
            elif boundary_file.match("*_overset.vtkhdf"):
                self._overset_path = boundary_file
            elif boundary_file.match("*_sliding.vtkhdf"):
                self._sliding_path = boundary_file
            elif boundary_file.match("*_inflow.vtkhdf"):
                self._inflow_boundary_path = boundary_file
            elif boundary_file.match("*_outflow.vtkhdf"):
                self._outflow_boundary_path = boundary_file
            else:
                print(
                    "error, unrecognised boundary file type: {}".format(boundary_file)
                )

        write_output = self.normalized_parameters["solver"]["output_settings"].get(
            "solution", {}
        )

        # set FWH wall data path
        has_fwh = False
        if write_output.get("fwh_wall_data"):
            has_fwh = True
            fwh_wall_stub = file_stub + "_wall_FWHData.h5"
            self._fwh_wall_data = search_dir / "ACOUSTIC_DATA" / fwh_wall_stub

        # check for permeable FWH surfaces
        fwh_interpolate_list = write_output.get("fwh_interpolate")
        if fwh_interpolate_list:
            has_fwh = True
            for fwh_interpolate in fwh_interpolate_list:
                stl_name = Path(fwh_interpolate).stem
                fwh_name = stl_name + "_FWHData.h5"
                self.fwh_permeable_paths.append(search_dir / "ACOUSTIC_DATA" / fwh_name)

        if has_fwh:
            self._acoustic_data_path = search_dir / "ACOUSTIC_DATA"

    def _check_status(self):
        if self._status_file_path.exists():
            import yaml

            with open(self._status_file_path, "r") as f:
                self.status = yaml.safe_load(f)

            if "problem" in self.status.keys():
                self.status["mesh"] = self.status["problem"]

            return True
        else:
            self._print(
                "Status file path ( {}) does not exist".format(self.status_file_path)
            )
            return False

    def force_prefixes(self) -> dict:
        """Report-column prefix of every force region, keyed by its dict key.

        A deck with ``{"FR_1": {"name": "wall", ...}}`` writes ``wall_Fx``
        while one without a name writes ``fr_1_Fx`` (``report_stems``).
        Post-processing must ask rather than assume: the same case can have
        either, depending on when it ran.
        """
        return report_stems(self._report_block(), "forces")

    def _report_block(self) -> dict:
        return (
            self.normalized_parameters.get("solver", {})
            .get("output_settings", {})
            .get("report", {})
        )

    def force_prefix(self, report: Optional[str] = None) -> str:
        """Report-column prefix of one force region (the only one by default).

        Resolved against the report CSV's own columns where they are readable:
        a control file edited after the run (adding a ``name``) would otherwise
        promise a prefix the existing data does not carry.
        """
        forces = self._report_block().get("forces", {})
        if not forces:
            raise AttributeError(
                f"No force regions in output_settings.report.forces for "
                f"{self.case_name}"
            )
        if report is None:
            key = next(iter(forces))
        elif report in forces:
            key = report
        else:
            raise AttributeError(
                f"No force region {report!r} in {self.case_name}: have {sorted(forces)}"
            )

        from zutil.validate.validation_utils import normalise_key

        declared = self.force_prefixes()[key]
        candidates = list(dict.fromkeys([declared, normalise_key(key)]))
        try:
            columns = list(self.report.data.columns)
        except Exception:
            return declared
        for candidate in candidates:
            if any(col.startswith(f"{candidate}_") for col in columns):
                return candidate
        return declared

    def monitor_prefixes(self) -> dict:
        """Report-column prefix of every monitor point, keyed by its dict key."""
        return report_stems(self._report_block(), "monitor")

    def get_reference_conditions(self) -> dict:
        """Extract reference conditions from config_version 2 parameters.

        Lookup order:
        1. ``solver.initialisation.reference`` → key into ``reference_conditions``
        2. ``solver.initialisation.initial_conditions`` → key into ``reference_conditions``
        3. First entry in ``solver.reference_conditions``
        """
        solver = self.normalized_parameters.get("solver", {})
        ref_conditions = solver.get("reference_conditions", {})
        init_block = solver.get("initialisation", {})

        # 1. Explicit reference key in initialisation block
        ref_name = init_block.get("reference", "")
        if ref_name and ref_name in ref_conditions:
            return ref_conditions[ref_name]

        # 2. Fall back to initial_conditions key
        ic_name = init_block.get("initial_conditions", "")
        if ic_name and ic_name in ref_conditions:
            return ref_conditions[ic_name]

        raise AttributeError(
            f"No reference conditions found: "
            f"initialisation.reference='{ref_name}', "
            f"initialisation.initial_conditions='{ic_name}', "
            f"reference_conditions keys={list(ref_conditions.keys())}"
        )

    @staticmethod
    def _sort_unsteady_key(filename):
        # Extracts the integer (cycle number) from the filename
        match = re.search(r"_(\d+)\.vtu$", filename)
        if match:
            return int(match.group(1))  # Return the integer for sorting
        return 0  # Default return value if the pattern is not found


def get_zcfd_result(
    control_file_path: str,
    verbose: bool = False,
    absolute=True,
    model: Optional[str] = None,
) -> zCFD_Result:
    """Read the supplied parameters file and return a zCFD_Result object.

    Multi-model (overset) cases are handled natively by zCFD_Result via
    the model_names / model_dirs / model_reports attributes.
    """
    control_file_path, valid = check_valid_control_file(control_file_path)

    # cache cwd
    cwd = os.getcwd()

    # move to file directory for relative imports
    file_dir = Path(control_file_path).parent
    os.chdir(file_dir)
    control_file_path = control_file_path.name

    if valid:
        result = zCFD_Result(
            control_file_path, verbose=verbose, absolute=absolute, model=model
        )
        os.chdir(cwd)
        return result
    else:
        os.chdir(cwd)
        raise FileNotFoundError(
            "Control file {} could not be found".format(control_file_path)
        )


def get_parameters_from_file(
    control_file_path: Path,
    parameters_key: Union[str, list[str]] = "parameters",
    namespace: Optional[dict] = None,
    working_dir: str = ".",
) -> Union[dict, tuple[dict, str]]:
    """
    Loads and executes a Python module file, extracting the 'parameters' attribute
    as a dictionary.

    Args:
        control_file_path (Path or str): Path to the Python file containing the
                                         `parameters` dictionary.
    Optional Args:
        parameters_key (str or list[str]): The key(s) to extract from the imported module.
                                           If a list is provided, returns the first found key's value and the key name.
        namespace (dict): The namespace to execute the file in- used to pass variables to the control file.

    Returns:
        dict: The 'parameters' dictionary from the imported module (if parameters_key is str).
        tuple: (dict, key) The dictionary and the key found (if parameters_key is list).

    Raises:
        ValueError: If the provided file path does not exist.
        KeyError: If the 'parameters' attribute is not found in the file.
    """
    import inspect

    namespace = namespace or {}

    # Convert string input to Path and handle relative paths
    if isinstance(control_file_path, str):
        if not control_file_path.endswith(".py"):
            control_file_path += ".py"
        control_file_path = Path(control_file_path)

    original_path = control_file_path

    # If the path is relative, resolve it relative to the calling file's directory
    if not control_file_path.is_absolute():
        caller_file = None

        # Try multiple approaches to find the calling file
        # 1. Check if namespace has __file__ (preferred for zCFD pattern)
        if namespace and "__file__" in namespace:
            caller_file = Path(namespace["__file__"])
        else:
            # 2. Inspect call stack to find the caller
            try:
                frame = inspect.currentframe()
                # Walk up the stack to find the first frame that's not in fileutils.py
                while frame:
                    frame = frame.f_back
                    if (
                        frame
                        and frame.f_code.co_filename
                        and "fileutils.py" not in frame.f_code.co_filename
                    ):
                        caller_file = Path(frame.f_code.co_filename)
                        break
            except:
                pass

        # 3. If still no caller file found and we have a namespace with globals,
        #    try to get __file__ from the frame's globals
        if caller_file is None:
            try:
                frame = inspect.currentframe()
                while frame:
                    frame = frame.f_back
                    if (
                        frame
                        and frame.f_code.co_filename
                        and "fileutils.py" not in frame.f_code.co_filename
                        and "__file__" in frame.f_globals
                    ):
                        caller_file = Path(frame.f_globals["__file__"])
                        break
            except:
                pass

        # If we found a caller file, resolve relative to its directory
        if caller_file and caller_file.exists():
            caller_dir = caller_file.parent
            control_file_path = caller_dir / control_file_path

    # Resolve the path to handle any .. or . components
    control_file_path = control_file_path.resolve()

    namespace = load_config_namespace(control_file_path, namespace)

    # check if case is an overset control file
    # Retrieve and return the 'parameters' dictionary from the namespace
    if isinstance(parameters_key, list):
        for key in parameters_key:
            if key in namespace:
                return namespace[key], key
        raise KeyError(
            "The file does not contain any of the following attributes: {}".format(
                parameters_key
            )
        )

    try:
        control_dict = namespace[parameters_key]
    except KeyError:
        raise KeyError(
            "The file does not contain a '{}' attribute".format(parameters_key)
        )

    return control_dict


def load_config_namespace(
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

    # Ensure control_file_path is a Path object (should already be resolved by caller)
    if isinstance(control_file_path, str):
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

    # Report this file to any active dependency tracker before executing it:
    # a deck's includes only become visible by running it, and this is the
    # one place a deck file is ever exec'd, so nested
    # get_parameters_from_file chains are captured for free. No-op unless a
    # caller opened a zutil.deck_dependencies.track_dependencies() block.
    record_executed_file(control_file_path)

    # Compile with filename to preserve stack frame info for relative imports
    code = compile(control_string, str(control_file_path), "exec")

    import sys

    script_dir = str(control_file_path.parent.absolute())
    sys_path_added = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        sys_path_added = True

    try:
        exec(code, namespace)
        # Helpers the deck pulled in by plain import (the reason script_dir
        # goes on sys.path at all) are part of what this read executed, but
        # the import machinery ran them, not us — collect them too.
        record_imported_siblings(script_dir)
    finally:
        if sys_path_added:
            sys.path.remove(script_dir)

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
    import inspect

    # Ensure the zone_dict_name has a .py extension for filename- ensures backwards compatibility
    if not zone_dict_name.endswith(".py"):
        zone_dict_name += ".py"

    zone_path = Path(zone_dict_name)

    # If the path is relative, resolve it relative to the calling file's directory
    if not zone_path.is_absolute():
        caller_file = None

        # Inspect call stack to find the caller
        try:
            frame = inspect.currentframe()
            # Walk up the stack to find the first frame that's not in fileutils.py
            while frame:
                frame = frame.f_back
                if (
                    frame
                    and frame.f_code.co_filename
                    and "fileutils.py" not in frame.f_code.co_filename
                ):
                    caller_file = Path(frame.f_code.co_filename)
                    break
        except:
            pass

        # If we found a caller file, resolve relative to its directory
        if caller_file and caller_file.exists():
            caller_dir = caller_file.parent
            zone_path = caller_dir / zone_path

    # Resolve the path
    zone_path = zone_path.resolve()
    zone_dict_name = str(zone_path)

    try:
        zone_module = load_module_from_file(zone_dict_name, zone_dict_name)
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
    filename: str, header: bool = False, remote: bool = False, delim: str = None
) -> pd.DataFrame:
    """Read CSV data, auto-detecting delimiter if not specified.

    When ``delim`` is None (default), sniffs the first line to choose
    between comma and space delimiters. This handles both the legacy
    space-delimited report format and the modern comma-delimited format.
    """
    if delim is None:
        # Auto-detect: sniff first line for commas
        try:
            with open(filename, "r") as f:
                first_line = f.readline()
            delim = "," if "," in first_line else " "
        except Exception:
            delim = " "  # fallback to legacy default

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


def _get_logo_path(strapline=False, invert=False):
    logo = "ZCFD"
    if strapline:
        logo += "_STRAPLINE"
    else:
        logo += "_NO_STRAPLINE"
    if invert:
        logo += "_WHITE"
    else:
        logo += "_CMYK"
    logo += ".png"
    file_loc = files("zutil").joinpath("assets") / logo
    return str(file_loc)


def _get_average_time_from_file(log_file: str, time_block: str = "solving") -> float:
    """Extract solve time from zCFD log file"""
    if not os.path.isfile(log_file):
        print("File not found: " + str(log_file))
        return 0.0

    with open(log_file, "r") as f:
        lines = f.readlines()
        time_pattern = re.compile(
            r"{}:\s+\d+\s+days\s+\d+\s+hrs\s+\d+\s+mins\s+(\d+)\s+s\s+\(count:\s+\d+\s+cycles,\s+avg:\s+(\d+(?:\.\d+)?)\s+ms\)".format(
                time_block
            )
        )
        for line in reversed(lines):  # Iterate in reverse to find the last match
            match = time_pattern.search(line)
            if match:
                return float(match.group(2))

    return 0.0  # Default return if no match is found


def load_module_from_file(module_name: str, file_path: str):
    """Load a Python module from a file path without modifying sys.path.

    Args:
        module_name: Name to give the loaded module
        file_path: Absolute path to the Python file

    Returns:
        The loaded module object

    Raises:
        ImportError: If the module cannot be loaded
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
