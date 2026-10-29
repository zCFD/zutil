"""Tests for zutil.paraview_state module."""

import os
import pprint
from pathlib import Path

import yaml


from zutil.paraview_state import (
    _collect_cases,
    _get_case_label,
    _get_surface_paths,
    _get_volume_path,
    _resolve_path,
    _sanitise_varname,
    generate_all,
    generate_all_state,
    generate_surface_state_for_bctype,
    generate_volume_state,
)


# ---------------------------------------------------------------------------
#  Helpers to create mock zCFD output structures
# ---------------------------------------------------------------------------


def _write_control_file(
    directory: Path,
    case_name: str,
    models: dict[str, str] | None = None,
) -> Path:
    """Write a minimal ``config version`` 2 control file that zCFD_Result parses.

    Args:
        models: Mapping of model name to mesh name.  More than one entry makes
            this a multi-model (overset) case.
    """
    if models is None:
        models = {"background": "TestMesh"}

    solution = {
        "format": "vtk",
        "surface variables": ["T", "p"],
        "volume variables": ["V", "p"],
    }

    parameters = {
        "config version": 2,
        "solver": {
            "equations": {"type": "RANS"},
            "numerical scheme": {"order": "second", "turbulence": {"model": "sst"}},
            "fluid properties": {
                "material": "ideal gas",
                "gamma": 1.4,
                "gas constant": 287.0,
            },
            "reference conditions": {
                "IC_1": {
                    "temperature": 300.0,
                    "pressure": 101325.0,
                    "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.5},
                }
            },
            "initialisation": {"reference": "IC_1", "initial conditions": "IC_1"},
            "time settings": {"type": "steady"},
            "convergence control": {
                "cycles": 100,
                "scheme": {"name": "implicit euler", "stage": 1},
            },
            "output settings": {"report": {"frequency": 1}, "solution": solution},
        },
        "model": {
            model_name: {
                "mesh": f"{mesh_name}.h5",
                "boundary conditions": {
                    "BC_1": {"zones": [1], "type": "wall", "kind": "no slip"},
                },
            }
            for model_name, mesh_name in models.items()
        },
    }

    path = directory / f"{case_name}.py"
    path.write_text("parameters = " + pprint.pformat(parameters, width=88) + "\n")
    return path


def _write_status_file(
    directory: Path,
    case_name: str,
    num_procs: int,
    model_ordering: list[str] | None = None,
) -> Path:
    """Write a mock status file — its presence is what marks the case as run."""
    status: dict = {
        "num processor": num_procs,
        "case": case_name,
        "version": "test",
    }
    if model_ordering:
        # The solver records the model DAG order here, and zCFD_Result uses it
        # to order the per-model output directories.
        status["model_ordering"] = list(model_ordering)

    path = directory / f"{case_name}_status.yaml"
    path.write_text(yaml.safe_dump(status))
    return path


def _create_output_dir(
    directory: Path,
    case_name: str,
    surface_types: list[str],
    models: dict[str, list[str]] | None = None,
) -> Path:
    """Create a mock output directory with volume and surface vtkhdf files.

    Args:
        surface_types: Boundary types to create files for in the single-model
            (flat) layout.
        models: For a multi-model case, model name -> boundary types.  Each
            model gets its own subdirectory, which is how zCFD_Result tells
            overset output from single-model output.
    """
    output_dir = directory / f"{case_name}_OUTPUT"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging directory (required by zCFD_Result)
    (output_dir / "LOGGING").mkdir(exist_ok=True)

    if models:
        for model_name, model_surfaces in models.items():
            model_dir = output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            (model_dir / f"{model_name}.vtkhdf").touch()
            for stype in model_surfaces:
                (model_dir / f"{model_name}_{stype}.vtkhdf").touch()
    else:
        (output_dir / f"{case_name}.vtkhdf").touch()
        for stype in surface_types:
            (output_dir / f"{case_name}_{stype}.vtkhdf").touch()

    return output_dir


def _setup_single_case(
    directory: Path,
    case_name: str = "TestCase",
    mesh_name: str = "TestMesh",
    num_procs: int = 4,
    surface_types: list[str] | None = None,
) -> Path:
    """Set up a complete single-case mock environment. Returns control file path."""
    if surface_types is None:
        surface_types = ["wall", "farfield"]

    control = _write_control_file(
        directory,
        case_name,
        models={"background": mesh_name},
    )
    _write_status_file(directory, case_name, num_procs)
    _create_output_dir(directory, case_name, surface_types)

    return control


def _setup_overset_case(
    directory: Path,
    case_name: str = "OversetCase",
    models: dict[str, dict] | None = None,
    num_procs: int = 64,
) -> Path:
    """Set up a mock multi-model (overset) case.

    An overset case is one control file with several ``model`` blocks and one
    output subdirectory per model.  Returns the control file path.
    """
    if models is None:
        models = {
            "background": {
                "mesh": "Background",
                "surfaces": ["farfield", "wall", "symmetry"],
            },
            "rotor": {
                "mesh": "Rotor",
                "surfaces": ["overset", "wall"],
            },
            "stator": {
                "mesh": "Stator",
                "surfaces": ["overset", "wall"],
            },
        }

    control = _write_control_file(
        directory,
        case_name,
        models={name: cfg["mesh"] for name, cfg in models.items()},
    )
    _write_status_file(directory, case_name, num_procs, model_ordering=list(models))
    _create_output_dir(
        directory,
        case_name,
        surface_types=[],
        models={name: cfg["surfaces"] for name, cfg in models.items()},
    )

    return control


# ---------------------------------------------------------------------------
#  Tests: _collect_cases
# ---------------------------------------------------------------------------


class TestCollectCases:
    def test_single_case(self, tmp_path: Path):
        control = _setup_single_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            assert len(cases) == 1
            assert _get_case_label(cases[0]) == "TestCase"
        finally:
            os.chdir(original_cwd)

    def test_overset_cases(self, tmp_path: Path):
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            assert len(cases) == 3
            labels = [_get_case_label(c) for c in cases]
            assert "background" in labels
            assert "rotor" in labels
            assert "stator" in labels
        finally:
            os.chdir(original_cwd)

    def test_overset_cases_preserve_order(self, tmp_path: Path):
        """Case order should match the model order recorded in the status file."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            labels = [_get_case_label(c) for c in cases]
            assert labels == ["background", "rotor", "stator"]
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
#  Tests: volume and surface path extraction
# ---------------------------------------------------------------------------


class TestPathExtraction:
    def test_volume_path(self, tmp_path: Path):
        control = _setup_single_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            vol = _get_volume_path(cases[0])
            assert vol is not None
            assert vol.name == "TestCase.vtkhdf"
        finally:
            os.chdir(original_cwd)

    def test_volume_path_none_when_missing(self, tmp_path: Path):
        """Returns None when the volume file doesn't exist on disk."""
        control = _setup_single_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            # Remove the volume file
            vol_file = cases[0]._volume_file_path
            vol_file.unlink()
            assert _get_volume_path(cases[0]) is None
        finally:
            os.chdir(original_cwd)

    def test_surface_paths(self, tmp_path: Path):
        control = _setup_single_case(tmp_path, surface_types=["wall", "farfield"])
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            surfaces = _get_surface_paths(cases[0])
            assert "wall" in surfaces
            assert "farfield" in surfaces
        finally:
            os.chdir(original_cwd)

    def test_no_ghost_paths_for_missing_surfaces(self, tmp_path: Path):
        """zCFD_Result initialises boundary paths to Path() — ensure these
        are not returned as valid surface paths."""
        control = _setup_single_case(tmp_path, surface_types=["wall"])
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            surfaces = _get_surface_paths(cases[0])
            # Only 'wall' should be present, not farfield/symmetry/etc.
            assert "wall" in surfaces
            assert "farfield" not in surfaces
            assert "symmetry" not in surfaces
            assert "periodic" not in surfaces
            assert "inflow" not in surfaces
            assert "outflow" not in surfaces
        finally:
            os.chdir(original_cwd)

    def test_overset_surfaces_detected(self, tmp_path: Path):
        """Overset boundary files must be detected despite the naming
        inconsistency in zCFD_Result (_overset_path vs _overset_boundary_path)."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            # rotor and stator have overset surfaces
            for case in cases:
                label = _get_case_label(case)
                surfaces = _get_surface_paths(case)
                if label in ("rotor", "stator"):
                    assert "overset" in surfaces, (
                        f"{label} should have overset surface detected"
                    )
                elif label == "background":
                    assert "overset" not in surfaces
        finally:
            os.chdir(original_cwd)

    def test_surface_paths_are_real_files(self, tmp_path: Path):
        """Every path returned by _get_surface_paths should point to a real file."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            for case in cases:
                for label, path in _get_surface_paths(case).items():
                    assert path.is_file(), (
                        f"{_get_case_label(case)} {label}: {path} should exist"
                    )
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
#  Tests: script generation
# ---------------------------------------------------------------------------


class TestGenerateVolumeState:
    def test_creates_volume_script(self, tmp_path: Path):
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_volume_state(started, tmp_path, base_dir)
            assert script is not None
            assert script.name == "paraview_view_volumes.py"
            assert script.exists()

            content = script.read_text()
            assert "from paraview.simple import *" in content
            assert "OpenDataFile" in content
            assert "background (volume)" in content
            assert "rotor (volume)" in content
            assert "stator (volume)" in content
        finally:
            os.chdir(original_cwd)


class TestGenerateSurfaceState:
    def test_creates_per_bctype_script(self, tmp_path: Path):
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_surface_state_for_bctype(
                started, "wall", tmp_path, base_dir
            )
            assert script is not None
            assert "wall" in script.name

            content = script.read_text()
            assert "background (wall)" in content
            assert "stator (wall)" in content
        finally:
            os.chdir(original_cwd)

    def test_creates_overset_surface_script(self, tmp_path: Path):
        """Overset surfaces should produce a script."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_surface_state_for_bctype(
                started, "overset", tmp_path, base_dir
            )
            assert script is not None
            assert "overset" in script.name

            content = script.read_text()
            assert "rotor (overset)" in content
            assert "stator (overset)" in content
            # background has no overset surface
            assert "background (overset)" not in content
        finally:
            os.chdir(original_cwd)

    def test_returns_none_for_absent_bctype(self, tmp_path: Path):
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_surface_state_for_bctype(
                started, "periodic", tmp_path, base_dir
            )
            assert script is None
        finally:
            os.chdir(original_cwd)


class TestGenerateAllState:
    def test_creates_combined_script(self, tmp_path: Path):
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_all_state(started, tmp_path, base_dir)
            assert script is not None
            assert script.name == "paraview_view_all.py"

            content = script.read_text()
            # Volumes
            assert "background (volume)" in content
            assert "rotor (volume)" in content
            # Surfaces
            assert "overset" in content
            assert "rotor (wall)" in content
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
#  Tests: generate_all orchestrator
# ---------------------------------------------------------------------------


class TestGenerateAll:
    def test_overset_mode(self, tmp_path: Path):
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(control, output_dir=tmp_path)
            assert len(scripts) > 0
            script_names = {s.name for s in scripts}
            assert "paraview_view_volumes.py" in script_names
            assert "paraview_view_all.py" in script_names
            assert "paraview_view_all_surfaces.py" in script_names
            # Overset surface script should now be generated
            assert "paraview_view_surfaces_overset.py" in script_names
        finally:
            os.chdir(original_cwd)

    def test_single_case_mode(self, tmp_path: Path):
        control = _setup_single_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(control, output_dir=tmp_path)
            assert len(scripts) > 0
            script_names = {s.name for s in scripts}
            assert "paraview_view_volumes.py" in script_names
        finally:
            os.chdir(original_cwd)

    def test_no_output_returns_empty(self, tmp_path: Path):
        """If the solver hasn't run (no status file), return empty list."""
        _write_control_file(tmp_path, "NoRun")
        # No status file → zcfd_started = False
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(tmp_path / "NoRun.py", output_dir=tmp_path)
            assert scripts == []
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
#  Tests: generated script properties
# ---------------------------------------------------------------------------


class TestGeneratedScriptProperties:
    """Test the content and structure of generated ParaView scripts."""

    def test_scripts_are_valid_python(self, tmp_path: Path):
        """All generated scripts must be valid Python (syntax check)."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(control, output_dir=tmp_path)
            for script in scripts:
                content = script.read_text()
                compile(content, script.name, "exec")
        finally:
            os.chdir(original_cwd)

    def test_scripts_have_guarded_reset_camera(self, tmp_path: Path):
        """All scripts guard ResetCamera with _loaded_count > 0."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(control, output_dir=tmp_path)
            for script in scripts:
                content = script.read_text()
                assert "_loaded_count" in content
                assert "if _loaded_count > 0:" in content
                # No bare ResetCamera() calls
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped == "ResetCamera()":
                        # Must be inside the if block (indented)
                        assert line.startswith("    "), (
                            f"Bare ResetCamera() in {script.name}"
                        )
        finally:
            os.chdir(original_cwd)

    def test_scripts_use_try_except_not_isfile(self, tmp_path: Path):
        """Scripts must use try/except (not os.path.isfile) for remote compat."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(control, output_dir=tmp_path)
            for script in scripts:
                content = script.read_text()
                assert "os.path.isfile" not in content, (
                    f"{script.name} should not use os.path.isfile "
                    "(breaks in client-server mode)"
                )
                assert "try:" in content
                assert "except Exception" in content
        finally:
            os.chdir(original_cwd)

    def test_scripts_have_base_dir_variable(self, tmp_path: Path):
        """Scripts must define _base_dir for remote mount override."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(control, output_dir=tmp_path)
            for script in scripts:
                content = script.read_text()
                assert "_base_dir = " in content, (
                    f"{script.name} should have _base_dir variable"
                )
                # _base_dir should be a resolved absolute path
                for line in content.splitlines():
                    if line.startswith("_base_dir = "):
                        path_val = line.split("=", 1)[1].strip().strip("'\"")
                        assert os.path.isabs(path_val), (
                            f"_base_dir should be absolute, got: {path_val}"
                        )
        finally:
            os.chdir(original_cwd)

    def test_scripts_create_render_view_upfront(self, tmp_path: Path):
        """Scripts must create the render view before any reader blocks."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(control, output_dir=tmp_path)
            for script in scripts:
                content = script.read_text()
                assert '_render_view = GetActiveViewOrCreate("RenderView")' in content
                # Ensure it appears before any OpenDataFile calls
                view_pos = content.index("_render_view = ")
                if "OpenDataFile" in content:
                    reader_pos = content.index("OpenDataFile")
                    assert view_pos < reader_pos, (
                        f"{script.name}: _render_view must be created before readers"
                    )
        finally:
            os.chdir(original_cwd)

    def test_scripts_use_relative_paths_from_base_dir(self, tmp_path: Path):
        """Paths in generated scripts must be relative to _base_dir,
        not absolute, so changing _base_dir works."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(control, output_dir=tmp_path)
            for script in scripts:
                content = script.read_text()
                for line in content.splitlines():
                    if "os.path.join(_base_dir," in line:
                        # Extract the second argument to os.path.join
                        rel_arg = line.split("os.path.join(_base_dir,")[1]
                        rel_arg = rel_arg.strip().rstrip(")")
                        # Should not be an absolute path
                        assert not rel_arg.strip("'\"").startswith("/"), (
                            f"Path should be relative in {script.name}: {line}"
                        )
        finally:
            os.chdir(original_cwd)

    def test_distinct_colours_per_case(self, tmp_path: Path):
        """Each case should get a distinct colour in the generated script."""
        control = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(control, output_dir=tmp_path)
            vol_script = [s for s in scripts if s.name == "paraview_view_volumes.py"][0]
            content = vol_script.read_text()

            colours = []
            for line in content.splitlines():
                if "DiffuseColor" in line:
                    colours.append(line.strip())

            # 3 cases → 3 different colours
            assert len(colours) == 3
            assert len(set(colours)) == 3, "Each case should have a unique colour"
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
#  Tests: utilities
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_sanitise_varname(self):
        assert _sanitise_varname("Aircraft_steady") == "Aircraft_steady"
        assert _sanitise_varname("mass flow") == "mass_flow"
        assert _sanitise_varname("my-case.py") == "my_case_py"

    def test_sanitise_varname_special_chars(self):
        """All non-alphanumeric/underscore chars become underscores."""
        assert _sanitise_varname("a.b-c d(e)") == "a_b_c_d_e_"
        assert _sanitise_varname("123") == "123"

    def test_resolve_path(self, tmp_path: Path):
        """_resolve_path returns a string of the resolved absolute path."""
        p = tmp_path / "sub" / "file.vtkhdf"
        resolved = _resolve_path(p)
        assert isinstance(resolved, str)
        assert os.path.isabs(resolved)

    def test_resolve_path_resolves_symlinks(self, tmp_path: Path):
        """Symlinks should be resolved to real paths."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_file = real_dir / "data.vtkhdf"
        real_file.touch()

        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)
        link_file = link_dir / "data.vtkhdf"

        resolved = _resolve_path(link_file)
        assert "link" not in resolved
        assert "real" in resolved
