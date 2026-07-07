"""Tests for zutil.paraview_state module."""

import json
import os
import textwrap
from pathlib import Path


from zutil.paraview_state import (
    _collect_cases,
    _get_case_label,
    _get_immersed_wall_paths,
    _get_surface_paths,
    _get_volume_path,
    _resolve_path,
    _sanitise_varname,
    generate_all,
    generate_all_state,
    generate_immersed_wall_state,
    generate_surface_state_for_bctype,
    generate_volume_state,
)


# ---------------------------------------------------------------------------
#  Helpers to create mock zCFD output structures
# ---------------------------------------------------------------------------


def _write_control_file(
    directory: Path,
    case_name: str,
    bc_config: dict,
    write_output_extra: dict | None = None,
) -> Path:
    """Write a minimal zCFD control file that zCFD_Result can parse.

    Args:
        write_output_extra: Additional keys to merge into the ``write output``
            dict (e.g. ``{"immersed boundary surface": [...]}``).  Values are
            written as their Python repr so they must be JSON-friendly.
    """
    bc_lines = ""
    for bc_key, bc_val in bc_config.items():
        bc_lines += f'    "{bc_key}": {bc_val},\n'

    wo_extra_lines = ""
    if write_output_extra:
        for wo_key, wo_val in write_output_extra.items():
            wo_extra_lines += f'                "{wo_key}": {wo_val!r},\n'

    content = textwrap.dedent(f"""\
        parameters = {{
            "units": "SI",
            "equations": "RANS",
            "RANS": {{"order": "second", "turbulence": {{"model": "sst"}}}},
            "time marching": {{
                "unsteady": {{"total time": 1.0, "time step": 1.0, "order": "second"}},
                "scheme": {{"name": "implicit euler", "stage": 1}},
                "cfl": 5,
                "cycles": 100,
            }},
            "material": "air",
            "air": {{"gamma": 1.4, "gas constant": 287.0}},
            "IC_1": {{"temperature": 300, "pressure": 101325, "V": {{"vector": [1, 0, 0], "Mach": 0.5}}}},
            "initial": "IC_1",
            "reference": "IC_1",
            "write output": {{
                "format": "vtk",
                "surface variables": ["T", "p"],
                "volume variables": ["V", "p"],
{wo_extra_lines}
            }},
            {bc_lines}
        }}
    """)
    path = directory / f"{case_name}.py"
    path.write_text(content)
    return path


def _write_status_file(
    directory: Path, case_name: str, mesh_name: str, num_procs: int
) -> Path:
    """Write a mock status file."""
    status = {
        "num processor": num_procs,
        "case": case_name,
        "mesh": mesh_name,
        "version": "test",
    }
    path = directory / f"{case_name}_status.txt"
    path.write_text(json.dumps(status))
    return path


def _create_output_dir(
    directory: Path,
    case_name: str,
    num_procs: int,
    surface_types: list[str],
    immersed_stl_stems: list[str] | None = None,
    immersed_cycles: list[int] | None = None,
) -> Path:
    """Create mock output directory with volume and surface vtkhdf files.

    Args:
        immersed_stl_stems: STL stem names to create mock .vtu files for.
        immersed_cycles: Cycle numbers for the mock .vtu files.
    """
    output_dir = directory / f"{case_name}_P{num_procs}_OUTPUT"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging directory (required by zCFD_Result)
    (output_dir / "LOGGING").mkdir(exist_ok=True)

    # Volume file
    (output_dir / f"{case_name}.vtkhdf").touch()

    # Surface files
    for stype in surface_types:
        (output_dir / f"{case_name}_{stype}.vtkhdf").touch()

    # Immersed wall .vtu files
    if immersed_stl_stems and immersed_cycles:
        for stl_stem in immersed_stl_stems:
            for cycle in immersed_cycles:
                (output_dir / f"{stl_stem}_{cycle}.vtu").touch()

    return output_dir


def _setup_single_case(
    directory: Path,
    case_name: str = "TestCase",
    mesh_name: str = "TestMesh",
    num_procs: int = 4,
    surface_types: list[str] | None = None,
    write_output_extra: dict | None = None,
    immersed_stl_stems: list[str] | None = None,
    immersed_cycles: list[int] | None = None,
) -> Path:
    """Set up a complete single-case mock environment. Returns control file path."""
    if surface_types is None:
        surface_types = ["wall", "farfield"]

    bc_config = {
        "BC_1": {"zone": [1], "type": "wall", "kind": "noslip"},
    }
    control = _write_control_file(
        directory, case_name, bc_config, write_output_extra=write_output_extra
    )
    _write_status_file(directory, case_name, mesh_name, num_procs)
    _create_output_dir(
        directory,
        case_name,
        num_procs,
        surface_types,
        immersed_stl_stems=immersed_stl_stems,
        immersed_cycles=immersed_cycles,
    )

    return control


def _setup_overset_case(
    directory: Path,
    cases: dict[str, dict] | None = None,
) -> Path:
    """Set up a mock overset case. Returns override file path."""
    if cases is None:
        cases = {
            "Background": {
                "mesh": "Background",
                "nprocs": 64,
                "surfaces": ["farfield", "wall", "symmetry"],
            },
            "Rotor": {
                "mesh": "Rotor",
                "nprocs": 64,
                "surfaces": ["overset", "immersed wall"],
            },
            "Stator": {
                "mesh": "Stator",
                "nprocs": 64,
                "surfaces": ["overset", "wall"],
            },
        }

    # Write control files, status files, and output dirs
    mesh_case_pairs = []
    for case_name, cfg in cases.items():
        bc_config = {
            "BC_1": {"zone": [1], "type": "wall", "kind": "noslip"},
        }
        _write_control_file(directory, case_name, bc_config)
        _write_status_file(directory, case_name, cfg["mesh"], cfg["nprocs"])
        _create_output_dir(directory, case_name, cfg["nprocs"], cfg["surfaces"])
        mesh_case_pairs.append((f"{cfg['mesh']}.h5", f"{case_name}.py"))

    # Write override file
    pairs_str = ",\n        ".join(f'("{m}", "{c}")' for m, c in mesh_case_pairs)
    override_content = textwrap.dedent(f"""\
        override = {{
            "mesh_case_pair": [
                {pairs_str},
            ]
        }}
    """)
    override_file = directory / "override_steady.py"
    override_file.write_text(override_content)

    return override_file


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
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(override)
            assert len(cases) == 3
            labels = [_get_case_label(c) for c in cases]
            assert "Background" in labels
            assert "Rotor" in labels
            assert "Stator" in labels
        finally:
            os.chdir(original_cwd)

    def test_overset_cases_preserve_order(self, tmp_path: Path):
        """Case order should match the mesh_case_pair order in the override."""
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(override)
            labels = [_get_case_label(c) for c in cases]
            assert labels == ["Background", "Rotor", "Stator"]
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
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(override)
            # Rotor and Stator have overset surfaces
            for case in cases:
                label = _get_case_label(case)
                surfaces = _get_surface_paths(case)
                if label in ("Rotor", "Stator"):
                    assert "overset" in surfaces, (
                        f"{label} should have overset surface detected"
                    )
                elif label == "Background":
                    assert "overset" not in surfaces
        finally:
            os.chdir(original_cwd)

    def test_immersed_wall_with_space_in_filename(self, tmp_path: Path):
        """Files with spaces in the name (e.g. 'immersed wall') are detected."""
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(override)
            for case in cases:
                label = _get_case_label(case)
                surfaces = _get_surface_paths(case)
                if label == "Rotor":
                    assert "immersed_wall" in surfaces, (
                        "Rotor should have immersed_wall surface"
                    )
        finally:
            os.chdir(original_cwd)

    def test_surface_paths_are_real_files(self, tmp_path: Path):
        """Every path returned by _get_surface_paths should point to a real file."""
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(override)
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
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(override)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_volume_state(started, tmp_path, base_dir)
            assert script is not None
            assert script.name == "paraview_view_volumes.py"
            assert script.exists()

            content = script.read_text()
            assert "from paraview.simple import *" in content
            assert "OpenDataFile" in content
            assert "Background (volume)" in content
            assert "Rotor (volume)" in content
            assert "Stator (volume)" in content
        finally:
            os.chdir(original_cwd)


class TestGenerateSurfaceState:
    def test_creates_per_bctype_script(self, tmp_path: Path):
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(override)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_surface_state_for_bctype(
                started, "wall", tmp_path, base_dir
            )
            assert script is not None
            assert "wall" in script.name

            content = script.read_text()
            assert "Background (wall)" in content
            assert "Stator (wall)" in content
        finally:
            os.chdir(original_cwd)

    def test_creates_overset_surface_script(self, tmp_path: Path):
        """Overset surfaces should produce a script."""
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(override)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_surface_state_for_bctype(
                started, "overset", tmp_path, base_dir
            )
            assert script is not None
            assert "overset" in script.name

            content = script.read_text()
            assert "Rotor (overset)" in content
            assert "Stator (overset)" in content
            # Background has no overset surface
            assert "Background (overset)" not in content
        finally:
            os.chdir(original_cwd)

    def test_returns_none_for_absent_bctype(self, tmp_path: Path):
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(override)
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
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(override)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_all_state(started, tmp_path, base_dir)
            assert script is not None
            assert script.name == "paraview_view_all.py"

            content = script.read_text()
            # Volumes
            assert "Background (volume)" in content
            assert "Rotor (volume)" in content
            # Surfaces
            assert "overset" in content
            assert "immersed_wall" in content
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
#  Tests: generate_all orchestrator
# ---------------------------------------------------------------------------


class TestGenerateAll:
    def test_overset_mode(self, tmp_path: Path):
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(override, output_dir=tmp_path)
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
        bc_config = {
            "BC_1": {"zone": [1], "type": "wall", "kind": "noslip"},
        }
        _write_control_file(tmp_path, "NoRun", bc_config)
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
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(override, output_dir=tmp_path)
            for script in scripts:
                content = script.read_text()
                compile(content, script.name, "exec")
        finally:
            os.chdir(original_cwd)

    def test_scripts_have_guarded_reset_camera(self, tmp_path: Path):
        """All scripts guard ResetCamera with _loaded_count > 0."""
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(override, output_dir=tmp_path)
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
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(override, output_dir=tmp_path)
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
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(override, output_dir=tmp_path)
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
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(override, output_dir=tmp_path)
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
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(override, output_dir=tmp_path)
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
        override = _setup_overset_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(override, output_dir=tmp_path)
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
        assert _sanitise_varname("immersed wall") == "immersed_wall"
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


# ---------------------------------------------------------------------------
#  Tests: interpolated immersed wall surface support
# ---------------------------------------------------------------------------


def _setup_immersed_case(tmp_path: Path) -> Path:
    """Create a single case with immersed boundary surface output."""
    return _setup_single_case(
        tmp_path,
        case_name="ImmersedCase",
        surface_types=["wall", "immersed wall"],
        write_output_extra={
            "immersed boundary surface": [
                {"stl": "rotor_blade.stl"},
                {"stl": "stator_blade.stl"},
            ],
        },
        immersed_stl_stems=["rotor_blade", "stator_blade"],
        immersed_cycles=[0, 1, 2, 10, 11],
    )


class TestImmersedWallPaths:
    def test_returns_entries_when_vtu_files_exist(self, tmp_path: Path):
        control = _setup_immersed_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            entries = _get_immersed_wall_paths(cases[0])
            assert len(entries) == 2
            stems = [stem for stem, _ in entries]
            assert "rotor_blade" in stems
            assert "stator_blade" in stems
        finally:
            os.chdir(original_cwd)

    def test_vtu_files_are_sorted_by_cycle(self, tmp_path: Path):
        control = _setup_immersed_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            entries = _get_immersed_wall_paths(cases[0])
            for _, vtu_files in entries:
                # Extract cycle numbers and check they're sorted
                import re

                cycles = []
                for f in vtu_files:
                    m = re.search(r"_(\d+)\.vtu$", f)
                    if m:
                        cycles.append(int(m.group(1)))
                assert cycles == sorted(cycles)
        finally:
            os.chdir(original_cwd)

    def test_returns_empty_when_no_immersed_config(self, tmp_path: Path):
        control = _setup_single_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            entries = _get_immersed_wall_paths(cases[0])
            assert entries == []
        finally:
            os.chdir(original_cwd)


class TestGenerateImmersedWallState:
    def test_creates_immersed_wall_script(self, tmp_path: Path):
        control = _setup_immersed_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_immersed_wall_state(started, tmp_path, base_dir)
            assert script is not None
            assert script.name == "paraview_view_interpolated_stl.py"
            assert script.exists()

            content = script.read_text()
            assert "XMLUnstructuredGridReader" in content
            assert "rotor_blade" in content
            assert "stator_blade" in content
        finally:
            os.chdir(original_cwd)

    def test_returns_none_when_no_immersed_data(self, tmp_path: Path):
        control = _setup_single_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_immersed_wall_state(started, tmp_path, base_dir)
            assert script is None
        finally:
            os.chdir(original_cwd)

    def test_immersed_wall_in_generate_all(self, tmp_path: Path):
        """generate_all should produce the immersed wall script."""
        control = _setup_immersed_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scripts = generate_all(control, output_dir=tmp_path)
            script_names = {s.name for s in scripts}
            assert "paraview_view_interpolated_stl.py" in script_names
        finally:
            os.chdir(original_cwd)

    def test_immersed_wall_in_all_state_script(self, tmp_path: Path):
        """generate_all_state should include immersed wall surfaces."""
        control = _setup_immersed_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_all_state(started, tmp_path, base_dir)
            assert script is not None
            content = script.read_text()
            assert "XMLUnstructuredGridReader" in content
            assert "rotor_blade" in content
            assert "immersed wall" in content
        finally:
            os.chdir(original_cwd)

    def test_generated_immersed_script_is_valid_python(self, tmp_path: Path):
        control = _setup_immersed_case(tmp_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cases = _collect_cases(control)
            started = [c for c in cases if c.zcfd_started]
            base_dir = str(tmp_path.resolve())
            script = generate_immersed_wall_state(started, tmp_path, base_dir)
            assert script is not None
            content = script.read_text()
            compile(content, script.name, "exec")
        finally:
            os.chdir(original_cwd)
