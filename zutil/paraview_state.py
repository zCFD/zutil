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

Generate ParaView Python state files for visualising zCFD overset case outputs.

This module does NOT require ParaView to be installed — it only generates
Python scripts that use ``paraview.simple``. The generated scripts can then
be run via ``pvpython`` or loaded inside the ParaView GUI.

Uses :class:`zutil.fileutils.zCFD_Result` and
:class:`zutil.fileutils.zCFD_Overset_Result` for output file discovery.

Usage (CLI)::

    generate_paraview_helpers override_steady.py
    generate_paraview_helpers -c Aircraft_steady   # single-case mode

Usage (Python)::

    from zutil.paraview_state import generate_all
    generate_all("override_steady.py")
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
from pathlib import Path
from typing import Optional

from zutil.fileutils import get_zcfd_result, zCFD_Result, zCFD_Overset_Result


# Boundary type attribute names on zCFD_Result → human-readable label.
# Each key is the private attribute name on zCFD_Result that holds the Path
# for that boundary type.
BOUNDARY_ATTRS = {
    "_wall_boundary_path": "wall",
    "_symmetry_boundary_path": "symmetry",
    "_farfield_boundary_path": "farfield",
    "_periodic_boundary_path": "periodic",
    "_immersed_boundary_path": "immersed_wall",
    "_inflow_boundary_path": "inflow",
    "_outflow_boundary_path": "outflow",
    "_overset_boundary_path": "overset",
    # Also check _overset_path (naming inconsistency in zCFD_Result)
    "_overset_path": "overset",
}

# A curated colour palette for distinguishing meshes (RGB, 0-1 range).
MESH_COLOURS = [
    (0.121, 0.466, 0.706),  # blue
    (1.000, 0.498, 0.055),  # orange
    (0.173, 0.627, 0.173),  # green
    (0.839, 0.153, 0.157),  # red
    (0.580, 0.404, 0.741),  # purple
    (0.549, 0.337, 0.294),  # brown
    (0.890, 0.467, 0.761),  # pink
    (0.498, 0.498, 0.498),  # grey
    (0.737, 0.741, 0.133),  # olive
    (0.090, 0.745, 0.812),  # cyan
    (0.682, 0.780, 0.910),  # light blue
    (0.996, 0.769, 0.580),  # peach
]


# ---------------------------------------------------------------------------
#  Output info extraction from zCFD_Result objects
# ---------------------------------------------------------------------------


def _get_case_label(result: zCFD_Result) -> str:
    """Return a human-readable label for a case (the control file stem)."""
    return str(result._control_file_stem)


def _get_volume_path(result: zCFD_Result) -> Optional[Path]:
    """Return the volume .vtkhdf path if it exists on disk."""
    vol = result._volume_file_path
    if vol and vol != Path() and vol.is_file():
        return vol
    return None


def _get_surface_paths(result: zCFD_Result) -> dict[str, Path]:
    """Return a dict of {bc_label: Path} for boundary files that exist on disk.

    Uses the boundary_files list from zCFD_Result which is populated by
    globbing the output directory, so only files that actually exist are
    included.
    """
    surfaces: dict[str, Path] = {}
    for attr, label in BOUNDARY_ATTRS.items():
        path = getattr(result, attr, None)
        # Path() (empty/unset) is truthy but points to '.', so check explicitly
        if path is not None and path != Path() and path.is_file():
            surfaces[label] = path
    return surfaces


def _get_immersed_wall_paths(result: zCFD_Result) -> list[tuple[str, list[str]]]:
    """Return a list of (stl_stem, [vtu_path, ...]) for interpolated immersed
    wall surfaces that exist on disk.

    Each entry corresponds to one STL surface defined in the control file's
    ``write output`` → ``immersed boundary surface`` list.  The VTU paths are
    sorted by cycle number.
    """
    entries: list[tuple[str, list[str]]] = []
    immersed_wall_paths = getattr(result, "immersed_wall_paths", [])
    if not immersed_wall_paths:
        return entries

    # Recover the STL stem names from the parameters to pair with the paths
    immersed_list = result.parameters.get("write output", {}).get(
        "immersed boundary surface", []
    )
    if not immersed_list:
        return entries

    for i, vtu_files in enumerate(immersed_wall_paths):
        if not vtu_files:
            continue
        if i < len(immersed_list):
            stl_stem = Path(immersed_list[i]["stl"]).stem
        else:
            # Fallback: derive the stem from the first VTU filename
            stl_stem = Path(vtu_files[0]).stem.rsplit("_", 1)[0]
        entries.append((stl_stem, vtu_files))

    return entries


def _collect_cases(
    control_path: str | Path,
) -> list[zCFD_Result]:
    """Return a list of :class:`zCFD_Result` objects for the given control file.

    If *control_path* is an override file (overset), returns one result per
    mesh.  Otherwise returns a single-element list.
    """
    result = get_zcfd_result(str(control_path))
    if isinstance(result, zCFD_Overset_Result):
        return result.overset_cases
    else:
        return [result]


# ---------------------------------------------------------------------------
#  ParaView state script generation
# ---------------------------------------------------------------------------


def _resolve_path(path: Path) -> str:
    """Return the real (symlink-resolved) absolute path as a string."""
    return str(path.resolve())


def _preamble(script_name: str, base_dir: str) -> str:
    """Return the common preamble for generated ParaView scripts."""
    return textwrap.dedent(f"""\
        # Auto-generated by zutil.paraview_state
        # Run with: pvpython {script_name}
        # Or load in ParaView via: File > Run Script...
        #
        # To use on a different machine / mount point, change _base_dir below:

        import os
        import sys

        from paraview.simple import *

        paraview.simple._DisableFirstRenderCameraReset()

        _base_dir = {base_dir!r}
        _loaded_count = 0
        _render_view = GetActiveViewOrCreate("RenderView")

    """)


def _reader_block(
    var_name: str,
    abs_path: str,
    display_name: str,
    colour: tuple,
    base_dir: str,
) -> str:
    """Generate code to open a VTKHDF file and show it."""
    r, g, b = colour
    # Store path relative to _base_dir so the override works
    rel = os.path.relpath(abs_path, base_dir)
    return textwrap.dedent(f"""\
        # --- {display_name} ---
        _{var_name}_path = os.path.join(_base_dir, {rel!r})
        try:
            {var_name} = OpenDataFile(_{var_name}_path)
            RenameSource({display_name!r}, {var_name})
            {var_name}_display = Show({var_name}, _render_view)
            {var_name}_display.DiffuseColor = [{r:.3f}, {g:.3f}, {b:.3f}]
            _loaded_count += 1
            print("  Loaded: " + {display_name!r})
        except Exception as _e:
            print("  Skipped: " + {display_name!r} + " (" + str(_e) + ")")

    """)


def _vtu_reader_block(
    var_name: str,
    abs_paths: list[str],
    display_name: str,
    colour: tuple,
    base_dir: str,
) -> str:
    """Generate code to open a VTU file series reader and show it."""
    r, g, b = colour
    # Build relative path list
    rel_paths = [os.path.relpath(p, base_dir) for p in abs_paths]
    return textwrap.dedent(f"""\
        # --- {display_name} ---
        _{var_name}_files = [os.path.join(_base_dir, p) for p in {rel_paths!r}]
        try:
            {var_name} = XMLUnstructuredGridReader(FileName=_{var_name}_files)
            UpdatePipeline(proxy={var_name})
            RenameSource({display_name!r}, {var_name})
            {var_name}_display = Show({var_name}, _render_view)
            {var_name}_display.DiffuseColor = [{r:.3f}, {g:.3f}, {b:.3f}]
            _loaded_count += 1
            print("  Loaded: " + {display_name!r} + " (" + str(len(_{var_name}_files)) + " timestep(s))")
        except Exception as _e:
            print("  Skipped: " + {display_name!r} + " (" + str(_e) + ")")

    """)


def _footer(count_label: str, pvsm_path: Optional[str] = None) -> str:
    """Generate the script footer with guarded ResetCamera.

    Args:
        pvsm_path: If set, append a ``SaveState()`` call that writes
            the pipeline state to this ``.pvsm`` file.
    """
    parts = [
        "if _loaded_count > 0:\n",
        "    GetAnimationScene().UpdateAnimationUsingDataTimeSteps()\n",
        "    ResetCamera()\n",
        f'    print("\\nDone \u2014 loaded %d {count_label}." % _loaded_count)\n',
    ]
    if pvsm_path:
        parts.append(f"    _pvsm_path = os.path.join(_base_dir, {pvsm_path!r})\n")
        parts.append("    try:\n")
        parts.append("        SaveState(_pvsm_path)\n")
        parts.append('        print("  Saved state: " + _pvsm_path)\n')
        parts.append("    except Exception:\n")
        parts.append('        print("  Could not save state (client-server mode?)")\n')
    parts.append("else:\n")
    parts.append(
        '    print("\\nNo files were loaded. Check that the output files exist.")\n'
    )
    return "".join(parts)


def _sanitise_varname(name: str) -> str:
    """Turn a case/bc name into a valid Python variable name."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


# ---------------------------------------------------------------------------
#  Script generators
# ---------------------------------------------------------------------------


def generate_volume_state(
    cases: list[zCFD_Result],
    output_path: Path,
    base_dir: str,
    save_pvsm: bool = True,
) -> Optional[Path]:
    """Generate a script that loads all volume meshes."""
    volume_entries = []
    for case in cases:
        vol = _get_volume_path(case)
        if vol is not None:
            volume_entries.append((_get_case_label(case), vol))

    if not volume_entries:
        return None

    script_name = "paraview_view_volumes.py"
    pvsm_name = "paraview_view_volumes.pvsm" if save_pvsm else None
    script_path = output_path / script_name

    lines = [_preamble(script_name, base_dir)]
    lines.append('print("Loading volume meshes...")\n')

    for i, (label, vol_path) in enumerate(volume_entries):
        colour = MESH_COLOURS[i % len(MESH_COLOURS)]
        var = _sanitise_varname(label) + "_vol"
        abs_path = _resolve_path(vol_path)
        display = f"{label} (volume)"
        lines.append(_reader_block(var, abs_path, display, colour, base_dir))

    lines.append(_footer("volume mesh(es)", pvsm_path=pvsm_name))

    script_path.write_text("".join(lines))
    return script_path


def generate_surface_state_for_bctype(
    cases: list[zCFD_Result],
    bctype: str,
    output_path: Path,
    base_dir: str,
    save_pvsm: bool = True,
) -> Optional[Path]:
    """Generate a script that loads surface meshes for a single BC type."""
    surface_entries = []
    for case in cases:
        surfaces = _get_surface_paths(case)
        if bctype in surfaces:
            surface_entries.append((_get_case_label(case), surfaces[bctype]))

    if not surface_entries:
        return None

    bc_var = _sanitise_varname(bctype)
    script_name = f"paraview_view_surfaces_{bc_var}.py"
    pvsm_name = f"paraview_view_surfaces_{bc_var}.pvsm" if save_pvsm else None
    script_path = output_path / script_name

    lines = [_preamble(script_name, base_dir)]
    lines.append(f'print("Loading {bctype} surfaces...")\n')

    for i, (label, surf_path) in enumerate(surface_entries):
        colour = MESH_COLOURS[i % len(MESH_COLOURS)]
        var = _sanitise_varname(label) + f"_{bc_var}"
        abs_path = _resolve_path(surf_path)
        display = f"{label} ({bctype})"
        lines.append(_reader_block(var, abs_path, display, colour, base_dir))

    lines.append(_footer(f"{bctype} surface(s)", pvsm_path=pvsm_name))

    script_path.write_text("".join(lines))
    return script_path


def generate_immersed_wall_state(
    cases: list[zCFD_Result],
    output_path: Path,
    base_dir: str,
    save_pvsm: bool = True,
) -> Optional[Path]:
    """Generate a script that loads interpolated immersed wall .vtu surfaces."""
    all_entries: list[tuple[str, str, list[str]]] = []  # (case_label, stl_stem, files)
    for case in cases:
        for stl_stem, vtu_files in _get_immersed_wall_paths(case):
            all_entries.append((_get_case_label(case), stl_stem, vtu_files))

    if not all_entries:
        return None

    script_name = "paraview_view_interpolated_stl.py"
    pvsm_name = "paraview_view_interpolated_stl.pvsm" if save_pvsm else None
    script_path = output_path / script_name

    lines = [_preamble(script_name, base_dir)]
    lines.append('print("Loading interpolated STL surfaces...")\n')

    for i, (case_label, stl_stem, vtu_files) in enumerate(all_entries):
        colour = MESH_COLOURS[i % len(MESH_COLOURS)]
        var = _sanitise_varname(case_label) + "_" + _sanitise_varname(stl_stem)
        display = f"{case_label} ({stl_stem})"
        abs_paths = [str(Path(f).resolve()) for f in vtu_files]
        lines.append(_vtu_reader_block(var, abs_paths, display, colour, base_dir))

    lines.append(_footer("immersed wall surface(s)", pvsm_path=pvsm_name))

    script_path.write_text("".join(lines))
    return script_path


def generate_all_surfaces_state(
    cases: list[zCFD_Result],
    output_path: Path,
    base_dir: str,
    save_pvsm: bool = True,
) -> Optional[Path]:
    """Generate a script that loads all surfaces, grouped by BC type."""
    all_bctypes: set[str] = set()
    for case in cases:
        all_bctypes.update(_get_surface_paths(case).keys())

    if not all_bctypes:
        return None

    script_name = "paraview_view_all_surfaces.py"
    pvsm_name = "paraview_view_all_surfaces.pvsm" if save_pvsm else None
    script_path = output_path / script_name

    lines = [_preamble(script_name, base_dir)]
    lines.append('print("Loading all surface meshes...")\n')

    for bctype in sorted(all_bctypes):
        lines.append(f'print("\\n--- {bctype} surfaces ---")\n')
        for i, case in enumerate(cases):
            surfaces = _get_surface_paths(case)
            if bctype not in surfaces:
                continue
            label = _get_case_label(case)
            colour = MESH_COLOURS[i % len(MESH_COLOURS)]
            var = _sanitise_varname(label) + f"_{_sanitise_varname(bctype)}"
            abs_path = _resolve_path(surfaces[bctype])
            display = f"{label} ({bctype})"
            lines.append(_reader_block(var, abs_path, display, colour, base_dir))

    lines.append(_footer("surface file(s)", pvsm_path=pvsm_name))

    script_path.write_text("".join(lines))
    return script_path


def generate_all_state(
    cases: list[zCFD_Result],
    output_path: Path,
    base_dir: str,
    save_pvsm: bool = True,
) -> Optional[Path]:
    """Generate a script that loads everything — volumes and all surfaces."""
    script_name = "paraview_view_all.py"
    pvsm_name = "paraview_view_all.pvsm" if save_pvsm else None
    script_path = output_path / script_name

    lines = [_preamble(script_name, base_dir)]
    lines.append('print("Loading all volume and surface meshes...")\n')

    # Volumes
    lines.append('print("\\n=== Volume meshes ===")\n')
    for i, case in enumerate(cases):
        vol = _get_volume_path(case)
        if vol is None:
            continue
        label = _get_case_label(case)
        colour = MESH_COLOURS[i % len(MESH_COLOURS)]
        var = _sanitise_varname(label) + "_vol"
        abs_path = _resolve_path(vol)
        display = f"{label} (volume)"
        lines.append(_reader_block(var, abs_path, display, colour, base_dir))

    # Surfaces grouped by BC type
    all_bctypes: set[str] = set()
    for case in cases:
        all_bctypes.update(_get_surface_paths(case).keys())

    for bctype in sorted(all_bctypes):
        lines.append(f'print("\\n=== {bctype} surfaces ===")\n')
        for i, case in enumerate(cases):
            surfaces = _get_surface_paths(case)
            if bctype not in surfaces:
                continue
            label = _get_case_label(case)
            colour = MESH_COLOURS[i % len(MESH_COLOURS)]
            var = _sanitise_varname(label) + f"_{_sanitise_varname(bctype)}"
            abs_path = _resolve_path(surfaces[bctype])
            display = f"{label} ({bctype})"
            lines.append(_reader_block(var, abs_path, display, colour, base_dir))

    # Interpolated immersed wall surfaces (.vtu series)
    imm_colour_offset = len(cases)  # offset colours so they don't clash
    for i, case in enumerate(cases):
        for stl_stem, vtu_files in _get_immersed_wall_paths(case):
            lines.append(f'print("\\n=== immersed wall: {stl_stem} ===")\n')
            colour = MESH_COLOURS[(imm_colour_offset + i) % len(MESH_COLOURS)]
            label = _get_case_label(case)
            var = _sanitise_varname(label) + "_" + _sanitise_varname(stl_stem)
            display = f"{label} ({stl_stem})"
            abs_paths = [str(Path(f).resolve()) for f in vtu_files]
            lines.append(_vtu_reader_block(var, abs_paths, display, colour, base_dir))

    lines.append(_footer("file(s) total", pvsm_path=pvsm_name))

    script_path.write_text("".join(lines))
    return script_path


# ---------------------------------------------------------------------------
#  Top-level orchestrator
# ---------------------------------------------------------------------------


def generate_all(
    control_path: str | Path,
    output_dir: Optional[str | Path] = None,
    save_pvsm: bool = True,
) -> list[Path]:
    """Generate all ParaView helper scripts.

    Args:
        control_path: Path to an override ``.py`` file (overset) or a single
            case control file.
        output_dir: Directory in which to write the generated scripts.
            Defaults to the directory containing the control file.
        save_pvsm: If True (default), generated scripts include a
            ``SaveState()`` call that writes a ``.pvsm`` file when run
            with ``pvpython``.  Set to False for Python-only scripts.

    Returns:
        List of paths to the generated scripts.
    """
    control_path = Path(control_path)
    if output_dir is None:
        output_dir = control_path.parent
    output_dir = Path(output_dir).resolve()

    cases = _collect_cases(control_path)

    if not cases:
        print("No cases found.")
        return []

    # Filter to cases where the solver has actually run
    started_cases = [c for c in cases if c.zcfd_started]
    if not started_cases:
        print("No output directories found. Has the solver been run?")
        return []

    fmt_label = "pvsm + python" if save_pvsm else "python only"
    print(f"Found output for {len(started_cases)} case(s) (format: {fmt_label}):")
    for case in started_cases:
        label = _get_case_label(case)
        vol = "yes" if _get_volume_path(case) else "no"
        surfs = ", ".join(sorted(_get_surface_paths(case).keys())) or "none"
        imm_stems = [stem for stem, _ in _get_immersed_wall_paths(case)]
        imm_info = f", interpolated_stl=[{', '.join(imm_stems)}]" if imm_stems else ""
        print(f"  {label}: volume={vol}, surfaces=[{surfs}]{imm_info}")

    generated: list[Path] = []

    # Resolve the base directory so generated paths work across symlinks
    base_dir = str(output_dir.resolve())

    # Volume script
    path = generate_volume_state(started_cases, output_dir, base_dir, save_pvsm)
    if path:
        generated.append(path)

    # Per-BC-type surface scripts
    all_bctypes: set[str] = set()
    for case in started_cases:
        all_bctypes.update(_get_surface_paths(case).keys())

    for bctype in sorted(all_bctypes):
        path = generate_surface_state_for_bctype(
            started_cases, bctype, output_dir, base_dir, save_pvsm
        )
        if path:
            generated.append(path)

    # Combined surfaces script
    if len(all_bctypes) > 0:
        path = generate_all_surfaces_state(
            started_cases, output_dir, base_dir, save_pvsm
        )
        if path:
            generated.append(path)

    # Interpolated immersed wall surfaces script
    path = generate_immersed_wall_state(started_cases, output_dir, base_dir, save_pvsm)
    if path:
        generated.append(path)

    # Everything script
    path = generate_all_state(started_cases, output_dir, base_dir, save_pvsm)
    if path:
        generated.append(path)

    print(f"\nGenerated {len(generated)} ParaView helper script(s):")
    for p in generated:
        print(f"  {p.relative_to(output_dir) if p.is_relative_to(output_dir) else p}")
    if save_pvsm:
        print("\nRun with pvpython to generate .pvsm state files:")
        print("  pvpython <script.py>")

    return generated


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``generate_paraview_helpers``."""
    parser = argparse.ArgumentParser(
        description="Generate ParaView helper scripts for visualising zCFD case outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Overset case — default generates .pvsm-saving scripts:
              generate_paraview_helpers override_steady.py

              # Python-only scripts (no SaveState):
              generate_paraview_helpers --python override_steady.py

              # Single-case mode:
              generate_paraview_helpers Aircraft_steady.py
        """),
    )
    parser.add_argument(
        "control_file",
        help=(
            "Path to the control .py file. For overset cases, pass the override file "
            "(e.g. override_steady.py). For single cases, pass the case control file "
            "(e.g. Aircraft_steady.py)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory to write helper scripts (default: same directory as control file)",
    )
    parser.add_argument(
        "--python",
        action="store_true",
        default=False,
        help=(
            "Generate Python-only scripts without SaveState(). "
            "By default, scripts include a SaveState() call that writes "
            ".pvsm files when run with pvpython."
        ),
    )

    args = parser.parse_args()

    generate_all(
        control_path=args.control_file,
        output_dir=args.output_dir,
        save_pvsm=not args.python,
    )


if __name__ == "__main__":
    main()
