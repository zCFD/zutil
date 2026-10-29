"""
Every file a deck points the solver at must exist and be readable when the
deck is validated, resolved against the working directory, where the solver
reads it. A roughness map must also be a VTK PolyData file whose points carry
a 'Roughness' array, and a canopy height map a 'Height' array.

The tests assert on the text the engineer reads.
"""

import copy
import os

import pytest

from zutil.validate import (
    in_case_directory,
    validate_zcfd_comprehensive,
    validate_zcfd_input,
)
from zutil.validate.tests.test_core_orchestration import _minimal_valid_config

WALL_FUNCTION = {"type": "wall", "kind": "wall function"}

ROUGHNESS_VTP = """<?xml version="1.0"?>
<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <PolyData>
    <Piece NumberOfPoints="1" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">
      <PointData>
        <DataArray type="Float64" Name="{name}" format="ascii">
          0.01
        </DataArray>
      </PointData>
      <Points>
        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">
          0 0 0
        </DataArray>
      </Points>
    </Piece>
  </PolyData>
</VTKFile>
"""


def _with_wall(bc):
    deck = _minimal_valid_config()
    deck["model"]["base"]["boundary_conditions"]["ground"] = {"zones": [5], **bc}
    return deck


def _with_canopy(path):
    deck = _minimal_valid_config()
    deck["model"]["base"]["fluid_zones"] = {"trees": {"type": "canopy", "field": path}}
    return deck


def _with_profile(path):
    deck = _minimal_valid_config()
    deck["solver"]["reference_conditions"]["IC_1"]["profile"] = {"field": path}
    return deck


SITES = {
    "wall roughness": (
        lambda path: _with_wall(
            {**WALL_FUNCTION, "roughness": {"type": "length", "field": path}}
        ),
        "model.base.boundary_conditions.ground.roughness.field",
    ),
    "wall temperature": (
        lambda path: _with_wall({"type": "wall", "temperature": {"field": path}}),
        "model.base.boundary_conditions.ground.temperature.field",
    ),
    "wall heat flux": (
        lambda path: _with_wall({**WALL_FUNCTION, "heat_flux": {"field": path}}),
        "model.base.boundary_conditions.ground.heat_flux.field",
    ),
    "canopy height": (
        _with_canopy,
        "model.base.fluid_zones.trees.field",
    ),
    "reference condition profile": (
        _with_profile,
        "solver.reference_conditions.IC_1.profile.field",
    ),
}

LABELS = {
    "wall roughness": "Wall roughness field",
    "wall temperature": "Wall temperature or heat flux field",
    "wall heat flux": "Wall temperature or heat flux field",
    "canopy height": "Canopy height field",
    "reference condition profile": "Reference condition profile field",
}


def _errors(deck):
    model, nodes = validate_zcfd_comprehensive(copy.deepcopy(deck))
    errors = [n for n in nodes if n.get("severity", "error") == "error"]
    assert (model is None) == bool(errors)
    return errors


POINT_ARRAYS = {"canopy height": "Height"}


def _write(directory, name, text=None, site=None):
    path = directory / name
    array = POINT_ARRAYS.get(site, "Roughness")
    path.write_text(ROUGHNESS_VTP.format(name=array) if text is None else text)
    return path


@pytest.mark.parametrize("site", SITES)
class TestFieldFileExists:
    def test_missing_file_is_rejected_naming_label_path_and_directory(
        self, site, tmp_path
    ):
        make, field_path = SITES[site]
        with in_case_directory(tmp_path):
            (error,) = _errors(make("absent.vtp"))
        assert error["field_path"] == field_path
        assert error["display_message"] == (
            f'{LABELS[site]} "absent.vtp" not found in {tmp_path}'
        )

    def test_present_file_is_accepted(self, site, tmp_path):
        make, _ = SITES[site]
        _write(tmp_path, "present.vtp", site=site)
        with in_case_directory(tmp_path):
            assert _errors(make("present.vtp")) == []

    def test_validate_input_resolves_against_the_working_directory(
        self, site, tmp_path, monkeypatch
    ):
        make, _ = SITES[site]
        _write(tmp_path, "present.vtp", site=site)
        monkeypatch.chdir(tmp_path)
        assert validate_zcfd_input(make("present.vtp")) is not None

    def test_a_dict_validated_outside_validate_input_skips_the_check(self, site):
        make, _ = SITES[site]
        assert _errors(make("absent.vtp")) == []

    def test_no_case_directory_skips_the_check(self, site):
        make, _ = SITES[site]
        with in_case_directory(None):
            assert _errors(make("absent.vtp")) == []


def test_absolute_path_message_has_no_directory(tmp_path):
    missing = str(tmp_path / "absent.vtp")
    make, _ = SITES["canopy height"]
    with in_case_directory(tmp_path):
        (error,) = _errors(make(missing))
    assert error["display_message"] == f'Canopy height field "{missing}" not found'


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads any file"
)
def test_unreadable_file_is_rejected(tmp_path):
    path = _write(tmp_path, "locked.vtp")
    path.chmod(0)
    make, _ = SITES["canopy height"]
    try:
        with in_case_directory(tmp_path):
            (error,) = _errors(make("locked.vtp"))
    finally:
        path.chmod(0o644)
    assert error["display_message"] == 'Canopy height field "locked.vtp" cannot be read'


def _deck_in_case_directory(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    deck = _with_wall(
        {**WALL_FUNCTION, "roughness": {"type": "length", "field": "rough.vtp"}}
    )
    (case / "deck.py").write_text(f"parameters = {deck!r}\n")
    return case


def test_a_deck_file_resolves_paths_against_the_working_directory(
    tmp_path, monkeypatch
):
    case = _deck_in_case_directory(tmp_path)
    _write(tmp_path, "rough.vtp")
    monkeypatch.chdir(tmp_path)
    assert validate_zcfd_input(str(case / "deck.py")) is not None


def test_a_file_beside_the_deck_but_not_in_the_working_directory_is_rejected(
    tmp_path, monkeypatch
):
    case = _deck_in_case_directory(tmp_path)
    _write(case, "rough.vtp")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception, match=f'"rough.vtp" not found in {tmp_path}'):
        validate_zcfd_input(str(case / "deck.py"))


def test_the_json_cli_resolves_against_the_working_directory(
    tmp_path, monkeypatch, capsys
):
    import json

    from zutil.validate.cli import _run_json

    case = _deck_in_case_directory(tmp_path)
    _write(case, "rough.vtp")
    monkeypatch.chdir(tmp_path)
    assert _run_json(case / "deck.py") == 1
    (error,) = json.loads(capsys.readouterr().out)["errors"]
    assert error["display_message"] == (
        f'Wall roughness field "rough.vtp" not found in {tmp_path}'
    )


class TestRoughnessMapHeader:
    def _roughness_errors(self, tmp_path, text):
        _write(tmp_path, "rough.vtp", text)
        deck = _with_wall(
            {**WALL_FUNCTION, "roughness": {"type": "length", "field": "rough.vtp"}}
        )
        with in_case_directory(tmp_path):
            return [e["display_message"] for e in _errors(deck)]

    def test_roughness_array_is_accepted(self, tmp_path):
        assert self._roughness_errors(tmp_path, None) == []

    def test_missing_roughness_array_names_the_arrays_found(self, tmp_path):
        text = ROUGHNESS_VTP.format(name="Height")
        assert self._roughness_errors(tmp_path, text) == [
            'Wall roughness field "rough.vtp" has no point array named '
            '"Roughness"; its point arrays are Height'
        ]

    def test_no_point_arrays(self, tmp_path):
        text = ROUGHNESS_VTP.replace(
            ROUGHNESS_VTP[
                ROUGHNESS_VTP.index("<PointData>") : ROUGHNESS_VTP.index("<Points>")
            ],
            "<PointData/>\n      ",
        )
        assert self._roughness_errors(tmp_path, text) == [
            'Wall roughness field "rough.vtp" has no point array named '
            '"Roughness"; it has no point arrays'
        ]

    def test_not_polydata_is_rejected(self, tmp_path):
        text = ROUGHNESS_VTP.replace('type="PolyData"', 'type="UnstructuredGrid"')
        assert self._roughness_errors(tmp_path, text) == [
            'Wall roughness field "rough.vtp" is not a VTK PolyData (.vtp) file'
        ]

    def test_appended_binary_data_after_the_header_is_not_parsed(self, tmp_path):
        header = ROUGHNESS_VTP.format(name="Roughness").replace("</VTKFile>\n", "")
        path = tmp_path / "rough.vtp"
        path.write_bytes(
            header.encode()
            + b'<AppendedData encoding="raw">_\x00\xff\xfe<PointData>\x00'
            + b"</AppendedData></VTKFile>"
        )
        deck = _with_wall(
            {**WALL_FUNCTION, "roughness": {"type": "length", "field": "rough.vtp"}}
        )
        with in_case_directory(tmp_path):
            assert _errors(deck) == []

    def test_canopy_map_without_a_height_array_is_rejected(self, tmp_path):
        _write(tmp_path, "canopy.vtp", ROUGHNESS_VTP.format(name="CanopyHeight"))
        with in_case_directory(tmp_path):
            errors = [e["display_message"] for e in _errors(_with_canopy("canopy.vtp"))]
        assert errors == [
            'Canopy height field "canopy.vtp" has no point array named '
            '"Height"; its point arrays are CanopyHeight'
        ]

    def test_temperature_map_needs_no_roughness_array(self, tmp_path):
        _write(tmp_path, "t.vtp", ROUGHNESS_VTP.format(name="Temperature"))
        deck = _with_wall({"type": "wall", "temperature": {"field": "t.vtp"}})
        with in_case_directory(tmp_path):
            assert _errors(deck) == []
