"""Converting an overset group: one ``override`` deck naming (mesh, deck) pairs.

Each pair becomes one model of a single config version 2 deck. Settings every
model shares sit once under ``solver``; a setting that differs becomes a
per-model override where v2 allows one, and a decision where it does not.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from zutil.update_input import cli
from zutil.update_input.convert import LegacyDeckError, convert_deck
from zutil.update_input.model import UNCONVERTED
from zutil.update_input.source import DeckReadError

DECKS = Path(__file__).parent / "decks"


@pytest.fixture
def groups(tmp_path) -> Path:
    shutil.copytree(DECKS / "groups", tmp_path / "groups")
    return tmp_path / "groups"


def convert_pair(groups: Path, run_deck):
    result = convert_deck(groups / "pair" / "override.py")
    return result, run_deck(result.text, groups / "pair")


def test_each_pair_becomes_a_model_named_after_its_deck(groups, run_deck):
    result, parameters = convert_pair(groups, run_deck)
    assert list(parameters["model"]) == ["background", "rotor"]
    assert parameters["model"]["background"]["mesh"] == "background.h5"
    assert parameters["model"]["rotor"]["mesh"] == "rotor.h5"
    assert result.output_path == groups / "pair" / "override_v2.py"


def test_shared_blocks_sit_once_in_solver(groups, run_deck):
    _, parameters = convert_pair(groups, run_deck)
    solver = parameters["solver"]
    assert solver["time settings"] == {
        "type": "unsteady",
        "total time": 1.0,
        "time step": 0.01,
        "kind": "dual time stepping",
        "order": 2,
        "start": 0,
    }
    assert solver["equations"] == {"type": "euler"}
    assert solver["reference conditions"]["IC_1"]["v"] == {
        "vector": [1.0, 0.0, 0.0],
        "mach": 0.2,
    }
    for model in parameters["model"].values():
        assert set(model) <= {
            "mesh",
            "boundary conditions",
            "fluid zones",
            "transforms",
            "output settings",
        }


def test_a_block_that_differs_is_written_whole_in_every_model(groups, run_deck):
    _, parameters = convert_pair(groups, run_deck)
    assert parameters["solver"]["output settings"] == {
        "solution": {
            "format": "vtk",
            "surface variables": ["p"],
            "volume variables": ["p"],
        },
        "report": {"frequency": 10},
    }
    assert parameters["model"]["rotor"]["output settings"] == {
        "solution": {
            "format": "vtk",
            "surface variables": ["p"],
            "volume variables": ["p", "overset"],
        },
        "report": {
            "frequency": 10,
            "forces": {"FR_1": {"name": "rotor", "zones": [8]}},
        },
    }
    assert (
        parameters["model"]["background"]["output settings"]
        == parameters["solver"]["output settings"]
    )


def test_the_most_common_block_goes_to_solver(groups, run_deck):
    deck = write_group(
        groups / "pair",
        '[("a.h5", "rotor.py"), ("b.h5", "background.py"), ("c.h5", "background.py")]',
    )
    parameters = run_deck(convert_deck(deck).text, groups / "pair")
    solution = parameters["solver"]["output settings"]["solution"]
    assert solution["volume variables"] == ["p"]
    for label in ("rotor", "background_1", "background_2"):
        assert "output settings" in parameters["model"][label]
    assert (
        parameters["model"]["background_1"]["output settings"]["solution"] == solution
    )


def test_a_block_every_model_shares_sits_in_solver_alone(groups, run_deck):
    _, parameters = convert_pair(groups, run_deck)
    assert "convergence control" in parameters["solver"]
    for model in parameters["model"].values():
        for block in ("initialisation", "convergence control", "numerical scheme"):
            assert block not in model


def test_the_overset_layers_go_to_the_models_with_an_overset_interface(
    groups, run_deck
):
    result, parameters = convert_pair(groups, run_deck)
    rotor_bc = parameters["model"]["rotor"]["boundary conditions"]["BC_1"]
    assert rotor_bc["additional active layers"] == 2
    assert UNCONVERTED not in parameters
    [layers] = [
        d
        for d in result.conversion.dispositions
        if d.v1_path
        == ("background", "mapping", "number of additional overset active layers")
    ]
    assert layers.kind == "dropped"
    assert "background has no overset interface" in layers.reason


def test_moving_mesh_becomes_the_motion_of_the_model_with_the_moving_zone(
    groups, run_deck
):
    _, parameters = convert_pair(groups, run_deck)
    rotor = parameters["model"]["rotor"]
    motion = rotor["transforms"]["motion"]
    assert motion["type"] == "rotating"
    assert motion["moving mesh"] is True
    assert motion["omega"] == pytest.approx(62.83185307179586)
    assert "zones" not in motion
    assert "fluid zones" not in rotor
    assert "transforms" not in parameters["model"]["background"]
    assert "fluid zones" not in parameters["model"]["background"]


def test_a_group_with_no_open_decisions_validates(groups, run_deck):
    from zutil.validate import validate_zcfd_input

    result, _ = convert_pair(groups, run_deck)
    assert result.conversion.decisions == []
    assert result.validated is True
    path = groups / "pair" / "override_v2.py"
    path.write_text(result.text)
    validate_zcfd_input(str(path))


def test_the_group_writes_one_deck_and_one_report(groups, run_deck):
    result, _ = convert_pair(groups, run_deck)
    assert [p.name for p, _ in result.files] == [
        "override_v2.py",
        "override_v2.conversion.md",
    ]
    assert result.bases == []
    assert {p.name for p in result.members} == {
        "background.py",
        "rotor.py",
        "shared.py",
    }


def test_models_that_disagree_on_a_solver_only_setting_get_a_decision(groups, run_deck):
    result = convert_deck(groups / "disagree" / "override.py")
    parameters = run_deck(result.text, groups / "disagree")
    total = parameters["solver"]["time settings"]["total time"]
    assert total.startswith("DECIDE: ")
    [decision] = [
        d
        for d in result.conversion.decisions
        if d.v2_path == ("solver", "time settings", "total time")
    ]
    assert decision.options == ("1.0", "2.0")
    assert "outer" in decision.details and "inner" in decision.details
    assert sorted(decision.v1_paths) == [
        ("inner", "time marching", "unsteady", "total time"),
        ("outer", "time marching", "unsteady", "total time"),
    ]
    temperature = parameters["solver"]["reference conditions"]["IC_1"]["temperature"]
    assert temperature.startswith("DECIDE: ")
    assert result.validated is False


def test_a_named_entry_only_one_model_defines_is_shared_without_a_decision(
    groups, run_deck
):
    result = convert_deck(groups / "disagree" / "override.py")
    parameters = run_deck(result.text, groups / "disagree")
    assert parameters["solver"]["reference conditions"]["IC_2"]["temperature"] == 300.0
    assert not [d for d in result.conversion.decisions if "IC_2" in d.v2_path]


def test_a_self_contained_function_is_copied_into_the_group_deck(groups, run_deck):
    result = convert_deck(groups / "functions" / "override.py")
    assert "import math\n" in result.text
    assert "def tilt(*, vector):" in result.text
    parameters = run_deck(result.text, groups / "functions")
    report = parameters["model"]["rotor_a"]["output settings"]["report"]
    tilt = report["forces"]["FR_2"]["transform"]
    assert tilt(vector=(1.0, 0.0, 0.0))[1] == pytest.approx(0.0998334166)


def test_a_function_that_reads_its_deck_globals_is_a_decision(groups):
    result = convert_deck(groups / "functions" / "override.py")
    assert "def place(" not in result.text
    path = (
        "model",
        "rotor_b",
        "output settings",
        "report",
        "forces",
        "FR_2",
        "transform",
    )
    [decision] = [d for d in result.conversion.decisions if d.v2_path == path]
    assert "offset" in decision.details
    assert "rotor_b.py" in decision.details
    assert decision.v1_paths == (("rotor b", "report", "forces", "fr 2", "transform"),)


def write_group(where: Path, pairs: str, extra: str = "") -> Path:
    path = where / "override.py"
    path.write_text(f'override = {{"mesh_case_pair": {pairs}{extra}}}\n')
    return path


def test_a_deck_listed_twice_gets_numbered_labels(groups, run_deck):
    deck = write_group(
        groups / "pair",
        '[("background.h5", "background.py"), ("a.h5", "rotor.py"), ("b.h5", "rotor.py")]',
    )
    result = convert_deck(deck)
    parameters = run_deck(result.text, groups / "pair")
    assert list(parameters["model"]) == ["background", "rotor_1", "rotor_2"]
    assert parameters["model"]["rotor_2"]["mesh"] == "b.h5"
    assert any(
        "rotor_1" in n.text and "rotor_2" in n.text for n in result.conversion.notes
    )


def test_names_without_extensions_get_them(groups, run_deck):
    deck = write_group(
        groups / "pair", '[("background", "background"), ("rotor", "rotor")]'
    )
    parameters = run_deck(convert_deck(deck).text, groups / "pair")
    assert parameters["model"]["background"]["mesh"] == "background.h5"
    assert list(parameters["model"]) == ["background", "rotor"]


def test_a_member_failing_the_legacy_validator_refuses_the_group(groups):
    bad = groups / "pair" / "rotor.py"
    bad.write_text(
        bad.read_text() + 'parameters["BC_3"] = {"zone": [9], "type": "farfield"}\n'
    )
    with pytest.raises(LegacyDeckError) as raised:
        convert_deck(groups / "pair" / "override.py")
    assert "rotor.py" in str(raised.value)
    result = convert_deck(groups / "pair" / "override.py", ignore_legacy_errors=True)
    assert any(
        n.text.startswith("rotor: The legacy validator rejected")
        for n in result.conversion.notes
    )


@pytest.mark.parametrize(
    "text, words",
    [
        ('override = {"mesh_case_pair": "a.py"}\n', "a list of (mesh, deck) pairs"),
        ('override = {"mesh_case_pair": []}\n', "names no (mesh, deck) pairs"),
        ('override = {"mesh_case_pair": [("a.h5",)]}\n', "pair 1"),
        ('override = {"mesh_case_pair": [("a.h5", "missing.py")]}\n', "missing.py"),
    ],
)
def test_a_malformed_group_is_refused(tmp_path, text, words):
    deck = tmp_path / "override.py"
    deck.write_text(text)
    with pytest.raises(DeckReadError) as raised:
        convert_deck(deck)
    assert words in str(raised.value)


def test_other_group_keys_are_left_as_decisions(groups, run_deck):
    deck = write_group(
        groups / "pair",
        '[("background.h5", "background.py"), ("rotor.h5", "rotor.py")]',
        ', "device": "gpu"',
    )
    result = convert_deck(deck)
    parameters = run_deck(result.text, groups / "pair")
    assert parameters[UNCONVERTED] == {"override > device": "gpu"}
    [decision] = result.conversion.decisions
    assert decision.v1_paths == (("override", "device"),)


def test_the_report_names_each_model_its_deck_and_its_mesh(groups, run_deck):
    result, _ = convert_pair(groups, run_deck)
    assert "Model background is background.py on background.h5." in result.report
    assert "Model rotor is rotor.py on rotor.h5." in result.report
    assert (
        "| `rotor > time marching > moving mesh` | re-spelled | "
        "`model > rotor > transforms > motion > moving mesh` |"
    ) in result.report
    assert result.summary.startswith("Converting override.py")
    assert "Wrote override_v2.py (valid config version 2)" in result.summary


def test_a_mesh_beside_the_group_fills_zones_from_ref_codes(groups, run_deck):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    with h5py.File(groups / "pair" / "background.h5", "w") as f:
        g = f.create_group("mesh")
        g.create_dataset("faceBC", data=np.array([0, 7, 7]))
        g.create_dataset("faceInfo", data=np.array([[0, 0], [5, 0], [6, 0]]))
    background = groups / "pair" / "background.py"
    background.write_text(
        background.read_text().replace(
            '{"zone": [3], "type": "symmetry"}', '{"ref": 7, "type": "symmetry"}'
        )
    )
    parameters = run_deck(
        convert_deck(groups / "pair" / "override.py").text, groups / "pair"
    )
    bc = parameters["model"]["background"]["boundary conditions"]["BC_2"]
    assert bc["zones"] == [5, 6]


def test_cli_converts_a_group(groups, capsys):
    deck = groups / "pair" / "override.py"
    assert cli.main([str(deck)]) == 0
    assert (groups / "pair" / "override_v2.py").exists()
    assert (groups / "pair" / "override_v2.conversion.md").exists()
    assert not (groups / "pair" / "rotor_v2.py").exists()
    assert "Wrote override_v2.py" in capsys.readouterr().out


def test_cli_refuses_a_mesh_for_a_group(groups, capsys):
    deck = groups / "pair" / "override.py"
    assert cli.main([str(deck), "--mesh", "rotor.h5"]) == cli.EXIT_USAGE
    assert "names its own meshes" in capsys.readouterr().err


def test_cli_refuses_to_write_over_a_member_deck(groups, capsys):
    deck = groups / "pair" / "override.py"
    out = groups / "pair" / "rotor.py"
    before = out.read_text()
    assert cli.main([str(deck), "--output", str(out), "--force"]) == cli.EXIT_USAGE
    assert "overwrite an input deck" in capsys.readouterr().err
    assert out.read_text() == before


def test_one_function_every_model_shares_is_one_decision(groups):
    result = convert_deck(groups / "functions" / "override.py")
    [decision] = [
        d for d in result.conversion.decisions if "replaces turn" in d.question
    ]
    assert {p[0] for p in decision.v1_paths} == {"tunnel", "rotor a", "rotor b"}
    assert result.text.count(f"item {decision.id})") == 4


def test_incompressible_models_may_differ_in_convergence_control(groups, run_deck):
    result = convert_deck(groups / "incomp" / "override.py")
    parameters = run_deck(result.text, groups / "incomp")
    assert result.conversion.decisions == []
    assert result.validated is True
    assert parameters["solver"]["convergence control"]["scheme"] == {
        "implicit relax": 0.5
    }
    inner = parameters["model"]["inner"]["convergence control"]
    assert inner["cycles"] == 9
    assert "scheme" not in inner
    assert "scheme" not in parameters["model"]["outer"]["convergence control"]


def test_incompressible_models_that_disagree_on_the_scheme_get_a_decision(
    groups, run_deck
):
    inner = groups / "incomp" / "inner.py"
    inner.write_text(
        inner.read_text() + 'parameters["incomp viscous"]["relax"] = 0.3\n'
    )
    result = convert_deck(groups / "incomp" / "override.py")
    parameters = run_deck(result.text, groups / "incomp")
    scheme = ("solver", "convergence control", "scheme")
    [decision] = [d for d in result.conversion.decisions if d.v2_path == scheme]
    assert "outer" in decision.details and "inner" in decision.details
    assert parameters["solver"]["convergence control"]["scheme"].startswith("DECIDE: ")
    for model in parameters["model"].values():
        assert "scheme" not in model["convergence control"]


def test_the_solver_block_holds_only_keys_every_differing_model_sets(groups, run_deck):
    deck = write_group(
        groups / "pair",
        '[("rotor.h5", "rotor.py"), ("background.h5", "background.py")]',
    )
    parameters = run_deck(convert_deck(deck).text, groups / "pair")
    solver = parameters["solver"]["output settings"]
    assert "forces" not in solver["report"]
    assert solver["solution"]["volume variables"] == ["p", "overset"]
    assert (
        "forces" not in parameters["model"]["background"]["output settings"]["report"]
    )
    assert parameters["model"]["rotor"]["output settings"]["report"]["forces"] == {
        "FR_1": {"name": "rotor", "zones": [8]}
    }


def test_a_positional_function_in_a_shared_base_is_rewritten_and_copied_once(
    groups, run_deck
):
    base = groups / "functions" / "base.py"
    base.write_text(
        base.read_text().replace(
            "def turn(*, vector):\n    return (vector[0] * scale, vector[1], vector[2])",
            "def turn(x, y, z):\n    return (2.0 * x, y, z)",
        )
    )
    result = convert_deck(groups / "functions" / "override.py")
    assert result.text.count("def turn(*, vector):") == 1
    assert "x, y, z = vector" in result.text
    assert not [d for d in result.conversion.decisions if "turn" in d.question]
    parameters = run_deck(result.text, groups / "functions")
    turn = parameters["solver"]["output settings"]["report"]["forces"]["FR_1"]
    assert turn["transform"](vector=(1.0, 2.0, 3.0)) == (2.0, 2.0, 3.0)
    assert "def turn(x, y, z):" in base.read_text()


def test_a_base_function_with_no_mechanical_rewrite_is_one_decision_to_copy_it(
    groups,
):
    base = groups / "functions" / "base.py"
    base.write_text(
        base.read_text().replace(
            "def turn(*, vector):\n    return (vector[0] * scale, vector[1], vector[2])",
            "def turn(x, y):\n    return (2.0 * x, y, 0.0)",
        )
    )
    result = convert_deck(groups / "functions" / "override.py")
    [decision] = [d for d in result.conversion.decisions if "turn" in d.question]
    assert decision.question.startswith("rewrite turn as def turn(*, vector)")
    assert {p[0] for p in decision.v1_paths} == {"tunnel", "rotor a", "rotor b"}
    assert "copy the function into override_v2.py" in decision.details
    assert "\ndef turn(" not in result.text
    assert result.text.count(f"item {decision.id})") == 4


@pytest.mark.parametrize(
    "deck, written",
    [
        ("pair/override.py", "override_v2.py"),
        ("pair/background.py", "background_v2.py"),
    ],
)
def test_cli_converts_a_deck_named_relative_to_a_parent_directory(
    groups, monkeypatch, deck, written
):
    monkeypatch.chdir(groups)
    assert cli.main([deck]) == 0
    assert (groups / "pair" / written).exists()


def test_a_model_mesh_is_the_name_the_group_gives_it(groups, run_deck):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    sub = groups / "pair" / "decks"
    sub.mkdir()
    for name in ("background.py", "rotor.py", "shared.py"):
        shutil.copy(groups / "pair" / name, sub / name)
    with h5py.File(groups / "pair" / "background.h5", "w") as f:
        g = f.create_group("mesh")
        g.create_dataset("faceBC", data=np.array([0, 3, 3]))
        g.create_dataset("faceInfo", data=np.array([[0, 0], [1, 0], [3, 0]]))
    deck = write_group(
        groups / "pair",
        '[("background.h5", "decks/background.py"), ("rotor.h5", "decks/rotor.py")]',
    )
    parameters = run_deck(convert_deck(deck).text, groups / "pair")
    assert parameters["model"]["background"]["mesh"] == "background.h5"
    assert parameters["model"]["rotor"]["mesh"] == "rotor.h5"


def test_a_group_written_elsewhere_names_its_meshes_from_the_new_deck(groups):
    out = groups / "pair" / "v2" / "override_v2.py"
    result = convert_deck(groups / "pair" / "override.py", output_path=out)
    meshes = {p[1]: v.value for p, v in result.conversion.v2.items() if p[-1] == "mesh"}
    assert meshes == {
        "background": (groups / "pair" / "background.h5").as_posix(),
        "rotor": (groups / "pair" / "rotor.h5").as_posix(),
    }
    deck = write_group(
        groups / "pair",
        '[("v2/background.h5", "background.py"), ("rotor.h5", "rotor.py")]',
    )
    result = convert_deck(deck, output_path=out)
    meshes = {p[1]: v.value for p, v in result.conversion.v2.items() if p[-1] == "mesh"}
    assert meshes["background"] == "background.h5"
    assert meshes["rotor"] == (groups / "pair" / "rotor.h5").as_posix()


def test_a_group_with_legacy_errors_is_a_draft_whatever_its_notes_say(
    groups, monkeypatch
):
    from zutil.update_input.model import Conversion

    note = Conversion.note
    monkeypatch.setattr(
        Conversion,
        "note",
        lambda self, text, *args: note(self, text.replace("legacy", "old"), *args),
    )
    rotor = groups / "pair" / "rotor.py"
    rotor.write_text(rotor.read_text() + 'parameters["report"]["frequency"] = "ten"\n')
    result = convert_deck(groups / "pair" / "override.py", ignore_legacy_errors=True)
    assert result.validated is False
    assert "as a v1 deck had legacy errors" in result.summary
