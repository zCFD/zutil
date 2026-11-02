"""Converting a variant together with the base decks it takes its parameters from.

These run the real source layer and the real rules on copies of the toy sets
in decks/source and the golden sets in decks/golden. Nothing is written to
disk by ``convert_deck``; the CLI writes ``result.files``.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

from zutil.update_input import cli, source
from zutil.update_input.convert import convert_deck, ordered_v2
from zutil.update_input.model import UNCONVERTED, Value
from zutil.update_input.tests.engine.fakes import (
    FakeSource,
    install_rules,
    install_source,
    rule,
    shocktube_deck,
)

DECKS = Path(__file__).parent / "decks"


@pytest.fixture
def toy(tmp_path) -> Path:
    for name in ("chain", "import_set", "namespace_set"):
        shutil.copytree(DECKS / "source" / name, tmp_path / name)
    return tmp_path


@pytest.fixture
def golden(tmp_path) -> Path:
    shutil.copytree(DECKS / "golden", tmp_path / "golden")
    return tmp_path / "golden"


def names(result) -> list[str]:
    return [p.name for p, _ in result.files]


def models(result) -> set[str]:
    return {
        p[1]
        for r in result.bases + [result]
        for p in r.conversion.v2
        if p[0] == "model"
    }


def test_a_chain_converts_innermost_first(toy):
    result = convert_deck(toy / "chain" / "chain_top.py", ignore_legacy_errors=True)
    assert names(result) == [
        "chain_root_v2.py",
        "chain_root_v2.conversion.md",
        "chain_mid_v2.py",
        "chain_mid_v2.conversion.md",
        "chain_top_v2.py",
        "chain_top_v2.conversion.md",
    ]
    assert [p.parent for p, _ in result.files] == [toy / "chain"] * 6


def test_a_set_shares_the_innermost_stem_as_its_label(toy):
    result = convert_deck(toy / "chain" / "chain_top.py", ignore_legacy_errors=True)
    assert models(result) == {"chain_root"}


def test_a_set_shares_the_mesh_label(toy, monkeypatch, stand_in_mesh):
    stand_in_mesh(toy / "chain" / "rotor.h5")
    monkeypatch.chdir(toy / "chain")
    result = convert_deck(
        toy / "chain" / "chain_top.py", mesh_path="rotor.h5", ignore_legacy_errors=True
    )
    assert models(result) == {"rotor"}
    for r in result.bases + [result]:
        assert r.conversion.v2[("model", "rotor", "mesh")] == Value("rotor.h5")


def test_a_base_elsewhere_gets_the_mesh_relative_to_itself(
    golden, monkeypatch, stand_in_mesh
):
    sub = golden / "shocktube_gts" / "sub"
    sub.mkdir()
    variant = sub / "variant.py"
    variant.write_text(
        LOADER_VARIANT.replace("shocktube_fv_first.py", "../shocktube_fv_first.py")
    )
    stand_in_mesh(sub / "ShockTube.h5")
    monkeypatch.chdir(sub)
    result = convert_deck(variant, mesh_path="ShockTube.h5")
    base = result.bases[0]
    assert (
        base.output_path.resolve()
        == (golden / "shocktube_gts" / "shocktube_fv_first_v2.py").resolve()
    )
    assert base.conversion.v2[("model", "ShockTube", "mesh")] == Value(
        "sub/ShockTube.h5"
    )
    assert result.conversion.v2[("model", "ShockTube", "mesh")] == Value("ShockTube.h5")
    assert result.validated is True
    assert "Wrote ../shocktube_fv_first_v2.py" in result.summary


def test_the_variant_renders_against_its_base(toy):
    result = convert_deck(
        toy / "import_set" / "spread_variant.py", ignore_legacy_errors=True
    )
    assert "import spread_base_v2 as spread_base" in result.text
    assert [b.output_path.name for b in result.bases] == ["spread_base_v2.py"]


@pytest.mark.parametrize(
    "variant",
    [
        "ns_variant.py",
        "ns_variant_shared.py",
        "ns_variant_dict.py",
        "ns_variant_edits.py",
    ],
)
def test_namespace_variants_convert(toy, variant):
    result = convert_deck(toy / "namespace_set" / variant, ignore_legacy_errors=True)
    assert names(result)[0] == "ns_base_v2.py"
    assert "ns_base_v2" in result.text


def test_a_shared_base_is_converted_once(toy, monkeypatch):
    real_related = source.related_decks
    monkeypatch.setattr(source, "related_decks", lambda deck: real_related(deck) * 2)
    result = convert_deck(
        toy / "import_set" / "spread_variant.py", ignore_legacy_errors=True
    )
    assert [b.output_path.name for b in result.bases] == ["spread_base_v2.py"]
    assert names(result).count("spread_base_v2.py") == 1


def test_a_base_failing_the_legacy_validator_alone_is_noted_not_refused(
    golden, stand_in_mesh
):
    case = golden / "shocktube_tf_variant"
    mesh = stand_in_mesh(case / "ShockTube.h5")
    result = convert_deck(case / "shocktube_fv_second_ets_tf.py", mesh_path=mesh)
    base = result.bases[0]
    assert base.output_path.name == "shocktube_fv_second_tf_base_v2.py"
    assert any("rejects this base on its own" in n.text for n in base.conversion.notes)
    assert base.validated is False
    assert "not a whole deck" in base.summary


def write_shocktube_variant(golden: Path, text: str) -> Path:
    case = golden / "shocktube_gts"
    path = case / "tube_variant.py"
    path.write_text(text)
    return path


LOADER_VARIANT = """import zutil

parameters = zutil.get_parameters_from_file("shocktube_fv_first.py")
parameters["report"]["frequency"] = 10
"""

IMPORT_VARIANT = """import shocktube_fv_first

parameters = {**shocktube_fv_first.parameters, "restart": False}
"""


@pytest.mark.parametrize(
    "text", [LOADER_VARIANT, IMPORT_VARIANT], ids=["loader", "import"]
)
def test_a_variant_validates_against_its_unwritten_base(golden, text, stand_in_mesh):
    path = write_shocktube_variant(golden, text)
    mesh = stand_in_mesh(path.parent / "ShockTube.h5")
    result = convert_deck(path, mesh_path=mesh)
    assert result.bases[0].validated is True
    assert result.validated is True
    assert "shocktube_fv_first_v2" in result.text
    assert not (path.parent / "shocktube_fv_first_v2.py").exists()
    assert "shocktube_fv_first_v2" not in sys.modules


def test_the_summary_and_report_list_every_file(golden, stand_in_mesh):
    path = write_shocktube_variant(golden, LOADER_VARIANT)
    mesh = stand_in_mesh(path.parent / "ShockTube.h5")
    result = convert_deck(path, mesh_path=mesh)
    for name in names(result):
        assert f"Wrote {name}" in result.summary
        assert f"`{name}`" in result.report
    assert result.summary.startswith("Converting shocktube_fv_first.py")


@pytest.mark.parametrize(
    "deck, files",
    [
        ("bump_namespace/3DBump_5.py", ["3DBump_base_v2.py", "3DBump_5_v2.py"]),
        (
            "overset_chain_a1/cylinderA1.py",
            ["parameters_base_v2.py", "cylinder_base_v2.py", "cylinderA1_v2.py"],
        ),
    ],
)
def test_golden_sets_convert(golden, deck, files):
    result = convert_deck(golden / deck)
    assert [n for n in names(result) if n.endswith(".py")] == files
    assert len(models(result)) == 1


def test_top_level_blocks_follow_deck_order(tmp_path, monkeypatch):
    fake = install_source(monkeypatch, FakeSource(shocktube_deck()))

    def write(c):
        c.set(("model", c.model, "boundary conditions", "BC_1", "type"), Value("wall"))
        c.set(("solver", "time settings", "type"), Value("steady"))
        c.set(("solver", "equations", "type"), Value("euler"))
        c.set(("model", c.model, "boundary conditions", "BC_1", "zones"), Value([1]))

    install_rules(monkeypatch, rule("write", 1, (), write))
    deck = tmp_path / "shocktube.py"
    deck.write_text("parameters = {}\n")
    convert_deck(deck)
    order = []
    for path in fake.rendered:
        if path[0] not in order:
            order.append(path[0])
    assert order == ["config version", "solver", "model", UNCONVERTED]
    assert [p for p in fake.rendered if p[0] == "solver"] == [
        ("solver", "time settings", "type"),
        ("solver", "equations", "type"),
    ]
    assert [p[-1] for p in fake.rendered if p[0] == "model"] == [
        "mesh",
        "type",
        "zones",
    ]


def test_ordered_v2_puts_other_blocks_before_unconverted():
    v2 = {
        (UNCONVERTED, "a"): Value(1),
        ("other",): Value(2),
        ("model", "m", "mesh"): Value(3),
        ("config version",): Value(2),
    }
    assert [p[0] for p in ordered_v2(v2)] == [
        "config version",
        "model",
        "other",
        UNCONVERTED,
    ]


def test_cli_writes_every_file_of_a_set(toy, capsys):
    deck = toy / "chain" / "chain_top.py"
    assert cli.main([str(deck), "--ignore-legacy-errors"]) == 0
    for name in ("chain_root_v2.py", "chain_mid_v2.py", "chain_top_v2.py"):
        assert (toy / "chain" / name).exists()
        assert (toy / "chain" / name.replace(".py", ".conversion.md")).exists()
    printed = capsys.readouterr().out
    assert "Wrote chain_root_v2.py" in printed


def test_cli_refuses_to_overwrite_a_base_output(toy, capsys):
    deck = toy / "chain" / "chain_top.py"
    (toy / "chain" / "chain_root_v2.py").write_text("keep me")
    assert cli.main([str(deck), "--ignore-legacy-errors"]) == 1
    assert "chain_root_v2.py" in capsys.readouterr().err
    assert (toy / "chain" / "chain_root_v2.py").read_text() == "keep me"
    assert not (toy / "chain" / "chain_top_v2.py").exists()


def test_cli_force_overwrites_a_base_output(toy):
    deck = toy / "chain" / "chain_top.py"
    (toy / "chain" / "chain_root_v2.py").write_text("old")
    assert cli.main([str(deck), "--ignore-legacy-errors", "--force"]) == 0
    assert (toy / "chain" / "chain_root_v2.py").read_text() != "old"


def test_cli_deck_read_error_names_the_file(tmp_path, capsys):
    deck = tmp_path / "broken.py"
    deck.write_text("parameters = {\n")
    assert cli.main([str(deck)]) == 1
    assert "broken.py" in capsys.readouterr().err


def test_cli_render_error_names_the_file(toy, monkeypatch, capsys):
    def refuse(deck, v2, base_v2=None):
        raise source.RenderError(f"{deck.path.name} cannot be written")

    monkeypatch.setattr(source, "render_deck", refuse)
    deck = toy / "chain" / "chain_top.py"
    assert cli.main([str(deck), "--ignore-legacy-errors"]) == 1
    err = capsys.readouterr().err
    assert "chain_top.py" in err and "cannot be written" in err


def test_a_base_leaves_keys_it_cannot_place_to_its_variant(tmp_path):
    base = tmp_path / "base.py"
    base.write_text(
        "parameters = {\n"
        '    "mapping": {"number of additional overset active layers": 1},\n'
        '    "BC_1": {"zone": [1], "type": "wall"},\n'
        "}\n"
    )
    variant = tmp_path / "rotor.py"
    variant.write_text(
        "import zutil\n"
        'parameters = zutil.get_parameters_from_file("base.py", namespace=globals())\n'
        'parameters["BC_2"] = {"zone": [2], "type": "overset"}\n'
    )
    result = convert_deck(variant, ignore_legacy_errors=True)
    [base_result] = result.bases
    assert "unconverted v1 keys" not in base_result.text
    assert "del " not in result.text
    assert "additional active layers" in result.text


def test_a_base_numbers_its_decisions_without_the_keys_left_to_its_variant(tmp_path):
    base = tmp_path / "base.py"
    base.write_text(
        "parameters = {\n"
        '    "mapping": {"number of additional overset active layers": 1},\n'
        '    "BC_1": {"zone": [1], "type": "wall"},\n'
        '    "FZ_1": {"type": "abl", "zone": [1]},\n'
        "}\n"
    )
    variant = tmp_path / "rotor.py"
    variant.write_text(
        "import zutil\n"
        'parameters = zutil.get_parameters_from_file("base.py", namespace=globals())\n'
        'parameters["BC_2"] = {"zone": [2], "type": "overset"}\n'
    )
    result = convert_deck(variant, ignore_legacy_errors=True)
    [base_result] = result.bases
    assert [d.id for d in base_result.conversion.decisions] == [1, 2]
    assert re.findall(r"item \d+", base_result.text) == ["item 1", "item 2"]
    assert re.findall(r"### \d+\.", base_result.report) == ["### 1.", "### 2."]


def test_a_three_level_rotor_chain_converts_and_validates(tmp_path):
    """The shape of a multi-rotor overset sub-deck.

    rotor_1 supplies ``rotor_number``, without which rotor_base does not run;
    it sets a force report with a transform function and adds the overset BC that
    places base's additional active layers.
    """
    from zutil.validate import validate_zcfd_input

    work = tmp_path / "rotor_chain"
    shutil.copytree(DECKS / "sets" / "rotor_chain", work)
    result = convert_deck(work / "rotor_1.py")

    texts = {path.name: text for path, text in result.files if path.suffix == ".py"}
    assert set(texts) == {"base_v2.py", "rotor_base_v2.py", "rotor_1_v2.py"}
    assert all("del " not in text for text in texts.values())
    assert UNCONVERTED not in texts["base_v2.py"]
    assert UNCONVERTED not in texts["rotor_base_v2.py"]
    assert 'get_parameters_from_file("rotor_base_v2.py"' in texts["rotor_1_v2.py"]
    assert '"transform": rotor_transform' in texts["rotor_1_v2.py"]
    assert '"additional active layers": 1' in texts["rotor_1_v2.py"]
    assert [d.question for d in result.conversion.decisions] == [
        "Which mesh file does this deck run on?"
    ]

    for path, text in result.files:
        path.write_text(text)
    base = work / "base_v2.py"
    base.write_text(re.sub(r'"DECIDE: [^"]*"', '"rotor.h5"', base.read_text()))
    validate_zcfd_input(str(work / "rotor_1_v2.py"))


def test_powered_restart_matches_its_checked_in_conversion(
    golden, monkeypatch, stand_in_mesh
):
    """The v2 deck is compared as a syntax tree, so ruff's layout does not matter."""
    import ast

    case = golden / "powered_restart"
    expected = (
        Path(__file__).parent / "decks" / "golden" / "powered_restart" / "expected"
    )
    stand_in_mesh(case / "wingbody.h5")
    monkeypatch.chdir(case)
    result = convert_deck(case / "wingbody_powered.py", mesh_path="wingbody.h5")
    assert result.conversion.decisions == []
    assert result.validated is True
    assert ast.dump(ast.parse(result.text)) == ast.dump(
        ast.parse((expected / "wingbody_powered_v2.py").read_text())
    )
    assert result.report == (expected / "wingbody_powered_v2.conversion.md").read_text()


def test_driven_condition_matches_its_checked_in_conversion(
    golden, monkeypatch, stand_in_mesh
):
    """The hand-written lift controller becomes one item naming the farfield target."""
    import ast

    case = golden / "naca_ic_callable"
    expected = (
        Path(__file__).parent / "decks" / "golden" / "naca_ic_callable" / "expected"
    )
    stand_in_mesh(case / "naca0012_0257.h5")
    monkeypatch.chdir(case)
    result = convert_deck(
        case / "n0012_257_cL_target_1_imp.py", mesh_path="naca0012_0257.h5"
    )
    [decision] = result.conversion.decisions
    assert decision.v2_path == ("solver", "reference conditions", "IC_2")
    assert 'give BC_3 a target, such as "target": {"alpha": {"cl": ...}}' in (
        decision.details
    )
    assert result.validated is False
    assert ast.dump(ast.parse(result.text)) == ast.dump(
        ast.parse((expected / "n0012_257_cL_target_1_imp_v2.py").read_text())
    )
    assert (
        result.report
        == (expected / "n0012_257_cL_target_1_imp_v2.conversion.md").read_text()
    )
