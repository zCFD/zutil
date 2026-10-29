"""The conversion engine, run against fake source and rule layers."""

from __future__ import annotations

import pytest

from zutil.update_input.convert import (
    UNCONVERTED,
    ConverterBugError,
    LegacyDeckError,
    convert_deck,
)
from zutil.update_input.model import Placeholder, Source, V1Deck, Value
from zutil.update_input.tests.engine.fakes import (
    FakeSource,
    complete_shocktube_rule,
    flatten,
    install_rules,
    install_source,
    rule,
    shocktube_deck,
)


@pytest.fixture
def deck_path(tmp_path):
    path = tmp_path / "shocktube.py"
    path.write_text("parameters = {}\n")
    return path


@pytest.fixture
def source(monkeypatch):
    return install_source(monkeypatch, FakeSource(shocktube_deck()))


def test_reads_the_deck_through_the_source_layer(deck_path, source, monkeypatch):
    install_rules(monkeypatch)
    convert_deck(deck_path)
    assert source.read_paths == [deck_path]


def test_legacy_errors_refuse_the_deck(deck_path, monkeypatch):
    evaluated = {"equations": "euler"}
    install_source(monkeypatch, FakeSource(V1Deck(flatten(evaluated), evaluated)))
    install_rules(monkeypatch)
    with pytest.raises(LegacyDeckError) as raised:
        convert_deck(deck_path)
    assert any("time marching" in e for e in raised.value.errors)
    assert "shocktube.py" in str(raised.value)
    assert "--ignore-legacy-errors" in str(raised.value)


def test_legacy_error_text_carries_no_terminal_colour(deck_path, monkeypatch):
    evaluated = {"equations": "euler"}
    install_source(monkeypatch, FakeSource(V1Deck(flatten(evaluated), evaluated)))
    install_rules(monkeypatch)
    with pytest.raises(LegacyDeckError) as raised:
        convert_deck(deck_path)
    assert all("\x1b" not in e for e in raised.value.errors)


def test_legacy_errors_name_the_key_path_with_arrows(deck_path, monkeypatch):
    evaluated = dict(shocktube_deck().evaluated)
    evaluated["air"] = dict(evaluated["air"], **{"bogus key": 1})
    install_source(monkeypatch, FakeSource(V1Deck(flatten(evaluated), evaluated)))
    install_rules(monkeypatch)
    with pytest.raises(LegacyDeckError) as raised:
        convert_deck(deck_path)
    assert any(e.startswith("air > bogus key") for e in raised.value.errors)


def test_ignore_legacy_errors_turns_them_into_notes(deck_path, monkeypatch):
    evaluated = {"equations": "euler"}
    install_source(monkeypatch, FakeSource(V1Deck(flatten(evaluated), evaluated)))
    install_rules(monkeypatch)
    result = convert_deck(deck_path, ignore_legacy_errors=True)
    notes = [n.text for n in result.conversion.notes]
    assert any("time marching" in n for n in notes)


def _with(evaluated: dict, extra: dict) -> dict:
    out = dict(evaluated)
    for key, value in extra.items():
        out[key] = (
            {**out[key], **value} if isinstance(value, dict) and key in out else value
        )
    return out


@pytest.mark.parametrize(
    "extra, key",
    [
        ({"restart": True, "restart casename": "elsewhere"}, "restart casename"),
        ({"write output": {"volume interpolate": ["gone.vtp"]}}, "volume interpolate"),
    ],
)
def test_a_missing_file_is_a_note_and_does_not_refuse(
    deck_path, monkeypatch, extra, key
):
    evaluated = _with(shocktube_deck().evaluated, extra)
    install_source(monkeypatch, FakeSource(V1Deck(flatten(evaluated), evaluated)))
    install_rules(monkeypatch)
    result = convert_deck(deck_path)
    assert result.conversion.legacy_errors == ()
    notes = [n.text for n in result.conversion.notes if key in n.text]
    assert any("File does not exist" in n for n in notes)
    assert not any("rejected" in n for n in notes)


def test_a_missing_file_does_not_excuse_another_legacy_error(deck_path, monkeypatch):
    evaluated = _with(
        shocktube_deck().evaluated,
        {"restart": True, "restart casename": "elsewhere", "air": {"bogus key": 1}},
    )
    install_source(monkeypatch, FakeSource(V1Deck(flatten(evaluated), evaluated)))
    install_rules(monkeypatch)
    with pytest.raises(LegacyDeckError) as raised:
        convert_deck(deck_path)
    assert any(e.startswith("air > bogus key") for e in raised.value.errors)
    assert not any("restart casename" in e for e in raised.value.errors)


def test_legacy_validator_does_not_change_the_evaluated_dict(
    deck_path, source, monkeypatch
):
    install_rules(monkeypatch)
    before = repr(source.deck.evaluated)
    convert_deck(deck_path)
    assert repr(source.deck.evaluated) == before


def test_model_label_is_the_deck_stem_without_a_mesh(deck_path, source, monkeypatch):
    seen = []
    install_rules(monkeypatch, rule("spy", 1, (), lambda c: seen.append(c.model)))
    convert_deck(deck_path)
    assert seen == ["shocktube"]


def test_mesh_is_a_decision_without_a_mesh(deck_path, source, monkeypatch):
    install_rules(monkeypatch)
    result = convert_deck(deck_path)
    placeholder = source.rendered[("model", "shocktube", "mesh")]
    assert isinstance(placeholder, Placeholder)
    decision = result.conversion.decisions[placeholder.decision_id - 1]
    assert "--mesh" in decision.details


def _stand_in_mesh(monkeypatch, path):
    from zutil.update_input.model import MeshInfo

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    monkeypatch.setattr(
        "zutil.update_input.convert.read_mesh",
        lambda p: MeshInfo(str(p), "ShockTube", {9: [12, 13]}),
    )
    return path


def test_mesh_sets_the_label_and_the_mesh_value(deck_path, source, monkeypatch):
    _stand_in_mesh(monkeypatch, deck_path.parent / "meshes" / "ShockTube.h5")
    monkeypatch.chdir(deck_path.parent)
    seen = []
    install_rules(monkeypatch, rule("spy", 1, (), lambda c: seen.append(c.mesh)))
    convert_deck(deck_path, mesh_path="meshes/ShockTube.h5")
    assert seen[0].label == "ShockTube"
    assert seen[0].zones_by_type == {9: [12, 13]}
    assert source.rendered[("model", "ShockTube", "mesh")] == Value(
        "meshes/ShockTube.h5"
    )


def test_an_absolute_mesh_path_stays_absolute(deck_path, source, monkeypatch):
    mesh = _stand_in_mesh(monkeypatch, deck_path.parent.parent / "m" / "ShockTube.h5")
    install_rules(monkeypatch)
    convert_deck(deck_path, mesh_path=str(mesh))
    assert source.rendered[("model", "ShockTube", "mesh")] == Value(mesh.as_posix())


def test_a_relative_mesh_path_is_taken_from_the_output_deck(
    deck_path, source, monkeypatch, tmp_path
):
    _stand_in_mesh(monkeypatch, tmp_path / "runs" / "meshes" / "ShockTube.h5")
    monkeypatch.chdir(tmp_path)
    install_rules(monkeypatch)
    convert_deck(
        deck_path,
        mesh_path="runs/meshes/ShockTube.h5",
        output_path=tmp_path / "runs" / "shocktube_v2.py",
    )
    assert source.rendered[("model", "ShockTube", "mesh")] == Value(
        "meshes/ShockTube.h5"
    )


def test_a_relative_mesh_path_outside_the_output_directory_is_written_absolute(
    deck_path, source, monkeypatch, tmp_path
):
    work = tmp_path / "work"
    work.mkdir()
    mesh = _stand_in_mesh(monkeypatch, tmp_path / "meshes" / "ShockTube.h5")
    monkeypatch.chdir(work)
    install_rules(monkeypatch)
    convert_deck(
        deck_path,
        mesh_path="../meshes/ShockTube.h5",
        output_path=work / "shocktube_v2.py",
    )
    assert source.rendered[("model", "ShockTube", "mesh")] == Value(mesh.as_posix())


def test_rules_run_in_the_order_rule_modules_gives(deck_path, source, monkeypatch):
    calls = []
    install_rules(
        monkeypatch,
        rule("a", 1, (), lambda c: calls.append("a")),
        rule("b", 2, (), lambda c: calls.append("b")),
    )
    convert_deck(deck_path)
    assert calls == ["a", "b"]


def test_unclaimed_leaves_become_decisions_with_their_source_kept(
    deck_path, source, monkeypatch
):
    install_rules(monkeypatch)
    result = convert_deck(deck_path)
    c = result.conversion
    assert c.unclaimed() == []
    kept = source.rendered[(UNCONVERTED, "BC_1 > zone")]
    assert isinstance(kept, Source)
    assert kept.leaf.source == "sym"
    questions = [d.question for d in c.decisions]
    assert "update_input has no rule for 'BC_1 > zone'" in questions
    assert "'unconverted v1 keys': {" in result.text


def test_claimed_leaves_do_not_reach_unconverted(deck_path, source, monkeypatch):
    def claim_units(c):
        c.drop(c.v1.leaf("units"), "the solver is always SI")

    install_rules(monkeypatch, rule("units", 1, ("units",), claim_units))
    convert_deck(deck_path)
    assert (UNCONVERTED, "units") not in source.rendered


def test_config_version_is_two(deck_path, source, monkeypatch):
    install_rules(monkeypatch)
    convert_deck(deck_path)
    assert source.rendered[("config version",)] == Value(2)


def test_open_decisions_skip_v2_validation(deck_path, source, monkeypatch):
    install_rules(monkeypatch)
    result = convert_deck(deck_path)
    assert result.conversion.decisions
    assert result.validated is False


def test_a_complete_conversion_is_validated(deck_path, source, monkeypatch, tmp_path):
    from zutil.update_input.model import MeshInfo

    mesh_file = tmp_path / "ShockTube.h5"
    mesh_file.write_bytes(b"")
    monkeypatch.setattr(
        "zutil.update_input.convert.read_mesh",
        lambda p: MeshInfo(str(p), "ShockTube", {}),
    )
    install_rules(monkeypatch, complete_shocktube_rule("ShockTube"))
    result = convert_deck(deck_path, mesh_path=mesh_file)
    assert result.conversion.decisions == []
    assert result.validated is True
    assert result.text.startswith("xmin = [12]")


def test_a_v2_failure_without_decisions_is_a_converter_bug(
    deck_path, source, monkeypatch, tmp_path
):
    from zutil.update_input.model import MeshInfo

    mesh_file = tmp_path / "ShockTube.h5"
    mesh_file.write_bytes(b"")
    monkeypatch.setattr(
        "zutil.update_input.convert.read_mesh",
        lambda p: MeshInfo(str(p), "ShockTube", {}),
    )

    def drop_all(c):
        for leaf in c.v1:
            c.drop(leaf, "dropped by the test")
        c.set(("solver", "time settings", "kind"), Value("no such kind"))

    install_rules(monkeypatch, rule("drop_all", 1, ("*",), drop_all))
    with pytest.raises(ConverterBugError) as raised:
        convert_deck(deck_path, mesh_path=mesh_file)
    message = str(raised.value)
    assert "time settings" in message
    assert "Zenotech" in message
    assert "shocktube.py" in message


def test_a_rule_contract_break_is_a_converter_bug(deck_path, source, monkeypatch):
    def claim_twice(c):
        leaf = c.v1.leaf("units")
        c.drop(leaf, "once")
        c.drop(leaf, "twice")

    install_rules(monkeypatch, rule("twice", 1, ("units",), claim_twice))
    with pytest.raises(ConverterBugError) as raised:
        convert_deck(deck_path)
    assert "claimed twice" in str(raised.value)


def test_validation_runs_in_the_deck_directory(
    deck_path, source, monkeypatch, tmp_path
):
    from zutil.update_input.model import MeshInfo

    (tmp_path / "helper_for_update_input_test.py").write_text("VALUE = 'IC_1'\n")
    source.prelude += (
        "import helper_for_update_input_test\n"
        "with open('marker.txt') as f:\n"
        "    MARK = f.read()\n"
    )
    (tmp_path / "marker.txt").write_text("ok")
    mesh_file = tmp_path / "ShockTube.h5"
    mesh_file.write_bytes(b"")
    monkeypatch.setattr(
        "zutil.update_input.convert.read_mesh",
        lambda p: MeshInfo(str(p), "ShockTube", {}),
    )
    install_rules(monkeypatch, complete_shocktube_rule("ShockTube"))
    monkeypatch.chdir(tmp_path.parent)
    result = convert_deck(deck_path, mesh_path=mesh_file)
    assert result.validated is True


def test_result_carries_report_and_summary(deck_path, source, monkeypatch):
    install_rules(monkeypatch)
    result = convert_deck(deck_path)
    assert result.output_path == deck_path.with_name("shocktube_v2.py")
    assert result.report_path == deck_path.with_name("shocktube_v2.conversion.md")
    assert result.summary.startswith(
        "Converting shocktube.py (config_version 0) to config_version 2"
    )
    assert result.report.startswith("# ")


def test_output_path_names_the_report(deck_path, source, monkeypatch, tmp_path):
    install_rules(monkeypatch)
    result = convert_deck(deck_path, output_path=tmp_path / "out" / "tube.py")
    assert result.report_path == tmp_path / "out" / "tube.conversion.md"
    assert "Wrote out/tube.py" in result.summary


def test_a_deck_already_at_config_version_2_is_refused(tmp_path):
    from zutil.update_input import source

    deck = tmp_path / "wing.py"
    deck.write_text('parameters = {"config version": 2, "solver": {}, "model": {}}\n')
    with pytest.raises(source.DeckReadError, match="already config version 2"):
        convert_deck(deck)


def test_a_mesh_transform_function_does_not_fail_the_legacy_check(tmp_path):
    from zutil.update_input.convert import legacy_errors

    def lift(**kwargs):
        return kwargs

    plain = legacy_errors({"units": "SI"}, tmp_path / "deck.py")
    with_function = legacy_errors(
        {"units": "SI", "mesh transform function": lift}, tmp_path / "deck.py"
    )
    assert with_function == plain
