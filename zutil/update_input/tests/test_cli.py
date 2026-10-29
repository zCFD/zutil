"""The update_input command line."""

from __future__ import annotations

import pytest

from zutil.update_input import cli
from zutil.update_input.model import MeshInfo, V1Deck
from zutil.update_input.tests.engine.fakes import (
    FakeSource,
    complete_shocktube_rule,
    flatten,
    install_rules,
    install_source,
    rule,
    shocktube_deck,
)

V1_TEXT = "parameters = {'units': 'SI'}\n"


@pytest.fixture
def deck(tmp_path):
    path = tmp_path / "wing_steady.py"
    path.write_text(V1_TEXT)
    return path


@pytest.fixture
def source(monkeypatch):
    return install_source(monkeypatch, FakeSource(shocktube_deck()))


def test_writes_the_default_outputs_beside_the_input(deck, source, monkeypatch, capsys):
    install_rules(monkeypatch)
    assert cli.main([str(deck)]) == 0
    out = deck.with_name("wing_steady_v2.py")
    md = deck.with_name("wing_steady_v2.conversion.md")
    assert out.read_text().startswith("xmin = [12]")
    assert md.read_text().startswith("# wing_steady.py to config version 2")
    assert deck.read_text() == V1_TEXT
    printed = capsys.readouterr().out
    assert printed.startswith(
        "Converting wing_steady.py (config_version 0) to config_version 2"
    )
    assert "draft:" in printed
    assert "Wrote wing_steady_v2.conversion.md" in printed


def test_output_option(deck, source, monkeypatch, tmp_path):
    install_rules(monkeypatch)
    target = tmp_path / "elsewhere" / "wing.py"
    target.parent.mkdir()
    assert cli.main([str(deck), "--output", str(target)]) == 0
    assert target.exists()
    assert (tmp_path / "elsewhere" / "wing.conversion.md").exists()


def test_never_overwrites_the_input(deck, source, monkeypatch, capsys):
    install_rules(monkeypatch)
    assert cli.main([str(deck), "--output", str(deck), "--force"]) == 1
    assert deck.read_text() == V1_TEXT
    assert "input" in capsys.readouterr().err


def test_refuses_to_overwrite_an_existing_output(deck, source, monkeypatch, capsys):
    install_rules(monkeypatch)
    out = deck.with_name("wing_steady_v2.py")
    out.write_text("keep me")
    assert cli.main([str(deck)]) == 1
    assert out.read_text() == "keep me"
    assert "--force" in capsys.readouterr().err


def test_refuses_to_overwrite_an_existing_report(deck, source, monkeypatch):
    install_rules(monkeypatch)
    md = deck.with_name("wing_steady_v2.conversion.md")
    md.write_text("keep me")
    assert cli.main([str(deck)]) == 1
    assert md.read_text() == "keep me"


def test_force_overwrites_an_existing_output(deck, source, monkeypatch):
    install_rules(monkeypatch)
    out = deck.with_name("wing_steady_v2.py")
    out.write_text("old")
    assert cli.main([str(deck), "--force"]) == 0
    assert out.read_text() != "old"


def test_missing_deck(tmp_path, capsys):
    assert cli.main([str(tmp_path / "nowhere.py")]) == 1
    assert "nowhere.py" in capsys.readouterr().err


def test_legacy_errors_exit_2(deck, monkeypatch, capsys):
    evaluated = {"equations": "euler"}
    install_source(monkeypatch, FakeSource(V1Deck(flatten(evaluated), evaluated)))
    install_rules(monkeypatch)
    assert cli.main([str(deck)]) == 2
    err = capsys.readouterr().err
    assert "time marching" in err
    assert not deck.with_name("wing_steady_v2.py").exists()


def test_ignore_legacy_errors_writes_a_draft(deck, monkeypatch):
    evaluated = {"equations": "euler"}
    install_source(monkeypatch, FakeSource(V1Deck(flatten(evaluated), evaluated)))
    install_rules(monkeypatch)
    assert cli.main([str(deck), "--ignore-legacy-errors"]) == 0
    assert deck.with_name("wing_steady_v2.py").exists()


def test_converter_bug_exits_3(deck, source, monkeypatch, capsys):
    def claim_twice(c):
        leaf = c.v1.leaf("units")
        c.drop(leaf, "once")
        c.drop(leaf, "twice")

    install_rules(monkeypatch, rule("twice", 1, ("units",), claim_twice))
    assert cli.main([str(deck)]) == 3
    assert "Zenotech" in capsys.readouterr().err
    assert not deck.with_name("wing_steady_v2.py").exists()


def test_mesh_option_and_a_valid_deck(deck, source, monkeypatch, capsys):
    mesh = deck.with_name("wing.h5")
    mesh.write_bytes(b"")
    monkeypatch.setattr(
        "zutil.update_input.convert.read_mesh", lambda p: MeshInfo(str(p), "wing", {})
    )
    install_rules(monkeypatch, complete_shocktube_rule("wing"))
    assert cli.main([str(deck), "--mesh", str(mesh)]) == 0
    printed = capsys.readouterr().out
    assert "Wrote wing_steady_v2.py (valid config version 2)" in printed


def test_unreadable_mesh_exits_1(deck, source, monkeypatch, capsys):
    install_rules(monkeypatch)
    mesh = deck.with_name("wing.h5")
    mesh.write_text("not a mesh")
    assert cli.main([str(deck), "--mesh", str(mesh)]) == 1
    assert "wing.h5" in capsys.readouterr().err


def test_console_script_is_declared():
    from pathlib import Path

    setup = Path(cli.__file__).resolve().parents[2] / "setup.py"
    assert "update_input=zutil.update_input.cli:main" in setup.read_text()
