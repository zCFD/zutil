"""The conversion report and the console summary."""

from __future__ import annotations

import pytest

from zutil.update_input import report
from zutil.update_input.model import Conversion, Leaf, V1Deck, Value


@pytest.fixture
def conversion():
    evaluated = {
        "units": "SI",
        "material": "air",
        "equations": "euler",
        "time marching": {"scheme": {"kind": "local timestepping"}, "cfl": 2.0},
        "IC_1": {"V": {"vector": [1.0, 0.0, 0.0]}},
    }
    leaves = [
        Leaf(("units",), "SI", "'SI'", 3),
        Leaf(("material",), "air", "'air'", 4),
        Leaf(("equations",), "euler", "'euler'", 5),
        Leaf(
            ("time marching", "scheme", "kind"),
            "local timestepping",
            "'local timestepping'",
            7,
        ),
        Leaf(("time marching", "cfl"), 2.0, "cfl", 8),
        Leaf(
            ("IC_1", "V", "vector"),
            [1.0, 0.0, 0.0],
            "zutil.vector_from_aoa(2.0, 0.0)",
            10,
        ),
    ]
    c = Conversion(V1Deck(leaves, evaluated), "wing_steady")
    v1 = c.v1
    c.drop(v1.leaf("units"), "always SI")
    c.drop(v1.leaf("material"), "folded into fluid properties")
    c.move(v1.leaf("equations"), ("solver", "equations", "type"))
    c.move(
        v1.leaf("IC_1", "V", "vector"),
        ("solver", "reference conditions", "IC_1", "v", "vector"),
    )
    c.move(
        v1.leaf("time marching", "cfl"),
        ("solver", "time settings", "cfl"),
        Value(2.0),
        respelled=True,
    )
    c.decide(
        [v1.leaf("time marching", "scheme", "kind")],
        ("solver", "time settings", "type"),
        "Is this run steady or unsteady?",
        options=("steady", "unsteady"),
        details="v1 local timestepping has no single v2 home.",
    )
    c.decide(
        [],
        ("model", "wing_steady", "mesh"),
        "Which mesh file does this deck run on?",
    )
    c.note("parameters was built in a loop at line 12", ("IC_1",))
    return c


@pytest.fixture
def names():
    return report.Names(
        deck="wing_steady.py",
        output="wing_steady_v2.py",
        report="wing_steady_v2.conversion.md",
        config_version=0,
    )


def test_summary_shape(conversion, names):
    text = report.summary(conversion, names, validated=False)
    lines = text.splitlines()
    assert (
        lines[0] == "Converting wing_steady.py (config_version 0) to config_version 2"
    )
    assert lines[1] == ""
    assert lines[2] == "  moved        2 keys"
    assert lines[3] == "  re-spelled   1 value"
    assert (
        lines[4]
        == "  dropped      2 keys   (units: always SI · material: folded into fluid properties)"
    )
    assert lines[5] == "  DECIDE       2 keys"
    assert "  1. time marching > scheme > kind = 'local timestepping'" in lines
    assert "     Is this run steady or unsteady? ('steady' or 'unsteady')" in lines
    assert "  2. model > wing_steady > mesh" in lines
    assert (
        lines[-2]
        == "Wrote wing_steady_v2.py (draft: 2 decisions open; replace each DECIDE string)"
    )
    assert (
        lines[-1] == "Wrote wing_steady_v2.conversion.md (every key and where it went)"
    )


def test_summary_never_joins_keys_with_dots(conversion, names):
    text = report.summary(conversion, names, validated=False)
    assert "time marching.scheme" not in text
    assert "time marching > scheme > kind" in text


def test_summary_says_a_complete_deck_validated(names):
    c = Conversion(
        V1Deck([Leaf(("equations",), "euler", "'euler'")], {"equations": "euler"}), "w"
    )
    c.move(c.v1.leaf("equations"), ("solver", "equations", "type"))
    text = report.summary(c, names, validated=True)
    assert "Wrote wing_steady_v2.py (valid config version 2)" in text
    assert "DECIDE" not in text
    assert "re-spelled" not in text


def test_summary_singular_decision(names):
    c = Conversion(V1Deck([], {}), "w")
    c.decide([], ("model", "w", "mesh"), "Which mesh?")
    text = report.summary(c, names, validated=False)
    assert "(draft: 1 decision open; replace each DECIDE string)" in text
    assert "  DECIDE       0 keys" not in text


def test_summary_shortens_a_long_dropped_list(names):
    leaves = [Leaf((f"k{i}",), i, str(i)) for i in range(6)]
    c = Conversion(V1Deck(leaves, {f"k{i}": i for i in range(6)}), "w")
    for leaf in leaves:
        c.drop(leaf, "unused")
    line = [
        ln for ln in report.summary(c, names, False).splitlines() if "dropped" in ln
    ][0]
    assert line.endswith("· ...)")


def test_summary_counts_notes(conversion, names):
    text = report.summary(conversion, names, validated=False)
    assert "  notes        1   (in the report)" in text


def test_markdown_header_and_counts(conversion, names):
    md = report.markdown(conversion, names, validated=False)
    assert md.startswith("# wing_steady.py to config version 2\n")
    assert "wing_steady_v2.py" in md
    assert "| moved | 2 |" in md
    assert "| re-spelled | 1 |" in md
    assert "| dropped | 2 |" in md
    assert "| DECIDE | 2 |" in md


def test_markdown_lists_every_v1_key_in_deck_order(conversion, names):
    md = report.markdown(conversion, names, validated=False)
    rows = [ln for ln in md.splitlines() if ln.startswith("| `")]
    keys = [r.split("`")[1] for r in rows]
    assert keys == [
        "units",
        "material",
        "equations",
        "time marching > scheme > kind",
        "time marching > cfl",
        "IC_1 > V > vector",
    ]
    assert "| `units` | dropped: always SI |  |" in md
    assert "| `equations` | moved | `solver > equations > type` |" in md
    assert (
        "| `time marching > cfl` | re-spelled | `solver > time settings > cfl` |" in md
    )
    assert (
        "| `time marching > scheme > kind` | DECIDE, item 1 | `solver > time settings > type` |"
        in md
    )


def test_markdown_numbered_decisions(conversion, names):
    md = report.markdown(conversion, names, validated=False)
    assert "### 1. Is this run steady or unsteady?" in md
    assert "Options: 'steady' or 'unsteady'" in md
    assert "v1 local timestepping has no single v2 home." in md
    assert "v1 keys: `time marching > scheme > kind`" in md
    assert "### 2. Which mesh file does this deck run on?" in md
    assert "v2 key: `model > wing_steady > mesh`" in md


def test_markdown_notes(conversion, names):
    md = report.markdown(conversion, names, validated=False)
    assert "## Notes" in md
    assert "- parameters was built in a loop at line 12 (`IC_1`)" in md


def test_one_note_at_several_keys_is_one_line_naming_every_key(conversion, names):
    conversion.note("parameters was built in a loop at line 12", ("equations",))
    md = report.markdown(conversion, names, validated=False)
    assert md.count("parameters was built in a loop") == 1
    assert "- parameters was built in a loop at line 12 (`IC_1`, `equations`)" in md
    text = report.summary(conversion, names, validated=False)
    assert "  notes        1   (in the report)" in text


def test_markdown_escapes_table_pipes(names):
    c = Conversion(V1Deck([Leaf(("a",), 1, "1")], {"a": 1}), "w")
    c.drop(c.v1.leaf("a"), "x | y")
    md = report.markdown(c, names, validated=False)
    assert "dropped: x \\| y" in md


def test_markdown_states_validation(names):
    c = Conversion(V1Deck([], {}), "w")
    md = report.markdown(c, names, validated=True)
    assert "passed config version 2 validation" in md
