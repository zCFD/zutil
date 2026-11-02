"""The source layer: reading v1 decks into leaves and writing the converted deck back.

Each toy deck in ``decks/source`` shows one way a v1 deck builds ``parameters``.
The round-trip test renders every deck with a v2 tree that mirrors its v1 leaves
and checks the result evaluates to the same dict, which proves the regenerated
statement, the folded edits and the base/variant rewrites keep the deck's meaning.
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from zutil.fileutils import get_parameters_from_file
from zutil.update_input import source
from zutil.update_input.model import Expr, Placeholder, Source, Value
from zutil.update_input.source import (
    BlockOverride,
    DeckReadError,
    RenderError,
    read_deck,
    read_related,
    related_decks,
    render_deck,
)

DECKS = Path(__file__).parent / "decks" / "source"
SINGLE = DECKS / "single"
REFUSED = {"refuse_zcfd.py", "refuse_paraview.py"}
SINGLE_DECKS = sorted(p.name for p in SINGLE.glob("*.py") if p.name not in REFUSED)
SET_DECKS = [
    "import_set/spread_variant.py",
    "import_set/dict_variant.py",
    "import_set/from_import_variant.py",
    "import_set/deepcopy_variant.py",
    "namespace_set/ns_variant.py",
    "namespace_set/ns_variant_edits.py",
    "namespace_set/ns_variant_dict.py",
    "namespace_set/ns_variant_shared.py",
    "chain/chain_top.py",
]


def same(a, b) -> bool:
    if a is b:
        return True
    if callable(a) and callable(b):
        ca, cb = a.__code__, b.__code__
        return (ca.co_code, ca.co_consts, ca.co_names) == (
            cb.co_code,
            cb.co_consts,
            cb.co_names,
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(same, a, b))
    return type(a) is type(b) and a == b


def leaf_sources(v1) -> dict:
    return {leaf.keys: leaf.source for leaf in v1}


def mirror(v1) -> dict:
    return {leaf.keys: Source(leaf) for leaf in v1}


def forget_modules(directory: Path) -> None:
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve().parent == directory.resolve():
            del sys.modules[name]


def load(path: Path) -> dict:
    try:
        return get_parameters_from_file(path.resolve())
    finally:
        forget_modules(path.parent)


def convert_set(path: Path) -> dict:
    """Write ``<deck>_v2.py`` for the deck and each base it takes parameters from, innermost first."""
    v1, deck = read_deck(path)
    return _write(v1, deck)


def _write(v1, deck) -> dict:
    base_tree = None
    for base_path in related_decks(deck):
        base_v1, base_deck = read_related(deck, base_path)
        base_tree = _write(base_v1, base_deck)
    tree = mirror(v1)
    text = render_deck(deck, tree, base_v2=base_tree)
    deck.path.with_name(f"{deck.path.stem}_v2.py").write_text(text)
    return tree


@pytest.fixture
def decks(tmp_path) -> Path:
    shutil.copytree(DECKS, tmp_path / "source")
    return tmp_path / "source"


@pytest.mark.parametrize("name", SINGLE_DECKS)
def test_round_trip_single(decks, name):
    path = decks / "single" / name
    v1, deck = read_deck(path)
    text = render_deck(deck, mirror(v1))
    out = path.with_name(f"{path.stem}_v2.py")
    out.write_text(text)
    assert same(load(out), v1.evaluated)


@pytest.mark.parametrize("name", SET_DECKS)
def test_round_trip_sets(decks, name):
    path = decks / name
    v1, _ = read_deck(path)
    convert_set(path)
    assert same(load(path.with_name(f"{path.stem}_v2.py")), v1.evaluated)


def test_a_pure_literal_keeps_every_source():
    v1, _ = read_deck(SINGLE / "literal.py")
    sources = leaf_sources(v1)
    assert sources[("time marching", "scheme", "name")] == '"runge kutta"'
    assert sources[("IC_1", "V", "vector")] == "[1.0, 0.0, 0.0]"
    assert sources[("write output", "volume variables")] == '("V", "p")'
    assert sources[("write output", "options")] == "{}"
    assert v1.leaf("ic 1", "temperature").line == 10
    assert v1.notes == []


def test_preamble_names_and_expressions_are_kept_as_source():
    v1, _ = read_deck(SINGLE / "preamble.py")
    sources = leaf_sources(v1)
    assert sources[("time marching", "unsteady", "total time")] == "100 * dt"
    assert sources[("time marching", "unsteady", "time step")] == "dt"
    assert sources[("IC_1", "reference length")] == "0.5 * c_ref"
    assert sources[("IC_1", "V", "vector")] == "[c_ref, 0.0, 0.0]"
    assert v1.value("time marching", "unsteady", "total time") == pytest.approx(0.1)


def test_helper_calls_keep_their_source_and_their_value():
    v1, _ = read_deck(SINGLE / "helpers.py")
    sources = leaf_sources(v1)
    assert sources[("IC_1", "temperature")] == "zutil.to_kelvin(540.0)"
    assert sources[("IC_1", "V", "vector")] == "zutil.vector_from_angle(alpha, 0.0)"
    assert sources[("IC_1", "speed of sound")] == "math.sqrt(1.4 * 287.0 * 288.15)"
    assert sources[("scale",)] == "np.array([1.0, 2.0, 3.0])"
    assert isinstance(v1.leaf("scale").value, np.ndarray)


def test_callables_keep_their_name_or_lambda_text():
    v1, _ = read_deck(SINGLE / "callables.py")
    sources = leaf_sources(v1)
    assert sources[("initial", "func")] == "shock"
    assert sources[("time marching", "cfl ramp")] == "ramp"
    assert sources[("time marching", "cfl transport")] == "lambda c: 2.0 * c"
    assert callable(v1.leaf("initial", "func").value)


def test_later_subscript_assignments_are_folded():
    v1, deck = read_deck(SINGLE / "subscript_edits.py")
    sources = leaf_sources(v1)
    assert sources[("time marching", "unsteady", "time step")] == "dt"
    assert v1.leaf("time marching", "unsteady", "time step").line == 8
    assert sources[("BC_2", "type")] == '"farfield"'
    assert sources[("time marching", "cycles")] == "50"
    text = render_deck(deck, mirror(v1))
    assert "parameters[" not in text
    assert text.startswith("dt = 0.1\n\nparameters = {")


def test_update_with_a_literal_is_folded():
    v1, deck = read_deck(SINGLE / "update.py")
    sources = leaf_sources(v1)
    assert sources[("restart",)] == "True"
    assert sources[("BC_1", "zone")] == "[1]"
    assert sources[("time marching", "cycles")] == "20"
    assert sources[("time marching", "cfl")] == "2.0"
    assert sources[("partitioner",)] == '"metis"'
    assert ".update" not in render_deck(deck, mirror(v1))


def test_dicts_built_in_code_are_literal_values_with_a_note():
    v1, deck = read_deck(SINGLE / "code_built.py")
    sources = leaf_sources(v1)
    assert sources[("time marching", "unsteady", "total time")] is None
    assert sources[("BC_1", "type")] is None
    assert sources[("BC_3", "zone")] is None
    assert sources[("restart",)] == "False"
    notes = " ".join(n.text for n in v1.notes)
    assert "line 13" in notes and "marching" in notes
    assert "line 14 builds 'BC_1'" in notes
    assert "line 15 builds 'BC_2', 'BC_3'" in notes
    assert "literal values" in notes
    text = render_deck(deck, mirror(v1))
    assert "for i, zone" not in text
    assert ".update" not in text
    assert "def boundaries():" in text


def test_a_dict_returned_by_a_function_is_literal_values():
    v1, deck = read_deck(SINGLE / "code_function.py")
    assert all(leaf.source is None for leaf in v1)
    assert any("line 5 builds" in n.text for n in v1.notes)
    text = render_deck(deck, mirror(v1))
    assert (
        'parameters = {"restart": False, "BC_1": {"zone": [1], "type": "wall"}}' in text
    )


def test_a_parameters_chosen_by_if_is_replaced_by_one_literal():
    v1, deck = read_deck(SINGLE / "code_if.py")
    assert v1.value("multigrid") == 2
    text = render_deck(deck, mirror(v1))
    assert "if coarse" not in text
    assert 'print("loaded")' in text
    assert text.startswith("coarse = True\n\nparameters = {")


def test_deleted_keys_are_absent_with_a_note():
    v1, deck = read_deck(SINGLE / "deletions.py")
    assert not v1.has("units")
    assert not v1.has("time marching", "cfl")
    assert v1.has("time marching", "cycles")
    deleted = [n for n in v1.notes if "deletes" in n.text]
    assert [n.v1_paths for n in deleted] == [(("units",),), (("time marching", "cfl"),)]
    text = render_deck(deck, mirror(v1))
    assert "del " not in text and ".pop" not in text


def test_deepcopy_of_a_local_literal_keeps_its_sources():
    v1, deck = read_deck(SINGLE / "deepcopy_local.py")
    sources = leaf_sources(v1)
    assert sources[("time marching", "unsteady", "time step")] == "dt"
    assert sources[("restart",)] == "True"
    assert any("defines base" in n.text for n in v1.notes)
    text = render_deck(deck, mirror(v1))
    assert "copy.deepcopy" not in text
    assert 'parameters["restart"]' not in text


def test_a_name_rebound_before_a_folded_edit_is_written_as_a_value():
    v1, deck = read_deck(SINGLE / "rebinding.py")
    sources = leaf_sources(v1)
    assert sources[("time marching", "unsteady", "time step")] == "dt"
    assert sources[("time marching", "unsteady", "total time")] is None
    assert any("dt changes between line 2 and line 4" in n.text for n in v1.notes)
    assert '"total time": 2.0' in render_deck(deck, mirror(v1))


@pytest.mark.parametrize(
    "name, module",
    [("refuse_zcfd.py", "zcfd"), ("refuse_paraview.py", "paraview.simple")],
)
def test_decks_that_import_solver_modules_are_refused(name, module):
    with pytest.raises(DeckReadError) as raised:
        read_deck(SINGLE / name)
    assert f"imports {module} at line 1" in str(raised.value)


def test_a_deck_that_does_not_run_names_the_line(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("x = 1\nparameters = {'a': missing_name}\n")
    with pytest.raises(DeckReadError) as raised:
        read_deck(path)
    assert "line 2: NameError" in str(raised.value)


def test_a_base_needing_names_from_its_variant_says_so():
    with pytest.raises(DeckReadError) as raised:
        read_deck(DECKS / "namespace_set" / "ns_base.py")
    text = str(raised.value)
    assert "ns_base.py does not run" in text and "'mach'" in text
    assert "convert that deck instead" in text


def test_a_misspelt_parameters_alone_is_named(tmp_path):
    path = tmp_path / "typo.py"
    path.write_text("paramters = {'restart': False}\n")
    with pytest.raises(DeckReadError) as raised:
        read_deck(path)
    assert "sets no parameters dict; it sets paramters" in str(raised.value)


def test_the_prelude_and_main_block_are_copied_byte_for_byte(decks):
    path = decks / "single" / "literal.py"
    original = path.read_text()
    v1, deck = read_deck(path)
    text = render_deck(deck, {("restart",): Value(True)})
    assert text.startswith('"""A deck written as one literal."""\n\n')
    assert text.endswith('\n\nif __name__ == "__main__":\n    print(parameters)\n')
    head, _, _ = original.partition("parameters = {")
    assert text.startswith(head)


def test_crlf_line_endings_survive(tmp_path):
    path = tmp_path / "crlf.py"
    path.write_bytes(
        b"x = 1\r\nparameters = {'a': x}\r\nparameters['b'] = 2\r\ny = 3\r\n"
    )
    v1, deck = read_deck(path)
    text = render_deck(deck, mirror(v1))
    assert text == 'x = 1\r\nparameters = {"a": x, "b": 2}\r\ny = 3\r\n'


def test_v2_values_are_written_by_kind(tmp_path):
    path = tmp_path / "kinds.py"
    path.write_text("dt = 2.0\nparameters = {'time step': dt}\n")
    v1, deck = read_deck(path)
    leaf = v1.leaf("time step")
    v2 = {
        ("solver", "time settings", "time step"): Source(leaf),
        ("solver", "time settings", "total time"): Expr("10 * dt", 20.0),
        ("solver", "time settings", "order"): Value("second"),
        ("model", "wing", "mesh"): Placeholder(1, "DECIDE: which mesh (see item 1)"),
        ("config version",): Value(2),
    }
    text = render_deck(deck, v2)
    namespace: dict = {}
    exec(text, namespace)
    assert namespace["parameters"] == {
        "solver": {
            "time settings": {"time step": 2.0, "total time": 20.0, "order": "second"}
        },
        "model": {"wing": {"mesh": "DECIDE: which mesh (see item 1)"}},
        "config version": 2,
    }
    assert '"total time": 10 * dt' in text
    assert list(namespace["parameters"]) == ["solver", "model", "config version"]


def test_a_callable_without_source_is_an_error(tmp_path):
    path = tmp_path / "fn.py"
    path.write_text(
        "def f():\n    return {'ramp': lambda c: c}\n\n\nparameters = f()\n"
    )
    v1, deck = read_deck(path)
    with pytest.raises(RenderError) as raised:
        render_deck(deck, {("solver", "ramp"): Source(v1.leaf("ramp"))})
    assert "solver > ramp is a callable with no source text" in str(raised.value)


def test_without_ruff_a_statement_that_fits_stays_on_one_line(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "_ruff_command", lambda: None)
    path = tmp_path / "plain.py"
    path.write_text("a = 1\nparameters = {'x': a}\n")
    v1, deck = read_deck(path)
    v2 = {
        ("solver", "x"): Source(v1.leaf("x")),
        ("solver", "y"): Value([1, 2]),
        ("z",): Value({}),
    }
    expected = 'a = 1\nparameters = {"solver": {"x": a, "y": [1, 2]}, "z": {}}\n'
    assert render_deck(deck, v2) == expected


def test_without_ruff_a_long_statement_breaks_one_entry_per_line(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "_ruff_command", lambda: None)
    path = tmp_path / "plain.py"
    path.write_text("a = 1\nparameters = {'x': a}\n")
    v1, deck = read_deck(path)
    v2 = {
        ("solver", "x"): Source(v1.leaf("x")),
        ("solver", "surface variables"): Value(["pressure", "temperature", "mach"]),
        ("solver", "volume variables"): Value(["velocity"]),
    }
    expected = (
        "a = 1\n"
        "parameters = {\n"
        '    "solver": {\n'
        '        "x": a,\n'
        '        "surface variables": ["pressure", "temperature", "mach"],\n'
        '        "volume variables": ["velocity"],\n'
        "    },\n"
        "}\n"
    )
    assert render_deck(deck, v2) == expected


@pytest.mark.skipif(source._ruff_command() is None, reason="ruff is not installed")
def test_ruff_formats_the_statement(tmp_path):
    path = tmp_path / "plain.py"
    path.write_text("parameters = {'x': 1}\n")
    v1, deck = read_deck(path)
    v2 = {("solver", f"key {i}"): Value(i) for i in range(12)}
    text = render_deck(deck, v2)
    assert text.startswith('parameters = {\n    "solver": {\n        "key 0": 0,\n')
    assert all(len(line) <= 88 for line in text.splitlines())


def test_a_loader_variant_names_its_base_and_its_namespace():
    v1, deck = read_deck(DECKS / "namespace_set" / "ns_variant.py")
    base_path = (DECKS / "namespace_set" / "ns_base.py").resolve()
    assert related_decks(deck) == [base_path]
    base_v1, base_deck = read_related(deck, base_path)
    assert base_v1.value("IC 1", "V", "vector") == [170.0, 0.0, 0.0]
    assert leaf_sources(base_v1)[("time marching", "unsteady", "total time")] == "t_end"
    assert related_decks(base_deck) == []
    assert deck.base.shared
    assert leaf_sources(v1)[("time marching", "cycles")] == "cycles"


def test_a_loader_variant_with_no_edits_only_renames_its_base(decks):
    path = decks / "namespace_set" / "ns_variant.py"
    convert_set(path)
    expected = path.read_text().replace('"ns_base.py"', '"ns_base_v2.py"')
    assert path.with_name("ns_variant_v2.py").read_text() == expected


def test_a_misspelt_parameters_variant_is_copied_with_its_base_renamed(decks):
    path = decks / "namespace_set" / "ns_variant_shared.py"
    v1, deck = read_deck(path)
    assert v1.value("time marching", "cycles") == 100
    convert_set(path)
    expected = path.read_text().replace('"ns_base.py"', '"ns_base_v2.py"')
    assert path.with_name("ns_variant_shared_v2.py").read_text() == expected


def test_a_loader_variant_with_a_dict_namespace_keeps_the_dict(decks):
    path = decks / "namespace_set" / "ns_variant_dict.py"
    v1, deck = read_deck(path)
    assert not deck.base.shared
    assert leaf_sources(v1)[("time marching", "cycles")] is None
    convert_set(path)
    expected = path.read_text().replace('"ns_base.py"', '"ns_base_v2.py"')
    assert path.with_name("ns_variant_dict_v2.py").read_text() == expected


def test_a_loader_variant_writes_only_its_overrides(decks):
    path = decks / "namespace_set" / "ns_variant_edits.py"
    v1, deck = read_deck(path)
    assert leaf_sources(v1)[("time marching", "unsteady", "time step")] == "dt"
    assert not v1.has("time marching", "cycles")
    replaced = [n for n in v1.notes if isinstance(n, BlockOverride)]
    assert [(n.v1_paths, n.base) for n in replaced] == [
        ((("time marching",),), "ns_base.py")
    ]
    convert_set(path)
    text = path.with_name("ns_variant_edits_v2.py").read_text()
    assert 'get_parameters_from_file("ns_base_v2", namespace=globals())' in text
    assert 'parameters["time marching"] = {' in text
    assert '"time step": dt' in text
    assert 'parameters["BC_2"] = {"zone": [2], "type": "farfield"}' in text
    assert "IC_1" not in text


def test_a_variant_override_sits_where_the_edits_were(decks):
    path = decks / "namespace_set" / "ns_variant_edits.py"
    v1, deck = read_deck(path)
    base_v1, _ = read_related(deck, related_decks(deck)[0])
    base_tree = mirror(base_v1)
    tree = dict(base_tree)
    tree[("BC_1", "type")] = Value("symmetry")
    text = render_deck(deck, tree, base_v2=base_tree)
    head = (
        path.read_text()
        .split("\n\nparameters[")[0]
        .replace('"ns_base"', '"ns_base_v2"')
    )
    assert text == head + '\n\nparameters["BC_1"]["type"] = "symmetry"\n'


def test_a_variant_placeholder_matching_the_base_is_inherited(decks):
    path = decks / "namespace_set" / "ns_variant.py"
    v1, deck = read_deck(path)
    base_tree = {("model", "m", "mesh"): Placeholder(1, "DECIDE: mesh (see item 1)")}
    tree = {("model", "m", "mesh"): Placeholder(4, "DECIDE: mesh (see item 4)")}
    text = render_deck(deck, tree, base_v2=base_tree)
    assert "parameters[" not in text


def test_a_variant_needs_its_base_tree():
    v1, deck = read_deck(DECKS / "namespace_set" / "ns_variant.py")
    with pytest.raises(RenderError) as raised:
        render_deck(deck, mirror(v1))
    assert "convert ns_base.py first" in str(raised.value)


def test_a_chain_is_read_link_by_link():
    v1, top = read_deck(DECKS / "chain" / "chain_top.py")
    assert v1.value("FZ_1", "axis") == [0.0, 0.0, 1.0]
    assert v1.value("FZ_1", "omega") == 20.0
    mid_path = related_decks(top)[0]
    assert mid_path.name == "chain_mid.py"
    mid_v1, mid = read_related(top, mid_path)
    assert leaf_sources(mid_v1)[("FZ_1", "axis")] == "[0.0, 0.0, angle]"
    assert [p.name for p in related_decks(mid)] == ["chain_root.py"]
    root_v1, _ = read_related(mid, related_decks(mid)[0])
    assert leaf_sources(root_v1)[("FZ_1", "omega")] == "omega"


def test_a_chain_link_writes_its_edit_as_an_override(decks):
    convert_set(decks / "chain" / "chain_top.py")
    mid = (decks / "chain" / "chain_mid_v2.py").read_text()
    assert 'get_parameters_from_file("chain_root_v2.py", namespace=globals())' in mid
    assert 'parameters["FZ_1"]["axis"] = [0.0, 0.0, angle]' in mid
    top = (decks / "chain" / "chain_top_v2.py").read_text()
    assert '"chain_mid_v2.py"' in top and "parameters[" not in top


def test_a_base_named_by_a_variable_is_renamed_where_it_is_defined(tmp_path):
    (tmp_path / "plain_base.py").write_text("parameters = {'restart': False}\n")
    path = tmp_path / "plain_sa.py"
    path.write_text(
        "import zutil\n\nbase_case = 'plain_base'\n\n"
        "parameters = zutil.get_parameters_from_file(base_case)\n"
        "parameters['restart'] = True\n"
    )
    v1, deck = read_deck(path)
    assert not deck.base.shared
    base_v1, _ = read_related(deck, related_decks(deck)[0])
    text = render_deck(deck, mirror(v1), base_v2=mirror(base_v1))
    assert "base_case = 'plain_base_v2'" in text
    assert 'parameters["restart"] = True' in text


@pytest.mark.parametrize(
    "name, rewritten",
    [
        ("spread_variant.py", "import spread_base_v2 as spread_base"),
        ("dict_variant.py", "import spread_base_v2 as base"),
        ("from_import_variant.py", "from spread_base_v2 import parameters"),
        ("deepcopy_variant.py", "import spread_base_v2 as spread_base"),
    ],
)
def test_import_variants_name_the_base_and_rewrite_the_import(decks, name, rewritten):
    path = decks / "import_set" / name
    v1, deck = read_deck(path)
    assert related_decks(deck) == [(decks / "import_set" / "spread_base.py").resolve()]
    convert_set(path)
    assert rewritten in path.with_name(f"{path.stem}_v2.py").read_text()


def test_a_spread_variant_writes_a_nested_spread(decks):
    path = decks / "import_set" / "spread_variant.py"
    v1, deck = read_deck(path)
    base_v1, _ = read_related(deck, related_decks(deck)[0])
    tree = mirror(v1)
    tree[("time marching", "cycles")] = Value(30)
    text = render_deck(deck, tree, base_v2=mirror(base_v1))
    statement = text.split("\n\n", 1)[1]
    parsed = ast.parse(statement)
    assert ast.unparse(parsed) == (
        "parameters = {**spread_base.parameters, 'restart': True, "
        "'time marching': {**spread_base.parameters['time marching'], 'cycles': 30}}"
    )


def test_inherited_names_a_variant_does_not_share_are_values(decks):
    v1, deck = read_deck(decks / "import_set" / "from_import_variant.py")
    sources = leaf_sources(v1)
    assert sources[("time marching", "unsteady", "time step")] is None
    assert sources[("time marching", "unsteady", "total time")] == "1.0"
    assert sources[("time marching", "cycles")] == "20"
    assert deck.inherited_text[("time marching", "unsteady", "time step")] == "dt"


def test_a_change_mixed_with_other_bindings_is_refused(tmp_path):
    path = tmp_path / "mixed.py"
    path.write_text(
        "parameters = {'a': 1}\n"
        "for z in [1, 2]:\n"
        "    parameters['BC_%d' % z] = {'zone': [z]}\n"
        "    last = z\n"
        "print(last)\n"
    )
    with pytest.raises(DeckReadError) as raised:
        read_deck(path)
    assert "line 2 changes parameters and also sets last" in str(raised.value)


def test_a_later_reader_of_parameters_is_noted(tmp_path):
    path = tmp_path / "reader.py"
    path.write_text("parameters = {'a': 1}\nprint(parameters['a'])\n")
    v1, _ = read_deck(path)
    assert any("line 2 reads parameters" in n.text for n in v1.notes)


def test_reading_does_not_change_the_namespace_given(tmp_path):
    path = tmp_path / "ns.py"
    path.write_text("y = x + 1\nparameters = {'y': y}\n")
    namespace = {"x": 1}
    v1, _ = read_deck(path, namespace=namespace)
    assert v1.value("y") == 2
    assert namespace == {"x": 1}


def test_a_symlinked_deck_keeps_the_name_it_was_given(tmp_path):
    (tmp_path / "real.py").write_text("parameters = {'restart': False}\n")
    (tmp_path / "alias.py").symlink_to(tmp_path / "real.py")
    _, deck = read_deck(tmp_path / "alias.py")
    assert deck.path.name == "alias.py"


def test_a_base_that_imports_zcfd_is_refused(tmp_path):
    (tmp_path / "dg_base.py").write_text("import zcfd\nparameters = {}\n")
    path = tmp_path / "dg.py"
    path.write_text(
        "import zutil\nparameters = zutil.get_parameters_from_file('dg_base.py')\n"
    )
    with pytest.raises(DeckReadError) as raised:
        read_deck(path)
    assert "dg.py line 2 runs code that imports zcfd" in str(raised.value)


def test_a_base_that_imports_zcfd_is_refused_even_where_zcfd_imports(
    tmp_path, monkeypatch
):
    import types

    zcfd = types.ModuleType("zcfd")
    monkeypatch.setitem(sys.modules, "zcfd", zcfd)
    (tmp_path / "dg_base.py").write_text("import zcfd\nparameters = {}\n")
    path = tmp_path / "dg.py"
    path.write_text(
        "import zutil\nparameters = zutil.get_parameters_from_file('dg_base.py')\n"
    )
    with pytest.raises(DeckReadError) as raised:
        read_deck(path)
    assert "dg.py line 2 runs code that imports zcfd" in str(raised.value)
    assert sys.modules["zcfd"] is zcfd


def test_loop_names_reused_by_a_later_loop_do_not_block_a_fold(tmp_path):
    path = tmp_path / "loops.py"
    path.write_text(
        "parameters = {'report': {'monitor': {}}}\n"
        "for key, value in {'FZ_1': {'type': 'disc'}}.items():\n"
        "    parameters[key] = value\n"
        "for key, value in {'MR_1': {'name': 'p'}}.items():\n"
        "    parameters['report']['monitor'][key] = value\n"
    )
    v1, deck = read_deck(path)
    assert v1.value("FZ_1", "type") == "disc"
    assert v1.value("report", "monitor", "MR_1", "name") == "p"
    assert "for key" not in render_deck(deck, mirror(v1))


def test_an_override_group_is_named_as_such(tmp_path):
    deck = tmp_path / "override.py"
    deck.write_text('override = {"mesh_case_pair": [("a.h5", "a.py")]}\n')
    with pytest.raises(source.DeckReadError, match="overset group"):
        source.read_deck(deck)
