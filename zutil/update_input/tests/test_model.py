import pytest

from zutil.update_input.model import (
    Conversion,
    ConversionError,
    Expr,
    Leaf,
    Placeholder,
    Source,
    UNCONVERTED,
    V1Deck,
    Value,
)


def _deck():
    leaves = [
        Leaf(("time marching", "cfl ramp factor", "growth"), 1.1, "1.1"),
        Leaf(("time marching", "cfl ramp factor", "initial"), 1.0, "1.0"),
        Leaf(("IC_2", "total pressure ratio"), 1.05, "p_ratio"),
    ]
    return V1Deck(leaves, {})


def test_fold_claims_every_leaf_into_one_value():
    deck = _deck()
    c = Conversion(deck, "wing")
    growth, initial = deck.leaves_under("time marching", "cfl ramp factor")
    path = ("solver", "convergence control", "cfl ramp")
    c.fold([growth, initial], path, Expr("[...]", None))
    assert c.unclaimed() == [deck.leaf("IC_2", "total pressure ratio")]
    assert [d.v2_path for d in c.dispositions] == [path, path]
    assert c.v2[path] == Expr("[...]", None)


def test_fold_rejects_a_claimed_leaf():
    deck = _deck()
    c = Conversion(deck, "wing")
    growth = deck.leaf("time marching", "cfl ramp factor", "growth")
    c.drop(growth, "test")
    with pytest.raises(ConversionError):
        c.fold([growth], ("solver", "x"), Expr("1", 1))


def test_also_copies_a_claimed_leaf_and_notes_it():
    deck = _deck()
    c = Conversion(deck, "wing")
    ratio = deck.leaf("IC_2", "total pressure ratio")
    first = ("model", "wing", "boundary conditions", "BC_3", "total pressure ratio")
    second = ("model", "wing", "boundary conditions", "BC_4", "total pressure ratio")
    c.move(ratio, first)
    c.also(ratio, second)
    assert c.v2[second] == Source(ratio)
    assert len(c.dispositions) == 1
    assert "BC_4" in c.notes[-1].text


def test_also_needs_a_claimed_leaf():
    deck = _deck()
    c = Conversion(deck, "wing")
    with pytest.raises(ConversionError):
        c.also(deck.leaf("IC_2", "total pressure ratio"), ("solver", "x"))


def test_park_keeps_the_value_under_the_unconverted_keys():
    deck = _deck()
    c = Conversion(deck, "wing")
    ratio = deck.leaf("IC_2", "total pressure ratio")
    c.park(
        [ratio], "IC_2", "Where does the ratio go?", details="v2 has no home for it."
    )
    [decision] = c.decisions
    assert c.v2[(UNCONVERTED, "IC_2")] == Source(ratio)
    assert decision.details == (
        f"v2 has no home for it. Its v1 value is kept under '{UNCONVERTED} > IC_2'."
    )
    assert c.unclaimed() == deck.leaves_under("time marching")


def test_finish_parks_every_unclaimed_leaf_with_its_source():
    deck = _deck()
    c = Conversion(deck, "wing")
    c.finish()
    name = "IC_2 > total pressure ratio"
    assert c.v2[(UNCONVERTED, name)] == Source(
        deck.leaf("IC_2", "total pressure ratio")
    )
    assert c.unclaimed() == []
    assert len(c.decisions) == 3
    assert all(d.kind == "decide" for d in c.dispositions)


def _parked_between_two_decisions():
    deck = _deck()
    c = Conversion(deck, "wing")
    growth, initial = deck.leaves_under("time marching", "cfl ramp factor")
    first = c.decide([growth], ("solver", "a"), "First?")
    c.park([deck.leaf("IC_2", "total pressure ratio")], "IC_2", "Where?")
    last = c.decide([initial], ("solver", "b"), "Last?")
    c.set(("solver", "c"), last)
    return c, first


def test_finish_for_a_variant_leaves_parked_keys_to_it():
    c, _ = _parked_between_two_decisions()
    c.finish(variant="rotor.py")
    assert not any(p[:1] == (UNCONVERTED,) for p in c.v2)
    [dropped] = [d for d in c.dispositions if d.kind == "dropped"]
    assert dropped.v1_path == ("ic 2", "total pressure ratio")
    assert "rotor.py" in dropped.reason
    assert dropped.decision_id is None


def test_finish_for_a_variant_numbers_the_remaining_decisions_from_one():
    c, first = _parked_between_two_decisions()
    c.finish(variant="rotor.py")
    assert [d.id for d in c.decisions] == [1, 2]
    assert [d.question for d in c.decisions] == ["First?", "Last?"]
    assert c.v2[("solver", "a")] == first
    for path in (("solver", "b"), ("solver", "c")):
        value = c.v2[path]
        assert isinstance(value, Placeholder)
        assert value.decision_id == 2
        assert "item 2" in value.text
    assert sorted(d.decision_id for d in c.dispositions if d.kind == "decide") == [1, 2]


def test_finish_keeps_a_value_given_to_a_renumbered_decision():
    deck = _deck()
    c = Conversion(deck, "wing")
    growth, initial = deck.leaves_under("time marching", "cfl ramp factor")
    c.park([growth], "growth", "Where?")
    c.decide([initial], ("solver", "b"), "Last?", value=Value(1.0))
    c.finish(variant="rotor.py")
    assert c.v2[("solver", "b")] == Value(1.0)
    assert [d.id for d in c.decisions] == [1]


def test_finish_without_a_variant_keeps_parked_keys():
    c, _ = _parked_between_two_decisions()
    c.finish()
    assert (UNCONVERTED, "IC_2") in c.v2
    assert [d.id for d in c.decisions] == [1, 2, 3]


def _equations_deck(**parameters):
    return V1Deck([], parameters)


@pytest.mark.parametrize(
    "spelling, expected_set",
    [
        ("incomp viscous", "viscous"),
        ("Incomp_RANS", "rans"),
        ("INCOMP_LES", "les"),
    ],
)
def test_incompressible_equation_names_classify_as_incompressible(
    spelling, expected_set
):
    deck = _equations_deck(equations=spelling)
    assert deck.incompressible
    assert deck.equation_set == expected_set


@pytest.mark.parametrize("spelling, name", [("euler", "euler"), ("RANS", "rans")])
def test_compressible_equation_names_keep_their_name(spelling, name):
    deck = _equations_deck(equations=spelling)
    assert deck.equations == name
    assert not deck.incompressible
    assert deck.equation_set == name


@pytest.mark.parametrize("parameters", [{}, {"equations": 3}, {"equations": None}])
def test_missing_or_non_string_equations_is_empty_and_compressible(parameters):
    deck = _equations_deck(**parameters)
    assert deck.equations == ""
    assert not deck.incompressible
    assert deck.equation_set == ""


def test_unknown_incomp_name_is_not_incompressible():
    deck = _equations_deck(equations="incomp euler")
    assert not deck.incompressible
    assert deck.equation_set == "incomp euler"
