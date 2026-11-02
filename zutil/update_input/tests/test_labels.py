from zutil.update_input.model import Conversion, Leaf, V1Deck, Value
from zutil.update_input.rules import boundaries


def _deck(bc: dict) -> V1Deck:
    evaluated = {"ic_1": {"temperature": 288.0}, "bc_1": bc}
    leaves = [Leaf(("ic_1", "temperature"), 288.0, "288.0")]
    leaves += [Leaf(("bc_1", k), v, repr(v)) for k, v in bc.items()]
    return V1Deck(leaves, evaluated)


def test_farfield_condition_is_respelt_to_the_ic_label():
    deck = _deck(
        {"type": "farfield", "kind": "riemann", "condition": "IC_1", "zone": [1]}
    )
    c = Conversion(deck, "m")
    boundaries.convert(c)
    path = ("model", "m", "boundary conditions", "bc_1", "condition")
    assert c.v2[path] == Value("ic_1")


def test_inflow_condition_naming_a_plain_ic_is_respelt_to_the_label():
    deck = _deck(
        {"type": "inflow", "kind": "default", "condition": "IC_1", "zone": [2]}
    )
    c = Conversion(deck, "m")
    boundaries.convert(c)
    path = ("model", "m", "boundary conditions", "bc_1", "reference")
    assert c.v2[path] == Value("ic_1")
