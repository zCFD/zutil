"""A key most published decks set does not list the cases that set it."""

from zutil.docs.examples import Deck, derive


def _entry(key):
    return {
        "key": key,
        "facts": {
            "default": {"value": None, "show": False},
            "required": False,
            "container": None,
        },
        "fields": [],
        "declared_in": {"model": "M", "module": "m"},
        "anchor": key,
    }


def test_widely_used_follows_the_share_of_the_corpus():
    decks = [Deck(f"case{i}", "c.py", None, {}) for i in range(4)]
    blocks = [
        (d, {"common": 1, "rare": 2} if i == 0 else {"common": 1})
        for i, d in enumerate(decks)
    ]
    common, rare = _entry("common"), _entry("rare")
    derive([common, rare], blocks, [], "reference/x", len(decks))
    assert common["widely_used"] is True
    assert rare["widely_used"] is False
    assert rare["examples"][0]["source"]["case"] == "case0"
