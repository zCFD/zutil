"""Tests for child pages, the output-variables page and option applicability."""

import pytest

from zutil.docs.anchors import anchor
from zutil.docs.build import add_outlines, walk_entries
from zutil.docs.tree import Walker

SOLUTION = "reference/solver/output-settings/solution"


@pytest.fixture(scope="module")
def pages():
    built = Walker().build()
    add_outlines(built)
    return {p.route: p for p in built}


def entry(page, anchor_text):
    return next(e for e in walk_entries(page.entries) if e["anchor"] == anchor_text)


FREQUENCY = f"{SOLUTION}/frequency"
RAMP = "reference/solver/convergence-control/cfl/cfl-ramp"


class TestChildPages:
    def test_a_block_gets_its_own_page_at_any_depth(self, pages):
        child = pages[FREQUENCY]
        assert child.parent == SOLUTION
        assert pages[SOLUTION].parent == "reference/solver/output-settings"
        assert child.deck_path == ["solver", "output_settings", "solution", "frequency"]
        assert child.title == "frequency"
        assert "volume-data" in {e["anchor"] for e in child.entries}

    def test_the_parent_keeps_a_linked_row(self, pages):
        row = entry(pages[SOLUTION], "frequency")
        assert row["link"] == FREQUENCY
        assert row["fields"] == []

    def test_a_collection_page_documents_one_instance(self, pages):
        assert pages[RAMP].deck_path[-2:] == ["cfl_ramp", "<n>"]
        assert pages[RAMP].facts["container"] == "list"
        monitor = pages["reference/solver/output-settings/report/monitor"]
        assert monitor.deck_path[-2:] == ["monitor", "<name>"]
        assert monitor.facts["container"] == "named"

    def test_a_union_page_merges_its_members(self, pages):
        velocity = pages["reference/model/boundary-conditions/wall/velocity"]
        assert len(velocity.variants) > 1
        assert entry(velocity, "type")["options"]

    def test_every_moved_anchor_has_an_alias(self, pages):
        aliases = {
            a["from"]: a for p in pages.values() for a in p.facts.get("aliases", [])
        }
        assert aliases[
            "reference/solver/output-settings#solution-frequency-volume-data"
        ] == {
            "from": "reference/solver/output-settings#solution-frequency-volume-data",
            "route": FREQUENCY,
            "field": "volume-data",
        }
        assert aliases[f"{SOLUTION}#frequency-volume-data"]["route"] == FREQUENCY

    def test_every_alias_lands_on_a_live_anchor(self, pages):
        live = {f"{p.route}#{e['anchor']}" for p in pages.values() for e in p.entries}
        dead = [
            a
            for p in pages.values()
            for a in p.facts.get("aliases", [])
            if "route" in a and f"{a['route']}#{a['field']}" not in live
        ]
        assert dead == []

    def test_deck_paths_resolve_to_the_child_page(self):
        assert (
            anchor("solver.output_settings.solution.frequency.volume_data")
            == f"{FREQUENCY}#volume-data"
        )
        assert anchor("solver.output_settings.solution") == SOLUTION
        assert anchor("model.m1.output_settings.solution.frequency") == FREQUENCY
        assert (
            anchor("solver.convergence_control.cfl.cfl_ramp.0.initial")
            == f"{RAMP}#initial"
        )

    def test_a_child_of_a_type_page_keeps_its_type(self, pages):
        fsi = pages["reference/model/boundary-conditions/wall/fsi"]
        assert fsi.type is None
        assert fsi.type_value == "wall"
        assert (
            anchor("model.m1.boundary_conditions.bc.fsi.zone", "wall")
            == "reference/model/boundary-conditions/wall/fsi#zone"
        )

    def test_a_child_outline_links_its_block_to_itself(self, pages):
        outline = pages["reference/model/boundary-conditions/wall/fsi"].outline
        node = outline
        while len(node["children"]) == 1 and node["children"][0].get("children"):
            node = node["children"][0]
        assert node["key"] == "fsi"
        assert node["link"] == "reference/model/boundary-conditions/wall/fsi"


class TestOutlines:
    def test_a_child_block_is_folded_and_links_to_its_page(self, pages):
        node = pages[SOLUTION].outline
        while len(node["children"]) == 1 and node["children"][0].get("children"):
            node = node["children"][0]
        frequency = next(c for c in node["children"] if c["key"] == "frequency")
        assert frequency["value"] == "{...}"
        assert frequency["link"] == FREQUENCY

    def test_a_list_page_draws_one_item_of_a_list(self, pages):
        node = pages[RAMP].outline
        while not node.get("list"):
            node = node["children"][0]
        assert node["key"] == "cfl ramp"
        (item,) = node["children"]
        assert item["key"] is None
        assert "initial" in [c["key"] for c in item["children"]]


class TestOptionApplicability:
    def test_an_option_offered_by_one_solver_type_says_so(self, pages):
        name = entry(pages["reference/solver/convergence-control/scheme"], "name")
        only = {o["value"]: o["only"] for o in name["options"]}
        assert only["coupled"] == "incompressible only"
        assert only["simple"] == "incompressible only"
        assert only["implicit euler"] == "compressible only"

    def test_an_option_that_only_names_its_own_block_has_no_label(self, pages):
        kind = entry(pages["reference/solver/time-settings"], "kind")
        assert {o["only"] for o in kind["options"]} == {None}
        velocity = entry(
            pages["reference/model/boundary-conditions/wall/velocity"], "type"
        )
        assert {o["only"] for o in velocity["options"]} == {None}


class TestDiscriminatorDescriptions:
    @pytest.mark.parametrize(
        "route, anchor_text",
        [
            ("reference/model/boundary-conditions/wall/velocity", "type"),
            ("reference/model/boundary-conditions/wall/fsi", "type"),
            (RAMP, "type"),
            ("reference/solver/time-settings", "kind"),
        ],
    )
    def test_is_described(self, pages, route, anchor_text):
        found = entry(pages[route], anchor_text)
        assert found["description"]
        assert not found["gap"]


class TestOutputVariables:
    @pytest.fixture(scope="class")
    def page(self, pages):
        return pages["reference/output-variables"]

    def test_lead_names_no_internal_class(self, page):
        assert "zCFDOutputField" not in page.description
        assert "`volume variables`" in page.description

    def test_a_numbered_family_is_one_row(self, page):
        keys = [e["key"] for e in page.entries]
        assert "modeshapedisplacement_<n>" in keys
        assert not any(k.startswith("modeshapedisplacement_1") for k in keys)
        family = next(
            e for e in page.entries if e["key"] == "modeshapedisplacement_<n>"
        )
        assert family["anchor"] == "modeshapedisplacement-n"
        assert family["description"] == "Displacement of mode shape `<n>`"
        assert family["pattern"] == "Replace `<n>` with a number from 1 to 14."

    def test_a_per_axis_family_is_one_row(self, page):
        keys = [e["key"] for e in page.entries]
        assert "pressuremoment<axis>" in keys
        assert "pressuremomentx" not in keys
        assert "pressuremoment" in keys

    def test_family_members_alias_to_the_family(self, page):
        assert {
            "from": "reference/output-variables#modeshaperotation-10",
            "field": "modeshaperotation-n",
        } in page.facts["aliases"]

    def test_every_variable_states_its_context(self, page):
        contexts = {e["key"]: e["facts"]["context"] for e in page.entries}
        assert contexts["pressure"] == "both"
        assert contexts["vorticity"] == "volume"
        assert contexts["yplus"] == "surface"
        assert set(contexts.values()) == {"both", "volume", "surface"}
