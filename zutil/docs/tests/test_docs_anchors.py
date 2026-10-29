"""Tests for reference routes and anchors (full key path from the entry)."""

import pytest

from zutil.docs import anchors as anchors_module
from zutil.docs.anchors import anchor, fragment, kebab, normalise_key, reference_url
from zutil.docs.build import anchor_collisions, walk_entries
from zutil.docs.tree import Walker


@pytest.fixture(scope="module")
def pages():
    return {p.route: p for p in Walker().build()}


def anchors_on(page):
    return {e["anchor"] for e in walk_entries(page.entries)}


class TestAnchorRule:
    def test_underscores_and_spaces_become_hyphens(self):
        assert kebab("heat_flux") == "heat-flux"
        assert kebab("Heat Flux") == "heat-flux"

    def test_key_path_joined_by_hyphens(self):
        assert fragment(["cfl", "cfl_turbulence"]) == "cfl-cfl-turbulence"
        assert fragment(["turbulence", "model"]) == "turbulence-model"

    def test_deck_spelling_folds_to_schema_key(self):
        assert normalise_key("Cfl Ramp") == "cfl_ramp"


class TestRealTree:
    def test_every_block_is_its_own_page(self, pages):
        assert "frequency-volume-data" not in anchors_on(
            pages["reference/solver/output-settings/solution"]
        )
        assert "volume-data" in anchors_on(
            pages["reference/solver/output-settings/solution/frequency"]
        )
        assert "zones" in anchors_on(
            pages["reference/solver/output-settings/report/forces"]
        )

    def test_no_page_nests_a_block_inline(self, pages):
        nested = [
            (p.route, e["key"])
            for p in pages.values()
            if p.kind != "output-variables"
            for e in walk_entries(p.entries)
            if e["fields"]
        ]
        assert nested == []

    def test_union_tag_adds_no_segment(self, pages):
        assert "omega" in anchors_on(
            pages["reference/model/boundary-conditions/wall/velocity"]
        )

    def test_list_items_add_no_segment(self, pages):
        ramp = pages["reference/solver/convergence-control/cfl/cfl-ramp"]
        assert ramp.deck_path[-1] == "<n>"
        assert "initial" in anchors_on(ramp)

    def test_instance_names_dropped(self, pages):
        assert "temperature" in anchors_on(
            pages["reference/solver/reference-conditions"]
        )

    def test_type_value_is_a_route_segment(self, pages):
        assert "reference/model/boundary-conditions/inflow" in pages
        assert "reference/model/fluid-zones/disc" in pages
        assert "reference/model/fluid-zones/porous" in pages
        assert "reference/model/motion/rotating" not in pages

    def test_override_nodes_are_rows_under_model(self, pages):
        rows = {e["key"]: e for e in pages["reference/model"].entries}
        assert (
            rows["convergence_control"]["link"]
            == "reference/solver/convergence-control"
        )
        assert rows["convergence_control"]["facts"]["overridable_per_model"]

    def test_output_variables_canonical_only(self, pages):
        page = pages["reference/output-variables"]
        names = anchors_on(page)
        assert "velocity" in names
        assert "v" not in names
        velocity = next(e for e in page.entries if e["key"] == "velocity")
        assert set(velocity["aliases"]) == {"V", "v"}
        assert {
            "from": "reference/output-variables#v",
            "field": "velocity",
        } in page.facts["aliases"]

    def test_no_anchor_collisions(self, pages):
        assert anchor_collisions(list(pages.values())) == []


def deck_paths(page):
    """Every (deck key path, expected anchor) the page emits, instance names filled in."""
    instance = {"<mesh>": "m1", "<name>": "n1", "<n>": "0"}
    roots = [page.deck_path] + page.extra_deck_paths
    for root in roots:
        deck = [instance.get(k, k) for k in root]
        yield from _entry_paths(page, page.entries, deck)


def _entry_paths(page, entries, deck):
    for entry in entries:
        path = deck + [entry["key"]]
        yield path, entry.get("link") or f"{page.route}#{entry['anchor']}"
        inner = path + ["0"] if entry["facts"].get("container") else path
        yield from _entry_paths(page, entry.get("fields") or [], inner)


def page_type(page):
    return page.type_value


class TestPublicAnchor:
    def test_every_generated_entry_round_trips(self, pages):
        wrong = [
            (page.route, ".".join(path), expected, anchor(path, page_type(page)))
            for page in pages.values()
            if page.kind != "output-variables"
            for path, expected in deck_paths(page)
            if anchor(path, page_type(page)) != expected
        ]
        assert wrong == []

    def test_every_page_root_resolves_to_its_route(self, pages):
        instance = {"<mesh>": "m1", "<name>": "n1", "<n>": "0"}
        for page in pages.values():
            if page.kind == "output-variables" or not page.deck_path:
                continue
            deck = [instance.get(k, k) for k in page.deck_path]
            assert anchor(deck, page_type(page)) == page.route

    def test_examples_from_the_deep_links_answer(self):
        assert (
            anchor("solver.convergence_control.scheme.krylov_subspace")
            == "reference/solver/convergence-control/scheme#krylov-subspace"
        )
        assert (
            anchor("model.m1.boundary_conditions.bc_1.velocity.omega", "wall")
            == "reference/model/boundary-conditions/wall/velocity#omega"
        )
        assert (
            anchor("model.m1.convergence_control.cfl.cfl_turbulence")
            == "reference/solver/convergence-control/cfl#cfl-turbulence"
        )

    def test_deck_spelling_is_accepted(self):
        assert (
            anchor(["solver", "Equations", "Turbulence", "Model"])
            == "reference/solver/equations/turbulence#model"
        )

    def test_fluid_zone_instance_name_dropped(self):
        assert (
            anchor("model.m1.fluid_zones.fz_1.axis", "rotating")
            == "reference/model/fluid-zones/rotating#axis"
        )

    def test_motion_is_a_page_under_transforms(self):
        assert (
            anchor("model.m1.transforms.motion.axis", "rotating")
            == "reference/model/transforms/motion#axis"
        )
        assert (
            anchor("model.m1.transforms.motion.velocity")
            == "reference/model/transforms/motion#velocity"
        )
        assert (
            anchor("model.m1.transforms.motion.moving_mesh")
            == "reference/model/transforms/motion#moving-mesh"
        )
        assert (
            anchor("model.m1.transforms.transform")
            == "reference/model/transforms#transform"
        )

    def test_unknown_key_anchors_its_parent(self):
        assert (
            anchor("solver.equations.turbulence.no_such_key")
            == "reference/solver/equations/turbulence"
        )

    def test_missing_type_falls_back_to_the_type_index(self):
        assert (
            anchor("model.m1.boundary_conditions.bc_1.temperature")
            == "reference/model/boundary-conditions"
        )

    def test_empty_path_has_no_anchor(self):
        assert anchor("") is None


class TestReferenceUrl:
    def test_release_tag_gets_a_frozen_folder(self):
        assert (
            reference_url(
                "reference/solver/equations/turbulence#model", "2026.08.14860"
            )
            == "https://docs.zenotech.com/v2026.08.14860/reference/solver/equations/turbulence#model"
        )

    def test_tag_spelled_with_its_v_is_not_doubled(self):
        assert reference_url("reference/model", "v2026.08.14860").startswith(
            "https://docs.zenotech.com/v2026.08.14860/"
        )

    @pytest.mark.parametrize(
        "version",
        ["2026.08.14860-39-g05b2c2a61c", "2026.08.14860-dirty", "0.0.unknown", ""],
    )
    def test_non_release_build_uses_the_root(self, version):
        assert (
            reference_url("reference/model", version)
            == "https://docs.zenotech.com/reference/model"
        )

    def test_no_solver_uses_the_root(self, monkeypatch):
        monkeypatch.setattr(anchors_module, "zcfd_version", lambda: None)
        assert (
            reference_url("reference/model")
            == "https://docs.zenotech.com/reference/model"
        )
