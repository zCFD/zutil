"""Tests for side-file parsing, key equality and fingerprint stamping."""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from zutil.docs import sidefiles
from zutil.docs.build import DRIFT, BuildResult
from zutil.docs.introspect import fingerprint
from zutil.validate.schemas.base_types import zCFD_Field


class Plate(BaseModel):
    type: Literal["plate"] = "plate"
    kind: Literal["no slip", "slip"] = zCFD_Field("no slip", description="Wall kind")
    heat_flux: float = zCFD_Field(0.0, units="W/m^2", description="Heat flux")


class Base(BaseModel):
    zones: list = []


class Child(Base):
    extra: int = 1


COMPLETE = """# Plate

## kind
fingerprint: pending
guide: /guide/choosing/choosing-a-wall-model#kind

### no slip
what: Velocity is zero at the wall.
when: Resolved boundary layers.

### slip
what: Tangential velocity is free.

## heat flux
fingerprint: pending
"""


def check(text, models=(Plate,)):
    return sidefiles.check(sidefiles.parse(text), list(models))


class TestParse:
    def test_attributes_and_options(self):
        side = sidefiles.parse(COMPLETE)
        kind = side.sections["Plate"].keys["kind"]
        assert kind.attrs["guide"] == "/guide/choosing/choosing-a-wall-model#kind"
        assert kind.options["no slip"].attrs["when"] == "Resolved boundary layers."

    def test_spaced_key_matches_schema_key(self):
        assert "heat_flux" in sidefiles.parse(COMPLETE).sections["Plate"].keys

    def test_prose_continues_across_lines(self):
        side = sidefiles.parse(
            "# Plate\n\n## kind\n\n### slip\nwhat: One\ntwo.\n\nThree.\n"
        )
        assert (
            side.sections["Plate"].keys["kind"].options["slip"].attrs["what"]
            == "One two.\n\nThree."
        )

    def test_formatter_escapes_in_headings_are_ignored(self):
        side = sidefiles.parse("# \\_Base\n\n## zones\n\nfingerprint: pending\n")
        assert "zones" in side.sections["_Base"].keys

    def test_text_outside_an_attribute_is_unplaced(self):
        report = check(
            COMPLETE.replace(
                "## heat flux\nfingerprint: pending\n",
                "## heat flux\nfingerprint: pending\n::: warning\n",
            )
        )
        assert report["unplaced_text"][0]["lines"] == ["::: warning"]


class TestKeyEquality:
    def test_complete_file_is_clean(self):
        assert all(not v for v in check(COMPLETE).values())

    def test_an_entry_may_be_left_out(self):
        assert all(
            not v
            for v in check(
                COMPLETE.replace("## heat flux\nfingerprint: pending\n", "")
            ).values()
        )

    def test_extra_key(self):
        report = check(COMPLETE + "\n## colour\nfingerprint: pending\n")
        assert [k["key"] for k in report["extra_keys"]] == ["colour"]

    def test_an_option_may_be_left_out_but_not_invented(self):
        report = check(COMPLETE.replace("### slip", "### sticky"))
        assert [o["option"] for o in report["extra_options"]] == ["sticky"]

    def test_unknown_section(self):
        report = check(COMPLETE + "\n# Ghost\n\n## x\n")
        assert report["unknown_sections"] == [{"file": "<memory>", "section": "Ghost"}]

    def test_fixed_discriminator_entry_is_allowed(self):
        assert (
            check(COMPLETE + "\n## type\nfingerprint: pending\n\n### plate\n")[
                "extra_keys"
            ]
            == []
        )

    def test_inherited_keys_belong_to_the_base(self):
        report = check(
            "# Child\n\n## extra\nfingerprint: pending\n\n# Base\n\n## zones\nfingerprint: pending\n",
            (Base, Child),
        )
        assert all(not v for v in report.values())
        report = check("# Child\n\n## extra\n\n## zones\n", (Base, Child))
        assert {"file": "<memory>", "model": "Child", "key": "zones"} in report[
            "extra_keys"
        ]


class TestFingerprint:
    def test_changes_with_facts(self):
        class Other(BaseModel):
            heat_flux: float = zCFD_Field(
                0.0, units="W/m^2", description="Different words"
            )

        class Moved(BaseModel):
            heat_flux: float = zCFD_Field(1.0, units="W/m^2", description="Heat flux")

        base = fingerprint(Plate.model_fields["heat_flux"], "heat_flux")
        assert fingerprint(Other.model_fields["heat_flux"], "heat_flux") == base
        assert fingerprint(Moved.model_fields["heat_flux"], "heat_flux") != base

    def test_stale_fingerprint_reported(self):
        report = check(
            COMPLETE.replace(
                "## heat flux\nfingerprint: pending",
                "## heat flux\nfingerprint: 000000000000",
            )
        )
        assert report["stale_fingerprints"][0]["key"] == "heat_flux"

    def test_restamp_names_one_entry(self):
        text, stamped = sidefiles.restamp(
            COMPLETE, {("Plate", "heat_flux"): "abc123abc123"}
        )
        assert stamped == [("Plate", "heat_flux")]
        assert "## heat flux\nfingerprint: abc123abc123\n" in text
        assert "## kind\nfingerprint: pending\n" in text

    def test_restamp_inserts_a_missing_line(self):
        text, _ = sidefiles.restamp(
            "# Plate\n\n## kind\n\n### slip\n", {("Plate", "kind"): "abc"}
        )
        assert text.startswith("# Plate\n\n## kind\nfingerprint: abc\n")

    def test_new_file_mode_leaves_stamped_entries(self):
        stamped_once = COMPLETE.replace(
            "## kind\nfingerprint: pending", "## kind\nfingerprint: 111111111111"
        )
        text, stamped = sidefiles.restamp(
            stamped_once,
            {("Plate", "kind"): "x", ("Plate", "heat_flux"): "y"},
            only_pending=True,
        )
        assert stamped == [("Plate", "heat_flux")]
        assert "fingerprint: 111111111111" in text


class TestBuildFailsOnDrift:
    class _Walker:
        unresolved = []

    def result(self, **drift):
        mismatches = {k: drift.get(k, []) for k in DRIFT}
        mismatches["pending_fingerprints"] = 0
        return BuildResult([], self._Walker(), {}, mismatches, [], [])

    def test_clean_side_files_pass(self):
        assert not self.result().failed

    def test_each_kind_of_drift_fails(self):
        for kind in DRIFT:
            assert self.result(**{kind: [{"file": "x.md"}]}).failed, kind


class Defined(BaseModel):
    kind: Literal["no slip", "slip"] = zCFD_Field(
        "no slip",
        description="Wall kind",
        options={"no slip": "Velocity is zero at the wall."},
    )


class Invented(BaseModel):
    kind: Literal["no slip", "slip"] = zCFD_Field(
        "no slip", description="Wall kind", options={"moving": "Not a kind."}
    )


def option_rows(model, side_text=""):
    from zutil.docs.tree import Walker
    from zutil.docs.introspect import Variant

    side = sidefiles.parse(side_text, module=model.__module__)
    walker = Walker({model.__module__: side})
    variant = Variant(model)
    rows = walker.options("kind", [(variant, model.model_fields["kind"])], [variant])
    return {row["value"]: row for row in rows}


class TestSchemaOptionDefinitions:
    """An option's one-sentence definition lives in the schema; the side file
    keeps only when-to-use guidance beside it."""

    def test_definition_reaches_the_json_schema(self):
        node = Defined.model_json_schema()["properties"]["kind"]
        assert node["zcfd"]["options"] == {"no slip": "Velocity is zero at the wall."}

    def test_reference_reads_the_schema_definition(self):
        rows = option_rows(
            Defined, "# Defined\n\n## kind\n\n### no slip\nwhen: Resolved layers.\n"
        )
        assert rows["no slip"]["what"] == "Velocity is zero at the wall."
        assert rows["no slip"]["when"] == "Resolved layers."

    def test_side_file_definition_is_the_fallback(self):
        rows = option_rows(Defined, "# Defined\n\n## kind\n\n### slip\nwhat: Free.\n")
        assert rows["slip"]["what"] == "Free."

    def test_a_side_file_may_not_restate_a_schema_definition(self):
        report = check(
            "# Defined\n\n## kind\n\n### no slip\nwhat: Zero velocity.\n", (Defined,)
        )
        assert [r["option"] for r in report["restated_options"]] == ["no slip"]

    def test_when_beside_a_schema_definition_is_clean(self):
        report = check(
            "# Defined\n\n## kind\n\n### no slip\nwhen: Resolved layers.\n", (Defined,)
        )
        assert not any(report.values())

    def test_definition_for_a_value_the_field_lacks_is_reported(self):
        from zutil.docs.introspect import unknown_option_definitions

        assert unknown_option_definitions(Invented) == [
            {"model": "Invented", "key": "kind", "option": "moving"}
        ]
        assert unknown_option_definitions(Defined) == []


class Patch(BaseModel):
    field: str


class Heated(BaseModel):
    temperature: Optional[Union[Annotated[float, Field(gt=0.0)], Patch]] = zCFD_Field(
        None, units="K", description="Wall temperature"
    )
    heat_flux: Optional[Union[float, Patch]] = zCFD_Field(None, units="W/m^2")


class TestBoundsInANumberOrBlockUnion:
    def test_the_numbers_bound_is_the_fields_bound(self):
        from zutil.docs.introspect import bounds, facts_of

        fi = Heated.model_fields["temperature"]
        assert bounds(fi) == {"gt": 0.0}
        assert facts_of(fi, "temperature")["bounds_text"] == "> 0.0 K"

    def test_an_unbounded_union_has_none(self):
        from zutil.docs.introspect import bounds

        assert bounds(Heated.model_fields["heat_flux"]) == {}


def _page(route, *keys, model="WallBoundary"):
    from zutil.docs.tree import Page

    page = Page(route, route, "type", None, 1, [])
    page.entries = [
        {
            "key": key,
            "anchor": key.replace("_", "-"),
            "fields": [],
            "declared_in": {"module": "walls", "model": model},
        }
        for key in keys
    ]
    return page


WALL = "reference/model/boundary-conditions/wall"
INFLOW = "reference/model/boundary-conditions/inflow"


def lifted(side_text, pages):
    from zutil.docs.build import lift_aliases

    side = sidefiles.parse(side_text, module="zutil.schemas.walls")
    return lift_aliases(pages, {"zutil.schemas.walls": side})


class TestWasAliases:
    def test_several_was_lines_are_kept_in_order(self):
        key = (
            sidefiles.parse("# Plate\n\n## kind\nwas: #wall-kind\nwas: #type\n")
            .sections["Plate"]
            .keys["kind"]
        )
        assert key.was == ["#wall-kind", "#type"]

    def test_was_under_an_option_is_unplaced(self):
        report = check(COMPLETE.replace("### slip\n", "### slip\nwas: #old\n"))
        assert report["unplaced_text"][0]["lines"] == ["was: #old"]

    def test_same_page_rename_keeps_the_page(self):
        aliases, errors = lifted(
            f"# WallBoundary\n\n## roughness\nwas: {WALL}#roughness-height\n",
            [_page(WALL, "roughness")],
        )
        assert errors == []
        assert aliases == [{"from": f"{WALL}#roughness-height", "field": "roughness"}]

    def test_fragment_alone_aliases_every_page_that_shows_the_key(self):
        aliases, errors = lifted(
            "# BoundaryConditionBase\n\n## zones\nwas: #zone-list\n",
            [
                _page(WALL, "zones", model="BoundaryConditionBase"),
                _page(INFLOW, "zones", model="BoundaryConditionBase"),
            ],
        )
        assert errors == []
        assert aliases == [
            {"from": f"{WALL}#zone-list", "field": "zones"},
            {"from": f"{INFLOW}#zone-list", "field": "zones"},
        ]

    def test_a_move_across_pages_names_the_new_page(self):
        aliases, _ = lifted(
            "# WallBoundary\n\n## roughness\nwas: reference/solver/equations#roughness\n",
            [_page(WALL, "roughness")],
        )
        assert aliases == [
            {
                "from": "reference/solver/equations#roughness",
                "route": WALL,
                "field": "roughness",
            }
        ]

    def test_a_move_onto_a_key_shown_on_several_pages_fails(self):
        _, errors = lifted(
            "# BoundaryConditionBase\n\n## zones\nwas: reference/model#zones\n",
            [
                _page(WALL, "zones", model="BoundaryConditionBase"),
                _page(INFLOW, "zones", model="BoundaryConditionBase"),
            ],
        )
        assert len(errors) == 1 and "no single target" in errors[0]

    def test_a_malformed_was_line_fails(self):
        _, errors = lifted(
            "# WallBoundary\n\n## roughness\nwas: roughness height\n",
            [_page(WALL, "roughness")],
        )
        assert len(errors) == 1 and "is not an old anchor" in errors[0]

    def test_a_key_no_page_shows_fails(self):
        _, errors = lifted(
            "# WallBoundary\n\n## roughness\nwas: #old\n", [_page(WALL, "kind")]
        )
        assert len(errors) == 1 and "no reference page shows" in errors[0]

    def test_alias_errors_fail_the_build(self):
        mismatches = {k: [] for k in DRIFT}
        result = BuildResult(
            [], TestBuildFailsOnDrift._Walker(), {}, mismatches, [], [], [], ["x"]
        )
        assert result.failed
