"""
Guards the zCFDOutputField registry in output_variables.py:

  - every field a case can write is categorised and (mostly) described;
  - the derived `*_VARIABLES` / `*_ADDITIONS` frozensets still agree with the
    registry they are now computed from;
  - the defaults offered to a UI are themselves valid for the case.

Without these, adding a field would silently leave it undocumented in any UI
reading `get_variable_catalogue()`, or leave a legacy view stale.
"""

import pytest

from zutil.validate.schemas.output_variables import (
    BASE_VARIABLES,
    CATEGORIES_BY_KEY,
    CFD_SURFACE_ADDITIONS,
    CFD_VOLUME_ADDITIONS,
    CONTEXTS,
    DEFAULT_VARIABLES,
    EULER_SURFACE_VARS,
    LES_SURFACE_VARS,
    LES_VOLUME_VARS,
    OUTPUT_FIELDS,
    OUTPUT_FIELDS_BY_NAME,
    RANS_SURFACE_VARS,
    RANS_VOLUME_VARS,
    SURFACE_VARS_BY_EQUATION,
    TIERS,
    TIERS_BY_EQUATION,
    VOLUME_VARS_BY_EQUATION,
    describe_variable,
    get_num_equations,
    get_output_fields,
    get_valid_variables_for_equation_type,
    get_variable_catalogue,
    zCFDOutputField,
)

EQUATION_TYPES = sorted(TIERS_BY_EQUATION)

# Names zCFD's reference guide leaves undefined. This set should only ever get
# shorter.
UNDESCRIBED = set()


class TestTheRegistry:
    def test_every_field_declares_a_known_category(self):
        unknown = {
            f.name: f.category
            for f in OUTPUT_FIELDS
            if f.category not in CATEGORIES_BY_KEY
        }
        assert unknown == {}

    def test_every_field_declares_a_known_tier_somewhere(self):
        for f in OUTPUT_FIELDS:
            tiers = [t for t in (f.volume_tier, f.surface_tier) if t is not None]
            assert tiers, f"{f.name} is written in neither context"
            assert all(t in TIERS for t in tiers), f"{f.name} has an unknown tier"

    def test_names_are_unique(self):
        assert len(OUTPUT_FIELDS_BY_NAME) == len(OUTPUT_FIELDS)

    def test_aliases_point_at_a_registered_field(self):
        dangling = {
            f.name: f.alias_of
            for f in OUTPUT_FIELDS
            if f.alias_of is not None and f.alias_of not in OUTPUT_FIELDS_BY_NAME
        }
        assert dangling == {}

    def test_an_alias_is_available_wherever_its_canonical_name_is(self):
        for f in OUTPUT_FIELDS:
            if f.alias_of is None:
                continue
            canonical = OUTPUT_FIELDS_BY_NAME[f.alias_of]
            for context in CONTEXTS:
                assert f.tier_in(context) == canonical.tier_in(context), (
                    f"{f.name} and {f.alias_of} disagree on {context} availability"
                )

    def test_only_the_documented_gaps_lack_a_description(self):
        missing = {f.name for f in OUTPUT_FIELDS if not f.description}
        assert missing == UNDESCRIBED

    def test_as_dict_omits_alias_of_unless_it_is_an_alias(self):
        assert OUTPUT_FIELDS_BY_NAME["pressure"].as_dict() == {
            "category": "primitive",
            "description": "Static pressure (Pa)",
        }
        assert OUTPUT_FIELDS_BY_NAME["p"].as_dict()["alias_of"] == "pressure"


class TestEveryValidVariableIsDescribed:
    @pytest.mark.parametrize("equation_type", EQUATION_TYPES)
    @pytest.mark.parametrize("context", CONTEXTS)
    def test_no_variable_falls_through_to_other(self, equation_type, context):
        variables = get_valid_variables_for_equation_type(
            equation_type,
            context,
            turbulence_model="sst-transition",
            has_mesh_velocity=True,
            has_fsi=True,
        )
        undocumented = sorted(
            name for name in variables if describe_variable(name)["category"] == "other"
        )
        assert undocumented == []

    def test_the_transition_tier_is_reached_through_the_turbulence_model(self):
        # No equation type inherits the transition tier: only the turbulence
        # model that solves those equations offers them, so a plain RANS case
        # cannot ask for a field its run would never write.
        assert "transition" not in set().union(*TIERS_BY_EQUATION.values())
        for name in ("turbulenceintermittency", "transitionreynoldsnumber"):
            assert OUTPUT_FIELDS_BY_NAME[name].category == "turbulence"
            assert name not in RANS_VOLUME_VARS
            assert name not in get_valid_variables_for_equation_type(
                "rans", "volume", turbulence_model="sst"
            )
            assert name in get_valid_variables_for_equation_type(
                "rans", "volume", turbulence_model="sst-transition"
            )

    def test_an_unregistered_name_is_reported_rather_than_raising(self):
        assert describe_variable("not_a_variable") == {
            "category": "other",
            "description": "",
        }


class TestDerivedViewsMatchTheRegistry:
    """The legacy frozensets are computed from OUTPUT_FIELDS, so these pin the
    tier declarations rather than a hand-maintained list."""

    def test_base_fields_are_available_to_every_equation_type(self):
        for equation_type in EQUATION_TYPES:
            assert BASE_VARIABLES <= set(
                get_valid_variables_for_equation_type(equation_type, "volume")
            )
            assert BASE_VARIABLES <= set(
                get_valid_variables_for_equation_type(equation_type, "surface")
            )

    def test_tiers_are_cumulative_up_the_physics(self):
        assert EULER_SURFACE_VARS < RANS_SURFACE_VARS
        assert CFD_VOLUME_ADDITIONS <= RANS_VOLUME_VARS
        assert CFD_SURFACE_ADDITIONS <= RANS_SURFACE_VARS

    def test_les_and_rans_offer_the_same_set(self):
        assert LES_VOLUME_VARS == RANS_VOLUME_VARS
        assert LES_SURFACE_VARS == RANS_SURFACE_VARS

    def test_incomp_variants_track_their_compressible_equivalents(self):
        for eq in ("viscous", "rans", "les"):
            assert (
                VOLUME_VARS_BY_EQUATION[f"incomp {eq}"] == VOLUME_VARS_BY_EQUATION[eq]
            )
            assert (
                SURFACE_VARS_BY_EQUATION[f"incomp {eq}"] == SURFACE_VARS_BY_EQUATION[eq]
            )

    def test_eddy_needs_a_turbulence_model_in_the_volume_but_not_on_a_wall(self):
        eddy = OUTPUT_FIELDS_BY_NAME["eddy"]
        assert eddy.volume_tier == "les"
        assert eddy.surface_tier == "viscous"
        viscous_volume = set(get_valid_variables_for_equation_type("viscous", "volume"))
        viscous_surface = set(
            get_valid_variables_for_equation_type("viscous", "surface")
        )
        assert "eddy" not in viscous_volume
        assert "eddy" in viscous_surface

    def test_modal_fields_appear_only_with_fsi(self):
        without = set(get_valid_variables_for_equation_type("rans", "surface"))
        with_fsi = set(
            get_valid_variables_for_equation_type("rans", "surface", has_fsi=True)
        )
        assert "modeshapedisplacement_14" not in without
        assert "meshvelocity" not in without
        assert "modeshapedisplacement_14" in with_fsi
        assert "meshvelocity" in with_fsi
        # Volume gets mesh velocity but no mode shapes.
        volume = set(
            get_valid_variables_for_equation_type("rans", "volume", has_fsi=True)
        )
        assert "meshvelocity" in volume
        assert "modaldeformation" not in volume

    def test_mesh_velocity_alone_offers_no_modal_fields(self):
        moving = set(
            get_valid_variables_for_equation_type(
                "rans", "surface", has_mesh_velocity=True
            )
        )
        assert "meshvelocity" in moving
        assert "modaldeformation" not in moving
        assert "modeshapedisplacement_1" not in moving
        assert "modeshaperotation_1" not in moving


class TestGetOutputFields:
    def test_returns_fields_in_name_order_without_repeats(self):
        fields = get_output_fields(
            "rans", "surface", "sst", has_mesh_velocity=True, has_fsi=True
        )
        names = [f.name for f in fields]
        assert names == sorted(names)
        assert len(names) == len(set(names))

    def test_agrees_with_the_name_list_it_backs(self):
        args = ("rans", "volume", "sa-neg", True)
        assert [f.name for f in get_output_fields(*args)] == sorted(
            get_valid_variables_for_equation_type(*args)
        )

    def test_every_returned_field_is_a_zcfd_output_field(self):
        for f in get_output_fields("rans", "volume", "sst"):
            assert isinstance(f, zCFDOutputField)

    def test_unknown_equation_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown equation type"):
            get_output_fields("magic", "surface")


class TestDynamicFields:
    @pytest.mark.parametrize(
        "equation_type,turbulence_model,expected",
        [
            ("euler", None, 5),
            ("viscous", None, 5),
            ("rans", None, 7),
            ("rans", "sst", 7),
            ("rans", "sa-neg", 6),
            ("rans", "sst-transition", 9),
            ("incomp rans", "sst-transition", 9),
            # A turbulence model on non-RANS equations does not move the count.
            ("les", "sst-transition", 5),
        ],
    )
    def test_equation_count_drives_the_indices(
        self, equation_type, turbulence_model, expected
    ):
        assert get_num_equations(equation_type, turbulence_model) == expected
        names = set(
            get_valid_variables_for_equation_type(
                equation_type, "volume", turbulence_model
            )
        )
        assert f"var_{expected}" in names
        assert f"var_{expected + 1}" not in names
        assert f"resvar_{expected}" in names
        assert f"resshare_{expected}" in names
        assert f"resshare_{expected + 1}" not in names
        assert f"vargrad_{expected}" in names

    def test_are_described_per_equation(self):
        catalogue = get_variable_catalogue("rans", "volume", "sst-transition")
        assert catalogue["metadata"]["var_9"] == {
            "category": "solver",
            "description": "Solution value of solved equation 9",
        }
        assert "var_10" not in catalogue["metadata"]


class TestDefaults:
    @pytest.mark.parametrize("context", CONTEXTS)
    def test_defaults_are_valid_for_rans(self, context):
        offered = set(get_valid_variables_for_equation_type("rans", context))
        assert set(DEFAULT_VARIABLES[context]) <= offered

    def test_euler_surface_drops_the_viscous_defaults(self):
        catalogue = get_variable_catalogue("euler", "surface")
        assert "yplus" not in catalogue["defaults"]
        assert "ut" not in catalogue["defaults"]
        assert "pressure" in catalogue["defaults"]

    def test_defaults_are_offered_by_the_catalogue_they_come_with(self):
        for equation_type in EQUATION_TYPES:
            for context in CONTEXTS:
                catalogue = get_variable_catalogue(equation_type, context)
                assert set(catalogue["defaults"]) <= set(catalogue["variables"])

    def test_defaults_are_canonical_names_not_aliases(self):
        for names in DEFAULT_VARIABLES.values():
            for name in names:
                assert OUTPUT_FIELDS_BY_NAME[name].alias_of is None


class TestGetVariableCatalogue:
    def test_agrees_with_the_valid_set_it_documents(self):
        catalogue = get_variable_catalogue(
            "rans", "volume", "sa-neg", has_mesh_velocity=True, has_fsi=True
        )
        assert catalogue["variables"] == sorted(
            get_valid_variables_for_equation_type(
                "rans", "volume", "sa-neg", True, True
            )
        )
        assert set(catalogue["metadata"]) == set(catalogue["variables"])

    def test_categories_are_those_in_use_in_display_order(self):
        catalogue = get_variable_catalogue("euler", "surface")
        used = {meta["category"] for meta in catalogue["metadata"].values()}
        keys = [c["key"] for c in catalogue["categories"]]
        assert set(keys) == used
        assert "turbulence" not in keys  # euler offers no turbulence variables
        assert keys.index("primitive") < keys.index("solver")

    def test_fsi_surface_variables_are_listed_once(self):
        # cell_centre is both a base field and an FSI one; the registry holds
        # a single entry for it, so it cannot be served twice.
        catalogue = get_variable_catalogue("rans", "surface", "sst", has_fsi=True)
        assert catalogue["variables"].count("cell_centre") == 1

    def test_unknown_equation_type_still_rejects(self):
        with pytest.raises(ValueError, match="Unknown equation type"):
            get_variable_catalogue("magic", "surface")
