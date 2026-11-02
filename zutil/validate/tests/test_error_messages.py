"""Validation errors are read by whoever is filling in the form.

They must name the field, say what to do about it, and never quote a
Python type or a pydantic error code.
"""

import copy

import pytest

from zutil.docs import anchors
from zutil.validate.error_mapping import (
    _clean_loc_parts,
    _humanise_message,
    format_errors_as_text,
    map_errors_to_nodes,
)
from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive


def base_deck():
    return {
        "config version": 2,
        "solver": {
            "solver settings": {"type": "compressible"},
            "equations": {"type": "rans", "turbulence": {"model": "sst"}},
            "fluid properties": {"material": "air"},
            "reference conditions": {
                "ic_1": {
                    "temperature": 300.0,
                    "pressure": 101325.0,
                    "v": [0.0, 0.0, 0.0],
                    "turbulence intensity": 0.05,
                    "eddy viscosity ratio": 1.0,
                    "reynolds no": 1.0e6,
                }
            },
            "initialisation": {"initial conditions": "ic_1"},
            "convergence control": {"cycles": 10, "cfl": {"cfl": 2.0}},
            "numerical scheme": {"order": "second"},
            "time settings": {"type": "steady"},
            "output settings": {
                "solution": {"frequency": 1},
                "report": {"frequency": 1},
            },
        },
        "model": {
            "m1": {
                "mesh": "m.h5",
                "boundary conditions": {
                    "bc_1": {"zones": [1], "type": "wall", "kind": "no slip"}
                },
            }
        },
    }


def messages(deck):
    _model, nodes = validate_zcfd_comprehensive(deck)
    return [n["display_message"] for n in nodes]


class TestZeroVelocityIsValid:
    """A stationary reference condition is the schema's own default: the
    flow can be driven by the boundaries or by a moving model."""

    def test_zero_vector_validates(self):
        assert messages(base_deck()) == []

    def test_zero_vector_is_the_default(self):
        deck = base_deck()
        del deck["solver"]["reference conditions"]["ic_1"]["v"]
        assert messages(deck) == []


class TestMessagesAreEnglish:
    def test_a_vector_given_as_text_says_what_a_vector_looks_like(self):
        deck = base_deck()
        deck["solver"]["reference conditions"]["ic_1"]["v"] = "0 0 0"
        (message,) = messages(deck)
        assert "must be 3 numbers" in message
        assert "Vector3d" not in message
        assert "Value error" not in message

    def test_union_branches_collapse_into_one_sentence(self):
        """`v` is a vector or an inflow spec. Two failures, one problem —
        showing both reads as two contradictory demands."""
        deck = base_deck()
        deck["solver"]["reference conditions"]["ic_1"]["v"] = "0 0 0"
        assert len(messages(deck)) == 1

    def test_no_message_quotes_a_pydantic_error_code(self):
        deck = base_deck()
        deck["solver"]["convergence control"]["cycles"] = "lots"
        deck["solver"]["reference conditions"]["ic_1"]["temperature"] = -5
        for message in messages(deck):
            for code in ("int_type", "model_type", "greater_than", "value_error"):
                assert code not in message

    @pytest.mark.parametrize(
        "error_type,msg,ctx,expected",
        [
            ("int_type", "", {}, "Must be a whole number."),
            ("float_type", "", {}, "Must be a number."),
            ("bool_type", "", {}, "Must be true or false."),
            ("greater_than", "", {"gt": 0}, "Must be greater than 0."),
            (
                "greater_than_equal",
                "",
                {"ge": 0},
                "Must be greater than or equal to 0.",
            ),
            ("less_than", "", {"lt": 1}, "Must be less than 1."),
            (
                "too_short",
                "",
                {"min_length": 3, "max_length": 3},
                "Must have exactly 3 values.",
            ),
            (
                "model_type",
                "Input should be a valid dictionary or instance of InflowVectorModel",
                {},
                "Must be a set of inflow vector settings.",
            ),
            (
                "value_error",
                "Value error, something is wrong",
                {},
                "Something is wrong",
            ),
        ],
    )
    def test_humanises_each_error_kind(self, error_type, msg, ctx, expected):
        assert _humanise_message(error_type, msg, ctx) == expected


class TestBoundErrorsNameTheField:
    """A bound error reads as a sentence about the field: its title, in
    sentence case with acronyms kept, and the bound in the field's units."""

    def test_wall_temperature_names_its_unit(self):
        deck = base_deck()
        deck["model"]["m1"]["boundary conditions"]["bc_1"]["temperature"] = 0.0
        assert messages(deck) == ["Wall temperature must be greater than 0 K."]

    def test_an_acronym_title_is_kept(self):
        deck = base_deck()
        deck["solver"]["convergence control"]["cfl"]["cfl"] = -1.0
        assert messages(deck) == ["CFL must be greater than 0."]

    def test_a_field_without_units_states_the_bare_bound(self):
        deck = base_deck()
        deck["solver"]["solver settings"][
            "mesh quality remediation angle threshold"
        ] = 200.0
        assert messages(deck) == [
            "Mesh quality remediation angle threshold must be less than or equal to 180."
        ]


class TestErrorPathsMatchTheUsersConfig:
    def test_a_union_tag_is_not_shown_as_a_level_of_config(self):
        """A wall's `loc` carries the tag `wall`; the user never wrote a
        key called `wall`, so it must not appear in the path or message."""
        deck = base_deck()
        deck["model"]["m1"]["boundary conditions"]["bc_1"]["nonsense"] = 1
        _model, nodes = validate_zcfd_comprehensive(deck)
        (node,) = [n for n in nodes if "nonsense" in n["field_path"]]
        assert node["field_path"] == "model.m1.boundary_conditions.bc_1.nonsense"
        assert "Wall" not in node["display_message"]

    def test_a_boundary_the_user_named_wall_survives(self):
        """A tag is told apart from a real key by the config, not by name:
        here the same word is both a BC the user named and the tag one
        level below it."""
        data = {"model": {"m1": {"boundary_conditions": {"wall": {"nonsense": 1}}}}}
        parts = _clean_loc_parts(
            ("model", "m1", "boundary_conditions", "wall", "wall", "nonsense"), data
        )
        assert parts == ["model", "m1", "boundary_conditions", "wall", "nonsense"]

    def test_a_field_sharing_a_tags_name_survives(self):
        """`pressure` is both an inflow kind and a reference-condition
        field; stripping tags by name alone silently ate the field."""
        data = {"solver": {"reference_conditions": {"IC1": {"pressure": -5}}}}
        parts = _clean_loc_parts(
            ("solver", "reference_conditions", "IC1", "pressure"), data
        )
        assert parts == ["solver", "reference_conditions", "IC1", "pressure"]

    def test_a_scalar_union_members_tag_is_dropped(self):
        """A number-or-block union discriminated by a callable puts the
        member's tag last, under a value that is a plain number."""
        data = {"bc": {"temperature": -5.0}}
        assert _clean_loc_parts(("bc", "temperature", "number"), data) == [
            "bc",
            "temperature",
        ]

    def test_a_field_under_a_validated_block_is_kept(self):
        """The whole-deck rules run on validated blocks, so the config can
        hold a model instance where the user wrote a dict."""
        deck = base_deck()
        deck["solver"]["fluid properties"]["gravity"] = [0.0, 0.0, -9.81]
        (node,) = error_nodes(deck)
        assert node["field_path"] == "solver.fluid_properties.gravity"
        assert node["anchor"] == "reference/solver/fluid-properties#gravity"

    def test_a_missing_key_is_kept(self):
        data = {"bc": {"type": "wall"}}
        assert _clean_loc_parts(("bc", "zones"), data) == ["bc", "zones"]

    def test_a_nested_field_is_named_with_its_parent(self):
        """`velocity.type` is not the boundary's own `type`; naming it
        bare put the error on the wrong widget."""
        deck = base_deck()
        deck["model"]["m1"]["boundary conditions"]["bc_1"]["velocity"] = {
            "vector": [1.0, 0.0, 0.0]
        }
        _model, nodes = validate_zcfd_comprehensive(deck)
        assert any(
            "Velocity" in n["display_message"] and "Type" in n["display_message"]
            for n in nodes
        ), [n["display_message"] for n in nodes]


class TestMergeKeepsUnrelatedErrorsApart:
    def test_two_different_fields_stay_two_nodes(self):
        deck = base_deck()
        deck["solver"]["reference conditions"]["ic_1"]["temperature"] = -5
        deck["solver"]["reference conditions"]["ic_1"]["pressure"] = -5
        assert len(messages(deck)) == 2

    def test_merge_is_a_no_op_on_an_empty_list(self):
        assert map_errors_to_nodes([]) == []


def test_deck_fixture_is_actually_valid():
    """Guards every negative test above: they only mean anything if the
    only error is the one they introduce."""
    model, nodes = validate_zcfd_comprehensive(copy.deepcopy(base_deck()))
    assert nodes == []
    assert model is not None


class TestANewReferenceConditionOnlyAsksForWhatItCannotGuess:
    """A fresh IC in the UI should not open covered in required-field
    errors. Only the freestream thermodynamic state has no sensible
    default; the turbulence quantities do, and had them in the
    pre-pydantic validator (base_ic_schema).
    """

    def test_temperature_and_pressure_are_the_only_required_fields(self):
        from zutil.validate.schemas.reference_conditions_schemas import (
            EulerReferenceConditions,
            RANSReferenceConditions,
            ViscousReferenceConditions,
        )

        for cls in (
            EulerReferenceConditions,
            ViscousReferenceConditions,
            RANSReferenceConditions,
        ):
            required = {n for n, f in cls.model_fields.items() if f.is_required()}
            assert required == {"temperature", "pressure"}, cls.__name__

    def test_turbulence_defaults_match_the_legacy_validator(self):
        deck = base_deck()
        ic = deck["solver"]["reference conditions"]["ic_1"]
        del ic["turbulence intensity"]
        del ic["eddy viscosity ratio"]
        model, nodes = validate_zcfd_comprehensive(deck)
        assert nodes == []
        applied = model.solver.reference_conditions["ic_1"]
        assert applied.turbulence_intensity == 0.01
        assert applied.eddy_viscosity_ratio == 0.1


class TestReferenceConditionPanelLayoutComesFromTheSchema:
    """Which settings are advanced, and what they group under, is the
    schema's call — the form only renders what it is told.
    """

    def test_turbulence_quantities_group_together(self):
        from zutil.validate.schemas.reference_conditions_schemas import (
            RANSReferenceConditions,
        )

        schema = RANSReferenceConditions.model_json_schema()
        for name in (
            "turbulence_intensity",
            "eddy_viscosity_ratio",
            "ambient_turbulence_intensity",
            "ambient_eddy_viscosity_ratio",
        ):
            assert schema["properties"][name]["zcfd"]["group"] == "Turbulence", name

    def test_ratios_group_together(self):
        from zutil.validate.schemas.reference_conditions_schemas import (
            EulerReferenceConditions,
        )

        schema = EulerReferenceConditions.model_json_schema()
        for name in (
            "total_pressure_ratio",
            "total_temperature_ratio",
            "static_pressure_ratio",
        ):
            assert schema["properties"][name]["zcfd"]["group"] == "Ratios", name

    def test_the_fields_every_ic_needs_stay_ungrouped(self):
        from zutil.validate.schemas.reference_conditions_schemas import (
            RANSReferenceConditions,
        )

        schema = RANSReferenceConditions.model_json_schema()
        for name in ("temperature", "pressure", "v", "reference_length"):
            assert "group" not in schema["properties"][name].get("zcfd", {}), name


class TestAModelLevelErrorPointsAtSomethingOnScreen:
    """`model.<name>.mesh` has no tree node of its own. Highlighting a
    "mesh" node that does not exist highlighted nothing, which is what made
    the error unactionable — point at the model instead.
    """

    def test_missing_mesh_highlights_the_model(self):
        deck = base_deck()
        del deck["model"]["m1"]["mesh"]
        _model, nodes = validate_zcfd_comprehensive(deck)
        (node,) = [n for n in nodes if n["field_path"].endswith("mesh")]
        assert node["nodes"] == ["m1"]
        assert node["field_name"] == "mesh"
        assert node["display_message"] == 'Missing required field "Mesh"'

    def test_a_section_level_error_still_names_its_section(self):
        deck = base_deck()
        deck["model"]["m1"]["boundary conditions"]["bc_1"]["nonsense"] = 1
        _model, nodes = validate_zcfd_comprehensive(deck)
        (node,) = [n for n in nodes if "nonsense" in n["field_path"]]
        assert node["nodes"] == ["boundary conditions", "bc_1"]


def error_nodes(deck):
    _model, nodes = validate_zcfd_comprehensive(deck)
    return [n for n in nodes if n.get("severity", "error") == "error"]


def rotating_wall():
    deck = base_deck()
    deck["model"]["m1"]["boundary conditions"]["bc_1"]["velocity"] = {
        "type": "rotating",
        "omega": "fast",
        "axis": [0, 0, 1],
        "origin": [0, 0, 0],
    }
    return deck


def unfinished_motion():
    deck = base_deck()
    deck["model"]["m1"]["transforms"] = {"motion": {"type": "rotating", "omega": 1.0}}
    return deck


def turbulence_typo():
    deck = base_deck()
    deck["solver"]["equations"]["turbulence"]["model"] = "k-omega"
    return deck


def unfinished_fluid_zone():
    deck = base_deck()
    deck["model"]["m1"]["fluid zones"] = {"fz_1": {"type": "rbf transform"}}
    return deck


class TestEveryErrorLinksToItsReferenceEntry:
    """Each node carries the anchor of the entry that documents its field;
    the boundary-condition type comes from the deck, since the error's own
    path does not name it."""

    def test_wall_temperature(self):
        deck = base_deck()
        deck["model"]["m1"]["boundary conditions"]["bc_1"]["temperature"] = -5.0
        (node,) = error_nodes(deck)
        assert node["nodes"] == ["boundary conditions", "bc_1"]
        assert node["field_path"] == "model.m1.boundary_conditions.bc_1.temperature"
        assert node["display_message"] == "Wall temperature must be greater than 0 K."
        assert node["anchor"] == "reference/model/boundary-conditions/wall/temperature"

    def test_nested_wall_field_carries_its_parent_key(self):
        (node,) = error_nodes(rotating_wall())
        assert node["field_path"] == "model.m1.boundary_conditions.bc_1.velocity.omega"
        assert (
            node["anchor"] == "reference/model/boundary-conditions/wall/velocity#omega"
        )

    def test_solver_field(self):
        (node,) = error_nodes(turbulence_typo())
        assert node["anchor"] == "reference/solver/equations/turbulence#model"

    def test_per_model_override_folds_to_its_solver_entry(self):
        deck = base_deck()
        deck["model"]["m1"]["convergence control"] = {
            "cycles": 10,
            "cfl": {"cfl": 2.0, "cfl turbulence": -1.0},
        }
        (node,) = [
            n for n in error_nodes(deck) if n["field_path"].endswith("cfl_turbulence")
        ]
        assert (
            node["anchor"] == "reference/solver/convergence-control/cfl#cfl-turbulence"
        )

    def test_hand_built_nodes_carry_an_anchor_too(self):
        deck = base_deck()
        deck["solver"]["initialisation"]["initial conditions"] = "ic_missing"
        nodes = error_nodes(deck)
        assert nodes and all("anchor" in n for n in nodes)


class TestFluidZoneNamesStayOutOfTheMessage:
    def test_the_zone_name_is_a_node_not_part_of_the_field(self):
        nodes = error_nodes(unfinished_fluid_zone())
        zones = next(n for n in nodes if n["field_path"].endswith(".zones"))
        assert zones["nodes"] == ["fluid zones", "fz_1"]
        assert zones["display_message"] == 'Missing required field "Zones"'
        assert zones["anchor"] == "reference/model/fluid-zones/rbf-transform#zones"
        assert not any("Fz 1" in n["display_message"] for n in nodes)


class TestMotionErrorsLinkToTheTransformsPage:
    def test_a_missing_motion_key_links_to_its_transforms_entry(self):
        nodes = error_nodes(unfinished_motion())
        axis = next(n for n in nodes if n["field_path"].endswith(".axis"))
        assert axis["field_path"] == "model.m1.transforms.motion.axis"
        assert axis["nodes"] == ["transforms"]
        assert axis["display_message"] == 'Missing required field "Motion → Axis"'
        assert axis["anchor"] == "reference/model/transforms/motion#axis"


ZONES_TEXT = (
    "'zones' is not a setting here: 'zones' names boundary faces. A fluid zone "
    "selects its cells with 'cell zones', the mesh cell zone numbers, or "
    "'definition', a closed-surface VTP file. A rotating region of the mesh is "
    'a \'rotating\' fluid zone: "fluid zones": {"<name>": {"type": "rotating", '
    '"cell zones": [...], "axis": [...], "origin": [...], "omega": ...}}.'
)


class TestRetiredSettingsNameTheirReplacement:
    def test_a_wall_rotation_centre_is_origin(self):
        deck = rotating_wall()
        velocity = deck["model"]["m1"]["boundary conditions"]["bc_1"]["velocity"]
        velocity["omega"] = 1.0
        velocity["center"] = velocity.pop("origin")
        nodes = error_nodes(deck)
        center = next(n for n in nodes if n["field_path"].endswith(".center"))
        assert center["field_path"] == (
            "model.m1.boundary_conditions.bc_1.velocity.center"
        )
        assert center["display_message"] == (
            "'center' is not a setting here: the centre of rotation is 'origin', "
            "as on every rotation."
        )

    @pytest.mark.parametrize(
        "settings",
        [
            {"kind": "pressure"},
            {"kind": "massflow", "mass flow rate": 1.0},
            {"kind": "radial pressure gradient", "reference radius": 1.0},
        ],
    )
    def test_an_outflow_has_no_total_temperature_ratio(self, settings):
        deck = base_deck()
        deck["model"]["m1"]["boundary conditions"]["bc_out"] = {
            "zones": [2],
            "type": "outflow",
            "reference": "ic_1",
            "total temperature ratio": 1.0,
            **settings,
        }
        (node,) = error_nodes(deck)
        assert node["field_path"] == (
            "model.m1.boundary_conditions.bc_out.total_temperature_ratio"
        )
        assert node["nodes"] == ["boundary conditions", "bc_out"]
        assert node["display_message"] == (
            "Outflow 'bc_out' sets 'total temperature ratio', which is not a "
            "setting of an outflow: the solver sets no temperature at an outlet. "
            "Total temperature is set on an inflow."
        )

    @pytest.mark.parametrize("kind", ["momentum", "porous", "canopy", "abl"])
    def test_zones_on_a_fluid_zone_names_cell_zones(self, kind):
        deck = base_deck()
        deck["model"]["m1"]["fluid zones"] = {"fz_1": {"type": kind, "zones": [0]}}
        (node,) = error_nodes(deck)
        assert node["field_path"] == "model.m1.fluid_zones.fz_1.zones"
        assert node["display_message"] == ZONES_TEXT

    def test_zones_under_transforms_names_faces_only(self):
        deck = base_deck()
        deck["model"]["m1"]["transforms"] = {"zones": [0]}
        (node,) = error_nodes(deck)
        assert node["field_path"] == "model.m1.transforms.zones"
        assert node["display_message"] == ZONES_TEXT

    def test_a_disc_without_definition_says_how_it_selects_cells(self):
        deck = base_deck()
        disc = {
            "type": "disc",
            "name": "d1",
            "rotation direction": "clockwise",
            "controller": {"type": "fixed", "omega": 1.0},
            "geometry": {
                "centre": [0.0, 0.0, 0.0],
                "normal": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "inner radius": 0.1,
                "outer radius": 1.0,
            },
            "discretisation": {"type": "disc", "number of elements": 10},
            "model": {"type": "simple", "power model": "fixed"},
        }
        deck["model"]["m1"]["fluid zones"] = {"fz_1": disc}
        (node,) = error_nodes(deck)
        assert node["field_path"] == "model.m1.fluid_zones.fz_1.definition"
        assert "closed-surface VTP" in node["display_message"]
        disc["definition"] = "disc.vtp"
        assert error_nodes(deck) == []

    def test_a_retired_zone_type_is_reported_on_its_type(self):
        deck = base_deck()
        deck["model"]["m1"]["fluid zones"] = {"FZ_1": {"type": "translating"}}
        (node,) = error_nodes(deck)
        assert node["field_path"] == "model.m1.fluid_zones.FZ_1.type"
        assert node["nodes"] == ["fluid zones", "FZ_1"]
        assert node["display_message"].startswith(
            "Fluid zone 'FZ_1' has type 'translating', which is not a fluid-zone type."
        )


class TestTerminalText:
    """The terminal names the field as a breadcrumb in the deck's own
    spelling, keeps instance names, and prints a link to the reference."""

    @pytest.fixture(autouse=True)
    def release(self, monkeypatch):
        monkeypatch.setattr(anchors, "zcfd_version", lambda: "2026.08.14860")

    def test_turbulence_model(self):
        deck = turbulence_typo()
        assert format_errors_as_text(error_nodes(deck), deck) == (
            "Configuration validation failed with 1 error:\n"
            "\n"
            "  solver › equations › turbulence › model\n"
            "    Must be one of 'sst', 'sas', 'sa-neg', 'sst-transition' or "
            "'spalart-allmaras'.\n"
            "    Reference: https://docs.zenotech.com/v2026.08.14860/"
            "reference/solver/equations/turbulence#model"
        )

    def test_instance_names_kept_in_the_breadcrumb(self):
        deck = rotating_wall()
        text = format_errors_as_text(error_nodes(deck), deck)
        assert "  model › m1 › boundary conditions › bc_1 › velocity › omega\n" in text
        assert text.endswith("reference/model/boundary-conditions/wall/velocity#omega")

    def test_an_applicability_error_names_the_key_in_deck_spelling(self):
        deck = base_deck()
        deck["model"]["m1"]["boundary conditions"]["bc_1"]["heat flux"] = 5.0e3
        assert format_errors_as_text(error_nodes(deck), deck) == (
            "Configuration validation failed with 1 error:\n"
            "\n"
            "  model › m1 › boundary conditions › bc_1 › heat flux\n"
            "    'heat flux' only applies when kind is 'wall function'\n"
            "    Reference: https://docs.zenotech.com/v2026.08.14860/"
            "reference/model/boundary-conditions/wall/heat-flux"
        )

    def test_several_errors_are_counted_in_the_plural(self):
        deck = unfinished_motion()
        text = format_errors_as_text(error_nodes(deck), deck)
        assert text.startswith("Configuration validation failed with 2 errors:\n")

    def test_a_pathless_error_is_its_message_alone(self):
        node = {
            "nodes": ["configuration"],
            "display_message": "Boom.",
            "field_path": "",
        }
        assert format_errors_as_text([node]) == (
            "Configuration validation failed with 1 error:\n\n  Boom."
        )

    def test_a_non_release_build_links_to_the_root(self, monkeypatch):
        monkeypatch.setattr(
            anchors, "zcfd_version", lambda: "2026.08.14860-39-g05b2c2a"
        )
        deck = turbulence_typo()
        assert format_errors_as_text(error_nodes(deck), deck).endswith(
            "Reference: https://docs.zenotech.com/"
            "reference/solver/equations/turbulence#model"
        )


class TestAMisspeltKeyIsOneError:
    """A misspelt key is one mistake. It must be reported once, under the
    key the engineer typed, with the key they probably meant."""

    def test_a_typo_suggests_the_close_key(self):
        deck = base_deck()
        deck["solver"]["convergence control"]["cyclse"] = 1
        assert messages(deck) == ['Unknown key "cyclse". Did you mean "cycles"?']

    def test_a_renamed_required_key_is_paired_with_the_missing_one(self):
        deck = rotating_wall()
        velocity = deck["model"]["m1"]["boundary conditions"]["bc_1"]["velocity"]
        velocity["omega"] = 10.0
        velocity["orgin"] = velocity.pop("origin")
        assert messages(deck) == ['Unknown key "orgin". Did you mean "origin"?']

    def test_an_unknown_key_with_nothing_close_has_no_suggestion(self):
        deck = base_deck()
        deck["solver"]["convergence control"]["zzzz"] = 1
        assert messages(deck) == ['Unknown key "zzzz".']

    def test_the_text_is_headed_by_the_deck_file(self):
        deck = base_deck()
        deck["solver"]["convergence control"]["cyclse"] = 1
        text = format_errors_as_text(error_nodes(deck), deck, "/runs/wing.py")
        assert text.startswith("wing.py: 1 error\n\n")


class TestNodesAreKeyedByFieldPath:
    """``nodes`` is deprecated, so one problem is one node whatever it says."""

    @pytest.fixture(autouse=True)
    def unstable_tree_names(self, monkeypatch):
        from zutil.validate import error_mapping

        placement = error_mapping._placement
        calls = iter(range(1000))
        monkeypatch.setattr(
            error_mapping,
            "_placement",
            lambda parts: ([f"tree {next(calls)}"], placement(parts)[1]),
        )

    def test_a_repeated_error_is_reported_once(self):
        error = {
            "type": "value_error",
            "loc": ("solver", "equations", "type"),
            "msg": "Value error, bad",
        }
        assert len(map_errors_to_nodes([error, dict(error)])) == 1

    def test_union_branch_failures_merge(self):
        loc = ("solver", "equations", "type")
        errors = [
            {"type": "value_error", "loc": loc, "msg": "Value error, one"},
            {"type": "value_error", "loc": loc, "msg": "Value error, two"},
        ]
        assert len(map_errors_to_nodes(errors)) == 1
