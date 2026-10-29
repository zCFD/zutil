"""
Tests for the ConditionalOptional / When / AppliesWhen applicability
machinery (schemas/base_types.py, applicability.py) and its wiring into
ZCFDInputModelBase (zcfd_schema.py).

A setting annotated ``ConditionalOptional[X, When({...})]`` only exists in
some configurations: it is optional by construction, the ``When`` clause is
exported into the JSON schema (``zcfd.applies_when``) so the UI can hide it,
and the same clause is evaluated by ``find_applicability_violations`` to
reject a config that sets the field while its clause is false.

There are currently zero ConditionalOptional fields anywhere in the real
schemas/ package, so all behavioural tests here build small local toy models
rather than depending on any real schema field. Section 5 below is the
exception: it walks the *real* model tree rooted at ZCFDInputModelBase to
make sure the path-resolution walker itself is correct, ready for the day a
real ConditionalOptional field lands.
"""

import typing
from typing import Annotated, Dict, List, Literal, Optional, Union

import pytest
from pydantic import BaseModel, ConfigDict

from zutil.validate.applicability import find_applicability_violations
from zutil.validate.schemas.base_types import (
    AnyOf,
    AppliesWhen,
    ConditionalChoice,
    ConditionalDefault,
    ConditionalOptional,
    ConditionalRequired,
    Disjoint,
    Exclusive,
    NONZERO,
    NotIn,
    RequiredWhen,
    SET,
    UNSET,
    When,
    zCFD_Field,
)
from zutil.validate.tests.test_core_orchestration import _minimal_valid_config
from zutil.validate.zcfd_schema import ZCFDInputModelBase


# ---------------------------------------------------------------------------
# 1. When
# ---------------------------------------------------------------------------


class TestWhenNormalisation:
    def test_single_value_normalised_to_one_tuple(self):
        w = When({"a.b": "x"})
        assert w.conditions == {"a.b": ("x",)}

    def test_list_value_normalised_to_tuple(self):
        w = When({"a.b": ["x", "y"]})
        assert w.conditions == {"a.b": ("x", "y")}

    def test_set_value_normalised_to_tuple(self):
        w = When({"a.b": {"x"}})
        assert w.conditions == {"a.b": ("x",)}

    def test_multiple_conditions_each_normalised(self):
        w = When({"a.b": "x", "c.d": ["y", "z"]})
        assert w.conditions == {"a.b": ("x",), "c.d": ("y", "z")}

    def test_to_schema_round_trip(self):
        w = When({"a.b": ["x", "y"], "c.d": "z"})
        schema = w.to_schema()
        assert schema == {"a.b": ["x", "y"], "c.d": ["z"]}
        # Every value in the schema form is a plain list (JSON-serialisable).
        assert all(isinstance(v, list) for v in schema.values())

        w2 = When(schema)
        assert w2.conditions == w.conditions

    def test_evaluate_true_when_value_matches(self):
        w = When({"a.b": "x"})
        assert w.evaluate({"a": {"b": "x"}}) is True

    def test_evaluate_false_when_value_does_not_match(self):
        w = When({"a.b": "x"})
        assert w.evaluate({"a": {"b": "y"}}) is False

    def test_evaluate_true_for_one_of_multiple_allowed_values(self):
        w = When({"a.b": ["x", "y"]})
        assert w.evaluate({"a": {"b": "y"}}) is True

    def test_evaluate_requires_all_conditions_to_hold(self):
        w = When({"a.b": "x", "c.d": "y"})
        assert w.evaluate({"a": {"b": "x"}, "c": {"d": "y"}}) is True
        assert w.evaluate({"a": {"b": "x"}, "c": {"d": "NOPE"}}) is False

    def test_evaluate_missing_path_returns_false(self):
        w = When({"a.b.c": "x"})
        assert w.evaluate({}) is False
        assert w.evaluate({"a": {}}) is False
        assert w.evaluate({"a": {"b": {}}}) is False

    def test_evaluate_missing_path_returns_false_when_node_not_a_mapping(self):
        """A path that dead-ends into a non-mapping value (e.g. a scalar
        sibling shares a name-prefix) must not raise — the setting simply
        does not apply.
        """
        w = When({"a.b": "x"})
        assert w.evaluate({"a": "not-a-dict"}) is False

    def test_describe_wording_single_condition_single_value(self):
        w = When({"a.b": "x"})
        assert w.describe("field1") == "'field1' only applies when a.b is 'x'"

    def test_describe_wording_single_condition_multiple_values(self):
        w = When({"a.b": ["x", "y"]})
        assert (
            w.describe("field1") == "'field1' only applies when a.b is one of 'x', 'y'"
        )

    def test_describe_wording_multiple_conditions_joined_with_and(self):
        w = When({"a.b": "x", "c.d": "y"})
        assert (
            w.describe("field2")
            == "'field2' only applies when a.b is 'x' and c.d is 'y'"
        )

    def test_empty_conditions_raises_value_error(self):
        with pytest.raises(ValueError):
            When({})

    def test_empty_allowed_values_raises_value_error(self):
        with pytest.raises(ValueError):
            When({"a.b": []})

    def test_non_string_path_raises_value_error(self):
        with pytest.raises(ValueError):
            When({1: "x"})

    def test_empty_string_path_raises_value_error(self):
        with pytest.raises(ValueError):
            When({"": "x"})


class TestWhenSetSentinel:
    """Direct unit coverage for the SET sentinel (present-with-any-non-null-
    value), and its combination with UNSET (present-or-absent), on
    ``When.evaluate`` — this machinery is exercised end-to-end via
    ConditionalOptional but had no direct evaluate() unit cases.
    """

    def test_present_non_null_value_matches(self):
        w = When({"a.b": SET})
        assert w.evaluate({"a": {"b": "x"}}) is True

    def test_present_none_value_does_not_match(self):
        w = When({"a.b": SET})
        assert w.evaluate({"a": {"b": None}}) is False

    def test_absent_path_does_not_match(self):
        w = When({"a.b": SET})
        assert w.evaluate({"a": {}}) is False

    def test_combined_unset_and_set_matches_absent_path(self):
        w = When({"a.b": (UNSET, SET)})
        assert w.evaluate({"a": {}}) is True

    def test_combined_unset_and_set_matches_present_non_null_value(self):
        w = When({"a.b": (UNSET, SET)})
        assert w.evaluate({"a": {"b": "x"}}) is True

    def test_combined_unset_and_set_still_rejects_explicit_none(self):
        """UNSET means the path is absent, not "present but null" — an
        explicit null value must not be treated as unset."""
        w = When({"a.b": (UNSET, SET)})
        assert w.evaluate({"a": {"b": None}}) is False


class TestWhenWildcard:
    """Direct unit coverage for the '*' existential wildcard segment."""

    def test_existential_match_one_branch_suffices(self):
        w = When({"a.*.b": "x"})
        root = {"a": {"m1": {"b": "y"}, "m2": {"b": "x"}}}
        assert w.evaluate(root) is True

    def test_no_matching_branch_fails(self):
        w = When({"a.*.b": "x"})
        root = {"a": {"m1": {"b": "y"}, "m2": {"b": "z"}}}
        assert w.evaluate(root) is False

    def test_non_dict_node_at_wildcard_level_treated_as_absent(self):
        w = When({"a.*.b": "x"})
        assert w.evaluate({"a": "not-a-dict"}) is False

    def test_non_dict_node_at_wildcard_level_matches_when_unset_allowed(self):
        w = When({"a.*.b": (UNSET, "x")})
        assert w.evaluate({"a": "not-a-dict"}) is True

    def test_empty_dict_at_wildcard_level_fails(self):
        w = When({"a.*.b": "x"})
        assert w.evaluate({"a": {}}) is False

    def test_empty_dict_at_wildcard_level_fails_even_if_unset_allowed(self):
        """An empty dict is existentially quantified over zero branches, so
        it fails regardless of UNSET — it is NOT the same as a missing/
        non-dict node, which is treated as absent."""
        w = When({"a.*.b": (UNSET, "x")})
        assert w.evaluate({"a": {}}) is False


class TestWhenMeshAnchor:
    """Direct unit coverage for the '<mesh>' anchor, resolved via the
    ``mesh_anchor`` kwarg threaded through from applicability.py's walker.
    """

    def test_mesh_anchor_resolves_against_the_given_subtree(self):
        w = When({"<mesh>.x": "y"})
        root = {"model": {"plate": {"x": "y"}, "wing": {"x": "z"}}}
        assert w.evaluate(root, mesh_anchor="model.plate") is True

    def test_mesh_anchor_does_not_leak_into_other_mesh(self):
        w = When({"<mesh>.x": "y"})
        root = {"model": {"plate": {"x": "y"}, "wing": {"x": "z"}}}
        assert w.evaluate(root, mesh_anchor="model.wing") is False

    def test_no_mesh_anchor_is_always_false(self):
        w = When({"<mesh>.x": "y"})
        root = {"model": {"plate": {"x": "y"}}}
        assert w.evaluate(root, mesh_anchor=None) is False
        assert w.evaluate(root) is False

    def test_mesh_anchor_combined_with_wildcard(self):
        w = When({"<mesh>.boundary_conditions.*.type": "overset"})
        root = {
            "model": {
                "plate": {
                    "boundary_conditions": {
                        "inlet": {"type": "wall"},
                        "far": {"type": "overset"},
                    }
                }
            }
        }
        assert w.evaluate(root, mesh_anchor="model.plate") is True

    def test_mesh_anchor_combined_with_wildcard_no_match(self):
        w = When({"<mesh>.boundary_conditions.*.type": "overset"})
        root = {
            "model": {
                "plate": {
                    "boundary_conditions": {
                        "inlet": {"type": "wall"},
                        "far": {"type": "wall"},
                    }
                }
            }
        }
        assert w.evaluate(root, mesh_anchor="model.plate") is False


class TestWhenDescribeAndSchemaForSentinels:
    def test_describe_renders_set_as_set(self):
        w = When({"a.b": SET})
        assert w.describe("field1") == "'field1' only applies when a.b is set"

    def test_describe_renders_unset_as_unset(self):
        w = When({"a.b": UNSET})
        assert w.describe("field1") == "'field1' only applies when a.b is unset"

    def test_to_schema_encodes_unset_as_none_and_set_as_present_marker(self):
        w = When({"a.b": (UNSET, SET)})
        assert w.to_schema() == {"a.b": [None, "__present__"]}


class TestWhenNotIn:
    """NotIn states "anything but these": the complement of an allowed set,
    so a rule that excludes one value need not enumerate every other one."""

    def test_other_value_matches(self):
        assert When({"a.b": NotIn("x")}).evaluate({"a": {"b": "y"}}) is True

    def test_excluded_value_does_not_match(self):
        assert When({"a.b": NotIn("x")}).evaluate({"a": {"b": "x"}}) is False

    def test_absent_path_matches(self):
        assert When({"a.b": NotIn("x")}).evaluate({}) is True

    def test_any_of_several_excluded_values_does_not_match(self):
        w = When({"a.b": NotIn("x", "y")})
        assert w.evaluate({"a": {"b": "y"}}) is False
        assert w.evaluate({"a": {"b": "z"}}) is True

    def test_empty_not_in_raises(self):
        with pytest.raises(ValueError):
            NotIn()

    def test_describe_single_and_multiple(self):
        assert (
            When({"a.b": NotIn("x")}).describe("f")
            == "'f' only applies when a.b is not 'x'"
        )
        assert (
            When({"a.b": NotIn("x", "y")}).describe("f")
            == "'f' only applies when a.b is none of 'x', 'y'"
        )

    def test_to_schema_round_trip(self):
        w = When({"a.b": NotIn("x", "y"), "c": ["z"]})
        schema = w.to_schema()
        assert schema == {"a.b": {"not": ["x", "y"]}, "c": ["z"]}
        assert When(schema).conditions == w.conditions


class TestWhenNonZero:
    """NONZERO holds for a present value that is neither zero nor a zero
    vector, so "gravity: [0, 0, 0]" counts as no gravity."""

    @pytest.mark.parametrize("value", [9.81, -0.5, [0.0, 0.0, -9.81], [1, 0, 0]])
    def test_non_zero_value_matches(self, value):
        assert When({"a.b": NONZERO}).evaluate({"a": {"b": value}}) is True

    @pytest.mark.parametrize("value", [0, 0.0, -0.0, [0.0, 0.0, 0.0], [0, 0, 0]])
    def test_zero_value_does_not_match(self, value):
        assert When({"a.b": NONZERO}).evaluate({"a": {"b": value}}) is False

    def test_absent_path_does_not_match(self):
        assert When({"a.b": NONZERO}).evaluate({}) is False

    def test_describe(self):
        assert (
            When({"<self>.g": NONZERO}).describe("f")
            == "'f' only applies when g is set and non-zero"
        )

    def test_to_schema(self):
        assert When({"a.b": NONZERO}).to_schema() == {
            "a.b": {"not": [None, 0.0, [0.0, 0.0, 0.0]]}
        }

    def test_describe_unset_with_other_excluded_values(self):
        assert (
            When({"a.b": NotIn(UNSET, "x")}).describe("f")
            == "'f' only applies when a.b is set and not 'x'"
        )
        assert (
            When({"a.b": NotIn(UNSET)}).describe("f")
            == "'f' only applies when a.b is set"
        )


class TestWhenSelfAnchor:
    """A <self> path is resolved against the dump of the model that owns the
    annotated field, so a sibling rule holds wherever that model sits."""

    def test_resolves_against_self_node(self):
        w = When({"<self>.flag": (True,)})
        assert w.evaluate({}, self_node={"flag": True}) is True
        assert w.evaluate({}, self_node={"flag": False}) is False

    def test_ignores_root_with_the_same_key(self):
        w = When({"<self>.flag": (True,)})
        assert w.evaluate({"flag": True}, self_node={}) is False

    def test_no_self_node_is_always_false(self):
        assert When({"<self>.flag": (True,)}).evaluate({"flag": True}) is False

    def test_uses_self_anchor(self):
        assert When({"<self>.flag": True}).uses_self_anchor is True
        assert When({"a.flag": True}).uses_self_anchor is False

    def test_describe_names_the_sibling_without_the_anchor(self):
        w = When({"<self>.flag": (True,)})
        assert w.describe("f") == "'f' only applies when flag is True"


class TestWhenDescribeForGatedValue:
    def test_value_is_named_after_the_field(self):
        w = When({"a.b": "x"})
        assert (
            w.describe("mode", value="fast")
            == "'mode': 'fast' only applies when a.b is 'x'"
        )


# ---------------------------------------------------------------------------
# 2. ConditionalOptional
# ---------------------------------------------------------------------------


class TestConditionalOptionalExpansion:
    def test_expands_to_annotated_optional_with_applies_when(self):
        result = ConditionalOptional[int, When({"a.b": 1})]

        assert typing.get_origin(result) is Annotated
        inner_type, marker = typing.get_args(result)
        assert inner_type == Optional[int]
        assert isinstance(marker, AppliesWhen)
        assert marker.when.conditions == {"a.b": (1,)}

    def test_accepts_raw_dict_auto_wrapped_into_when(self):
        result = ConditionalOptional[int, {"a.b": 1}]
        _, marker = typing.get_args(result)
        assert isinstance(marker, AppliesWhen)
        assert isinstance(marker.when, When)
        assert marker.when.conditions == {"a.b": (1,)}

    def test_wrong_arity_single_param_raises_type_error(self):
        with pytest.raises(TypeError):
            ConditionalOptional[int]

    def test_wrong_arity_three_params_raises_type_error(self):
        with pytest.raises(TypeError):
            ConditionalOptional[int, When({"a.b": 1}), int]

    def test_non_when_non_dict_second_param_raises_type_error(self):
        with pytest.raises(TypeError):
            ConditionalOptional[int, "not-a-when"]

    def test_model_with_conditional_optional_field_accepts_none_default(self):
        class Toy(BaseModel):
            model_config = ConfigDict(extra="forbid")

            mode: str = "off"
            setting: ConditionalOptional[float, When({"mode": "special"})] = None

        instance = Toy(mode="off")
        assert instance.setting is None

    def test_model_with_conditional_optional_field_accepts_concrete_value(self):
        class Toy(BaseModel):
            model_config = ConfigDict(extra="forbid")

            mode: str = "off"
            setting: ConditionalOptional[float, When({"mode": "special"})] = None

        instance = Toy(mode="special", setting=3.0)
        assert instance.setting == 3.0


class TestConditionalChoiceExpansion:
    def test_expands_to_annotated_type_with_one_marker_per_value(self):
        result = ConditionalChoice[
            Literal["a", "b", "c"], {"b": When({"x": 1}), "c": {"y": 2}}
        ]
        assert typing.get_origin(result) is Annotated
        inner, *markers = typing.get_args(result)
        assert inner == Literal["a", "b", "c"]
        assert [m.values for m in markers] == [("b",), ("c",)]
        assert markers[1].when.conditions == {"y": (2,)}

    def test_type_is_not_made_optional(self):
        inner, _ = typing.get_args(ConditionalChoice[Literal["a"], {"a": {"x": 1}}])
        assert inner == Literal["a"]

    def test_wrong_arity_raises_type_error(self):
        with pytest.raises(TypeError):
            ConditionalChoice[Literal["a"]]

    def test_empty_mapping_raises_type_error(self):
        with pytest.raises(TypeError):
            ConditionalChoice[Literal["a"], {}]

    def test_non_when_clause_raises_type_error(self):
        with pytest.raises(TypeError):
            ConditionalChoice[Literal["a"], {"a": "not-a-when"}]


class _ChoiceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ConditionalChoice[
        Literal["slow", "fast"], {"fast": When({"<self>.turbo": (True,)})}
    ] = "slow"
    turbo: bool = False


class _ChoiceDefaultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ConditionalChoice[Literal["only"], {"only": When({"gate": "open"})}] = "only"


class _ChoiceRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate: str = "shut"
    top: _ChoiceModel = _ChoiceModel()
    items: Dict[str, _ChoiceModel] = {}
    fixed: Optional[_ChoiceDefaultModel] = None


class TestConditionalChoiceViolations:
    def test_ungated_value_never_violates(self):
        assert find_applicability_violations(_ChoiceRoot()) == []

    def test_gated_value_with_sibling_clause_holding_is_accepted(self):
        root = _ChoiceRoot(top=_ChoiceModel(mode="fast", turbo=True))
        assert find_applicability_violations(root) == []

    def test_gated_value_with_sibling_clause_false_is_rejected(self):
        root = _ChoiceRoot(top=_ChoiceModel(mode="fast", turbo=False))
        assert find_applicability_violations(root) == [
            ("top.mode", "'mode': 'fast' only applies when turbo is True")
        ]

    def test_self_anchor_follows_each_instance(self):
        root = _ChoiceRoot(
            top=_ChoiceModel(mode="fast", turbo=True),
            items={
                "on": _ChoiceModel(mode="fast", turbo=True),
                "off": _ChoiceModel(mode="fast", turbo=False),
            },
        )
        assert [p for p, _ in find_applicability_violations(root)] == ["items.off.mode"]

    def test_gated_default_value_is_still_checked(self):
        root = _ChoiceRoot(fixed=_ChoiceDefaultModel())
        assert find_applicability_violations(root) == [
            ("fixed.name", "'name': 'only' only applies when gate is 'open'")
        ]
        assert (
            find_applicability_violations(
                _ChoiceRoot(gate="open", fixed=_ChoiceDefaultModel())
            )
            == []
        )


class _HintModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ConditionalChoice[
        Literal["slow", "fast"],
        {"fast": (When({"<self>.turbo": (True,)}), "use 'slow' instead")},
    ] = "slow"
    turbo: bool = False


class TestConditionalChoiceHint:
    def test_hint_is_appended_to_the_rejection(self):
        assert find_applicability_violations(_HintModel(mode="fast")) == [
            (
                "mode",
                "'mode': 'fast' only applies when turbo is True; use 'slow' instead",
            )
        ]

    def test_hint_does_not_change_the_schema(self):
        zcfd = _HintModel.model_json_schema()["properties"]["mode"]["zcfd"]
        assert zcfd == {"applies_when_value": {"fast": {"<self>.turbo": [True]}}}


# ---------------------------------------------------------------------------
# 3. JSON schema export
# ---------------------------------------------------------------------------


class TestJsonSchemaExport:
    def test_idle_motion_warning_is_in_the_form_schema(self):
        """A form shows the idle 'motion' warning without asking the
        validator: the clause and its text travel with the field, beside the
        applies_when clause that gates the field itself."""
        defs = ZCFDInputModelBase.model_json_schema()["$defs"]
        zcfd = defs["TransformsModel"]["properties"]["motion"]["zcfd"]
        assert zcfd["advises"]["when"] == {
            "any_of": [
                {
                    "<self>.motion.moving_mesh": [True],
                    "<self>.motion.omega": [0.0],
                },
                {
                    "<self>.motion.moving_mesh": [True],
                    "<self>.motion.velocity": [[0.0, 0.0, 0.0]],
                },
            ]
        }
        assert zcfd["advises"]["advice"].startswith("'motion' has a zero rate")
        assert "applies_when" not in zcfd

    def test_motion_rules_are_in_the_form_schema(self):
        """The form gates each motion setting with the clause validation uses."""
        defs = ZCFDInputModelBase.model_json_schema()["$defs"]
        rotating = defs["RotatingMotion"]["properties"]
        assert rotating["moving_mesh"]["zcfd"]["applies_when_value"] == {
            True: {
                "<mesh>.boundary_conditions.*.kind": ["overset", "sliding"],
                "solver.time_settings.type": ["unsteady"],
                "solver.solver_settings.type": [None, "compressible"],
            }
        }
        assert rotating["ramp_time"]["zcfd"]["applies_when"] == {
            "<self>.moving_mesh": [True]
        }
        assert rotating["rotate_nodes"]["zcfd"]["applies_when"] == {
            "<self>.moving_mesh": [True]
        }
        assert "rotate_nodes" not in defs["TranslatingMotion"]["properties"]

    def test_every_form_schema_clause_is_a_pure_condition_map(self):
        """A form evaluates each key of a clause as a path and each value as
        the allowed values, so a clause carries nothing else: its wording
        comes from the validator."""
        schema = ZCFDInputModelBase.model_json_schema()
        clauses = []

        def collect(node):
            if isinstance(node, dict):
                zcfd = node.get("zcfd")
                if isinstance(zcfd, dict):
                    if "applies_when" in zcfd:
                        clauses.append(zcfd["applies_when"])
                    if "required_when" in zcfd:
                        clauses.append(zcfd["required_when"])
                    clauses.extend((zcfd.get("applies_when_value") or {}).values())
                for child in node.values():
                    collect(child)
            elif isinstance(node, list):
                for child in node:
                    collect(child)

        def branches(clause):
            if set(clause) == {"any_of"}:
                return clause["any_of"]
            return [clause]

        collect(schema)
        assert clauses
        for clause in clauses:
            for branch in branches(clause):
                for path, allowed in branch.items():
                    assert "." in path or path.startswith("<"), path
                    assert isinstance(allowed, list) or (
                        isinstance(allowed, dict)
                        and set(allowed) == {"not"}
                        and isinstance(allowed["not"], list)
                    ), (path, allowed)

    def test_conditional_choice_field_carries_applies_when_value(self):
        zcfd = _ChoiceModel.model_json_schema()["properties"]["mode"]["zcfd"]
        assert zcfd == {"applies_when_value": {"fast": {"<self>.turbo": [True]}}}
        assert "applies_when" not in zcfd

    def test_conditional_optional_field_carries_applies_when(self):
        class Toy(BaseModel):
            model_config = ConfigDict(extra="forbid")

            mode: str = "off"
            setting: ConditionalOptional[float, When({"mode": "special"})] = None

        schema = Toy.model_json_schema()
        node = schema["properties"]["setting"]
        assert node["zcfd"]["applies_when"] == {"mode": ["special"]}

    def test_zcfd_field_extras_appear_under_zcfd_key(self):
        class Toy(BaseModel):
            model_config = ConfigDict(extra="forbid")

            plain: float = zCFD_Field(
                default=1.0,
                advanced=True,
                units="Pa",
                group="Numerics",
                placeholder="e.g. 3.0",
            )

        schema = Toy.model_json_schema()
        node = schema["properties"]["plain"]
        assert node["zcfd"] == {
            "advanced": True,
            "updatable": False,
            "units": "Pa",
            "group": "Numerics",
            "placeholder": "e.g. 3.0",
        }

    def test_conditional_optional_plus_zcfd_field_merge_not_clobber(self):
        """A field using BOTH ConditionalOptional (annotation-level
        applies_when) AND zCFD_Field(...) as its default (field-level
        advanced/updatable/units/... extras) must end up with both sets of
        extras merged under the single 'zcfd' key — neither side may
        clobber the other.
        """

        class Toy(BaseModel):
            model_config = ConfigDict(extra="forbid")

            mode: str = "off"
            combo: ConditionalOptional[float, When({"mode": "special"})] = zCFD_Field(
                default=None,
                advanced=True,
                units="Pa",
                group="Numerics",
                placeholder="e.g. 3.0",
            )

        schema = Toy.model_json_schema()
        node = schema["properties"]["combo"]
        zcfd = node["zcfd"]

        # From AppliesWhen (annotation-level).
        assert zcfd["applies_when"] == {"mode": ["special"]}
        # From zCFD_Field (field-level) — must survive the merge.
        assert zcfd["advanced"] is True
        assert zcfd["updatable"] is False
        assert zcfd["units"] == "Pa"
        assert zcfd["group"] == "Numerics"
        assert zcfd["placeholder"] == "e.g. 3.0"


# ---------------------------------------------------------------------------
# 4. find_applicability_violations
# ---------------------------------------------------------------------------


class _SubModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "off"
    # Root-anchored path: When clauses always reference the full dotted
    # path into the *root* model's dump, not a path relative to whichever
    # model the annotated field happens to live on.
    multigrid: ConditionalOptional[int, When({"sub.mode": "on"})] = None


class _RootModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "off"
    sub: _SubModel = _SubModel()
    items: Dict[str, _SubModel] = {}


class TestFindApplicabilityViolations:
    def test_no_violation_when_clause_holds(self):
        root = _RootModel(sub=_SubModel(mode="on", multigrid=5))
        assert find_applicability_violations(root) == []

    def test_no_violation_when_field_is_none(self):
        root = _RootModel(sub=_SubModel(mode="off", multigrid=None))
        assert find_applicability_violations(root) == []

    def test_violation_when_set_while_inapplicable(self):
        root = _RootModel(sub=_SubModel(mode="off", multigrid=5))
        violations = find_applicability_violations(root)
        assert len(violations) == 1
        path, message = violations[0]
        assert path == "sub.multigrid"
        assert message == When({"sub.mode": "on"}).describe("multigrid")

    def test_nested_model_path_is_dotted(self):
        root = _RootModel(sub=_SubModel(mode="off", multigrid=5))
        violations = find_applicability_violations(root)
        assert violations[0][0] == "sub.multigrid"

    def test_dict_of_models_path_includes_dict_key(self):
        root = _RootModel(items={"foo": _SubModel(mode="off", multigrid=5)})
        violations = find_applicability_violations(root)
        assert len(violations) == 1
        assert violations[0][0] == "items.foo.multigrid"

    def test_dict_of_models_no_violation_when_clause_holds(self):
        root = _RootModel(
            sub=_SubModel(mode="on"),
            items={"foo": _SubModel(mode="off", multigrid=5)},
        )
        # The clause references root path "sub.mode", which is "on" here,
        # so the items['foo'].multigrid setting is applicable everywhere.
        assert find_applicability_violations(root) == []


class _Part(BaseModel):
    ids: Annotated[List[int], Disjoint("id", "part")]


class _Mesh(BaseModel):
    parts: Dict[str, _Part] = {}


class _Case(BaseModel):
    model: Dict[str, _Mesh] = {}


def _case(**models) -> _Case:
    return _Case(
        model={
            name: _Mesh(parts={part: _Part(ids=ids) for part, ids in parts.items()})
            for name, parts in models.items()
        }
    )


class TestDisjoint:
    def test_separate_lists_pass(self):
        assert find_applicability_violations(_case(m={"a": [1], "b": [2, 3]})) == []

    def test_a_shared_entry_is_reported_on_the_later_holder(self):
        assert find_applicability_violations(_case(m={"a": [1, 2], "b": [3, 2]})) == [
            (
                "model.m.parts.b.ids",
                "Id 2 is in 'ids' here and on 'a'. Each id belongs to one part "
                "at most: remove it from one of them.",
            )
        ]

    def test_several_shared_entries_are_named_together(self):
        [(_, message)] = find_applicability_violations(
            _case(m={"a": [1, 2], "b": [2, 1]})
        )
        assert message.startswith("Ids 2, 1 are in 'ids' here and on 'a'.")

    def test_each_model_is_its_own(self):
        assert find_applicability_violations(_case(m={"a": [1]}, n={"a": [1]})) == []

    def test_the_rule_reaches_the_json_schema(self):
        schema = _Part.model_json_schema()
        assert schema["properties"]["ids"]["zcfd"]["disjoint"] is True


# ---------------------------------------------------------------------------
# 5. Path-existence guard for the REAL schema package
# ---------------------------------------------------------------------------
#
# There are zero ConditionalOptional fields in the real schemas today, so
# the walker below discovers an empty marker set and the test passes
# trivially. It is written to be exercised properly the day the first real
# ConditionalOptional field lands: if that field's When() condition path
# doesn't resolve to a real field chain from ZCFDInputModelBase (typo,
# renamed field, wrong nesting, ...) this test will start failing.


def _is_basemodel(tp) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _unwrap_targets(annotation):
    """Given a field's type annotation, return a list of
    (needs_wildcard_key, model_cls_or_None) pairs describing how to
    continue resolving a dotted path through this field.

    - Optional[X] / Union[...] / Annotated[...] are unwrapped transparently.
    - Dict[str, Model] fields report (True, Model): the next path segment
      is an arbitrary dict key, then continue into Model.
    - Dict[str, <scalar>] fields report (True, None): the next path
      segment is an arbitrary dict key, and it must be the final segment.
    - List[Model] elements are not addressable by dotted path (When()
      conditions are evaluated against Mapping nodes only — see
      When.evaluate), so lists contribute no targets.
    - A bare BaseModel subclass reports (False, ModelCls).
    - Anything else (plain scalars, Callable, custom validators, ...) is a
      leaf: no targets.
    """
    origin = typing.get_origin(annotation)

    if origin is Annotated:
        return _unwrap_targets(typing.get_args(annotation)[0])

    if origin is Union:
        targets = []
        for arg in typing.get_args(annotation):
            if arg is type(None):
                continue
            targets.extend(_unwrap_targets(arg))
        return targets

    if origin is dict:
        args = typing.get_args(annotation)
        if len(args) == 2:
            # Unwrap the value type recursively so Dict[str, Union[ModelA,
            # ModelB]] (e.g. fluid_zones) contributes its member models —
            # a bare _is_basemodel check here left dict-of-union markers
            # invisible to the path guard.
            inner = _unwrap_targets(args[1])
            if inner:
                return [(True, model) for _, model in inner if model is not None] or [
                    (True, None)
                ]
        return [(True, None)]

    if origin in (list, tuple, set, frozenset):
        # Not addressable by dotted path today.
        return []

    if _is_basemodel(annotation):
        return [(False, annotation)]

    return []


def _collect_model_classes(root_cls, visited=None):
    """All BaseModel subclasses reachable from root_cls via model_fields."""
    if visited is None:
        visited = set()
    if root_cls in visited:
        return visited
    visited.add(root_cls)
    for field_info in root_cls.model_fields.values():
        for _, model_cls in _unwrap_targets(field_info.annotation):
            if model_cls is not None:
                _collect_model_classes(model_cls, visited)
    return visited


def _collect_applies_when_markers(root_cls):
    """(owning_model_cls, field_name, When) for every AppliesWhen marker
    reachable from root_cls.
    """
    markers = []
    for model_cls in _collect_model_classes(root_cls):
        for name, field_info in model_cls.model_fields.items():
            for marker in field_info.metadata:
                if isinstance(marker, AppliesWhen):
                    markers.append((model_cls, name, marker.when))
    return markers


def _path_resolves(path: str, root_cls) -> bool:
    """True if the dotted path resolves to a real field chain starting
    from root_cls, per the unwrap rules in _unwrap_targets.

    Discriminated (and plain) unions are handled by trying every member:
    a path segment is valid if ANY union member resolves it.

    Path features from When(): a leading "<mesh>" anchor resolves as
    model.<any-key> (i.e. into ModelSettings); a "*" segment sits in a
    dict-key position and is consumed like any other wildcard key.
    """
    segments = path.split(".")
    if segments[0] == "<mesh>":
        segments = ["model", "*"] + segments[1:]
    return _segments_resolve(segments, {root_cls})


def _segments_resolve(segments, candidate_models) -> bool:
    if not segments:
        return True

    seg, rest = segments[0], segments[1:]
    matched = False

    for model_cls in candidate_models:
        if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
            continue
        field_info = model_cls.model_fields.get(seg)
        if field_info is None:
            continue

        targets = _unwrap_targets(field_info.annotation)
        if not targets:
            # Scalar leaf: only a match if this was the last segment.
            if not rest:
                matched = True
            continue

        for needs_wildcard, next_model in targets:
            if needs_wildcard:
                if not rest:
                    # Dict field referenced with no key — nothing more to
                    # resolve against.
                    continue
                key_consumed_rest = rest[1:]
                if next_model is None:
                    if not key_consumed_rest:
                        matched = True
                elif _segments_resolve(key_consumed_rest, {next_model}):
                    matched = True
            else:
                if _segments_resolve(rest, {next_model}):
                    matched = True

    return matched


class _WalkerLeafA(BaseModel):
    kind: Literal["a"]
    value: int


class _WalkerLeafB(BaseModel):
    kind: Literal["b"]
    count: int


class _WalkerWrapper(BaseModel):
    variant: Union[_WalkerLeafA, _WalkerLeafB]
    items: Dict[str, _WalkerLeafA] = {}
    name: str = "x"


class _WalkerRoot(BaseModel):
    top: _WalkerWrapper


class TestApplicabilityPathWalkerCorrectness:
    """Unit tests for the walker itself (section 5's helpers), using local
    synthetic models so the walker's logic is proven correct independently
    of whether any real ConditionalOptional fields exist yet.
    """

    def test_resolves_field_present_on_only_one_union_member(self):
        assert _path_resolves("top.variant.value", _WalkerRoot) is True

    def test_resolves_other_field_present_on_only_the_other_union_member(self):
        assert _path_resolves("top.variant.count", _WalkerRoot) is True

    def test_rejects_field_present_on_no_union_member(self):
        assert _path_resolves("top.variant.bogus", _WalkerRoot) is False

    def test_resolves_through_dict_of_models_wildcard_key(self):
        assert _path_resolves("top.items.anykey.value", _WalkerRoot) is True

    def test_rejects_field_missing_after_dict_wildcard_key(self):
        assert _path_resolves("top.items.anykey.bogus", _WalkerRoot) is False

    def test_resolves_scalar_leaf_as_final_segment(self):
        assert _path_resolves("top.name", _WalkerRoot) is True

    def test_rejects_descending_past_a_scalar_leaf(self):
        assert _path_resolves("top.name.extra", _WalkerRoot) is False

    def test_rejects_unknown_top_level_field(self):
        assert _path_resolves("bogus.path", _WalkerRoot) is False


def test_real_schema_markers_are_discovered():
    # Guard the guard: the real schemas now carry ConditionalOptional
    # fields (multigrid, precondition, convergence_control, motion, a wall's
    # most block, ...) — if the collector ever reports none, the path test
    # below has gone blind, not clean.
    markers = _collect_applies_when_markers(ZCFDInputModelBase)
    marker_fields = {name for _, name, _ in markers}
    assert "multigrid" in marker_fields
    assert {"moving_mesh", "ramp_time", "rotate_nodes"} <= marker_fields
    assert ("WallBoundary", "most") in {
        (getattr(owner, "__name__", owner), name) for owner, name, _ in markers
    }, "dict-of-union unwrapping regressed — boundary-condition markers invisible"


def _parents_of(model_cls, root_cls):
    """Every model class holding a field whose value may be ``model_cls``."""
    return {
        candidate
        for candidate in _collect_model_classes(root_cls)
        for field_info in candidate.model_fields.values()
        if any(
            target is model_cls for _, target in _unwrap_targets(field_info.annotation)
        )
    }


def _branches(when):
    return getattr(when, "branches", (when,))


def test_real_schema_applies_when_paths_all_resolve():
    markers = _collect_applies_when_markers(ZCFDInputModelBase)

    for owning_model_cls, field_name, when in markers:
        for path in (p for branch in _branches(when) for p in branch.conditions):
            if path.startswith("<self>."):
                resolved = _segments_resolve(path.split(".")[1:], {owning_model_cls})
            elif path.startswith("<parent>."):
                parents = _parents_of(owning_model_cls, ZCFDInputModelBase)
                resolved = bool(parents) and _segments_resolve(
                    path.split(".")[1:], parents
                )
            else:
                resolved = _path_resolves(path, ZCFDInputModelBase)
            assert resolved, (
                f"{owning_model_cls.__name__}.{field_name}: When() condition "
                f"path {path!r} does not resolve to a real field chain from "
                "ZCFDInputModelBase"
            )


def test_real_schema_exclusive_members_are_sibling_fields():
    found = []
    for model_cls in _collect_model_classes(ZCFDInputModelBase):
        for name, field_info in model_cls.model_fields.items():
            for marker in field_info.metadata:
                if isinstance(marker, Exclusive):
                    found.append(name)
                    for other in marker.others:
                        assert other in model_cls.model_fields, (
                            f"{model_cls.__name__}.{name}: Exclusive names "
                            f"{other!r}, which is not a field of the same model"
                        )
    assert {"scalar", "temperature", "mass_flow_rate"} <= set(found)


# ---------------------------------------------------------------------------
# 5b. AnyOf, <parent>, ConditionalRequired, ConditionalDefault, Exclusive
# ---------------------------------------------------------------------------


class TestAnyOf:
    def test_holds_when_any_branch_holds(self):
        clause = AnyOf(When({"a": "x"}), When({"b": "y"}))
        assert clause.evaluate({"a": "x"})
        assert clause.evaluate({"b": "y"})
        assert not clause.evaluate({"a": "z", "b": "z"})

    def test_mapping_branches_become_when_clauses(self):
        clause = AnyOf({"a": "x"}, {"b": "y"})
        assert all(isinstance(b, When) for b in clause.branches)

    def test_needs_two_branches(self):
        with pytest.raises(ValueError):
            AnyOf(When({"a": "x"}))

    def test_branches_must_be_when_clauses(self):
        with pytest.raises(TypeError):
            AnyOf(When({"a": "x"}), "b")

    def test_schema_lists_each_branch(self):
        clause = AnyOf(When({"a": "x"}), When({"b": NotIn("y")}))
        assert clause.to_schema() == {"any_of": [{"a": ["x"]}, {"b": {"not": ["y"]}}]}

    def test_describe_without_shared_conditions(self):
        clause = AnyOf(When({"<self>.type": "wall"}), When({"<self>.kind": NotIn("x")}))
        assert (
            clause.describe("temperature")
            == "'temperature' only applies when either type is 'wall' or kind is not 'x'"
        )

    def test_describe_factors_out_shared_conditions(self):
        clause = AnyOf(
            When({"<parent>.kind": "wall function", "<parent>.temperature": SET}),
            When({"<parent>.kind": "wall function", "<parent>.heat_flux": SET}),
        )
        assert clause.condition_text() == (
            "kind is 'wall function' and either temperature is set or heat flux is set"
        )

    def test_anchor_flags_cover_every_branch(self):
        clause = AnyOf(When({"a": "x"}), When({"<parent>.b": "y"}))
        assert clause.uses_parent_anchor and not clause.uses_self_anchor

    def test_accepted_by_conditional_optional(self):
        typ = ConditionalOptional[int, AnyOf(When({"a": "x"}), When({"b": "y"}))]
        marker = typing.get_args(typ)[1]
        assert isinstance(marker.when, AnyOf)


class TestDeckSpelling:
    def test_describe_uses_the_deck_spelling_of_keys(self):
        assert (
            When({"solver.time_settings.kind": "x"}).describe("heat_flux")
            == "'heat flux' only applies when solver.time settings.kind is 'x'"
        )


class _Wall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = "plain"
    heat: Optional[float] = None
    most: ConditionalRequired[int, When({"<self>.kind": "most"})] = None
    alpha: Annotated[Optional[float], Exclusive("beta")] = None
    beta: Optional[float] = None
    gamma: Annotated[Optional[float], Exclusive("delta", required=True)] = 1.0
    delta: Optional[float] = None


class _Rough(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ConditionalDefault[
        str, When({"<parent>.kind": "hot", "<parent>.heat": SET})
    ] = "default"


class _RoughWall(_Wall):
    rough: Optional[_Rough] = None


class _Root(BaseModel):
    model_config = ConfigDict(extra="forbid")

    walls: Dict[str, _RoughWall] = {}
    rough: Optional[_Rough] = None


class TestParentAnchor:
    def test_holds_on_the_containing_model(self):
        root = _Root(
            walls={"w": _RoughWall(kind="hot", heat=1.0, rough={"model": "x"})}
        )
        assert find_applicability_violations(root) == []

    def test_violation_names_the_owning_key(self):
        root = _Root(walls={"w": _RoughWall(kind="plain", rough={"model": "x"})})
        assert find_applicability_violations(root) == [
            (
                "walls.w.rough.model",
                "'rough.model' only applies when kind is 'hot' and heat is set",
            )
        ]

    def test_never_holds_without_a_parent_model(self):
        root = _Root(rough={"model": "x"})
        assert [p for p, _ in find_applicability_violations(root)] == ["rough.model"]


class TestConditionalDefault:
    def test_keeps_type_and_default(self):
        assert _Rough().model == "default"
        assert _Rough.model_fields["model"].annotation is str

    def test_unwritten_default_is_accepted(self):
        root = _Root(walls={"w": _RoughWall(rough={})})
        assert find_applicability_violations(root) == []

    def test_written_default_is_rejected_while_inapplicable(self):
        root = _Root(walls={"w": _RoughWall(rough={"model": "default"})})
        assert len(find_applicability_violations(root)) == 1

    def test_marker_is_written(self):
        marker = _Rough.model_fields["model"].metadata[0]
        assert marker.written and not marker.required


class TestConditionalRequired:
    def test_missing_while_applicable_is_rejected(self):
        root = _Root(walls={"w": _RoughWall(kind="most")})
        assert find_applicability_violations(root) == [
            ("walls.w.most", "'most' is required when kind is 'most'")
        ]

    def test_set_while_inapplicable_is_rejected(self):
        root = _Root(walls={"w": _RoughWall(most=3)})
        assert find_applicability_violations(root) == [
            ("walls.w.most", "'most' only applies when kind is 'most'")
        ]

    def test_set_while_applicable_is_accepted(self):
        root = _Root(walls={"w": _RoughWall(kind="most", most=3)})
        assert find_applicability_violations(root) == []

    def test_schema_marks_it_required_when_it_applies(self):
        schema = _Wall.model_json_schema()["properties"]["most"]
        assert schema["zcfd"]["applies_when"] == {"<self>.kind": ["most"]}
        assert schema["zcfd"]["required_when_applies"] is True

    def test_value_gated_clause_cannot_be_required(self):
        with pytest.raises(ValueError):
            AppliesWhen(When({"a": "x"}), values=("v",), required=True)


class _Inlet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: Optional[str] = None
    ratio: Annotated[
        ConditionalOptional[float, When({"<self>.ref": SET})],
        RequiredWhen(When({"mode": "hot"})),
    ] = None


class _InletRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "cold"
    inlet: _Inlet = _Inlet()


class TestRequiredWhen:
    def test_missing_while_required_is_rejected(self):
        root = _InletRoot(mode="hot", inlet={"ref": "a"})
        assert find_applicability_violations(root) == [
            ("inlet.ratio", "'ratio' is required when mode is 'hot'")
        ]

    def test_missing_while_not_required_is_accepted(self):
        root = _InletRoot(inlet={"ref": "a"})
        assert find_applicability_violations(root) == []

    def test_set_while_not_required_is_accepted(self):
        root = _InletRoot(inlet={"ref": "a", "ratio": 1.0})
        assert find_applicability_violations(root) == []

    def test_missing_while_required_and_inapplicable_names_both_clauses(self):
        root = _InletRoot(mode="hot")
        assert find_applicability_violations(root) == [
            (
                "inlet.ratio",
                "'ratio' is required when mode is 'hot', and only applies when ref is set",
            )
        ]

    def test_where_it_applies_is_left_to_its_own_clause(self):
        root = _InletRoot(mode="hot", inlet={"ratio": 1.0})
        assert find_applicability_violations(root) == [
            ("inlet.ratio", "'ratio' only applies when ref is set")
        ]

    def test_schema_carries_both_clauses(self):
        zcfd = _Inlet.model_json_schema()["properties"]["ratio"]["zcfd"]
        assert zcfd["applies_when"] == {"<self>.ref": ["__present__"]}
        assert zcfd["required_when"] == {"mode": ["hot"]}
        assert "required_when_applies" not in zcfd


class TestExclusive:
    def test_one_member_is_accepted(self):
        root = _Root(walls={"w": _RoughWall(beta=1.0)})
        assert find_applicability_violations(root) == []

    def test_both_members_are_rejected(self):
        root = _Root(walls={"w": _RoughWall(alpha=1.0, beta=2.0)})
        assert find_applicability_violations(root) == [
            ("walls.w.alpha", "give either 'alpha' or 'beta', not both")
        ]

    def test_required_group_with_no_member_is_rejected(self):
        root = _Root(walls={"w": _RoughWall(gamma=None)})
        assert find_applicability_violations(root) == [
            ("walls.w.gamma", "give either 'gamma' or 'delta'")
        ]

    def test_three_way_wording(self):
        marker = Exclusive("mass_flow_ratio", "b")
        assert marker.describe("a") == "give only one of 'a', 'mass flow ratio' or 'b'"
        assert marker.describe_missing("a") == (
            "give one of 'a', 'mass flow ratio' or 'b'"
        )

    def test_schema(self):
        schema = _Wall.model_json_schema()["properties"]
        assert schema["alpha"]["zcfd"]["exclusive"] == {
            "with": ["beta"],
            "required": False,
        }
        assert schema["gamma"]["zcfd"]["exclusive"]["required"] is True

    def test_needs_another_member(self):
        with pytest.raises(ValueError):
            Exclusive()


# ---------------------------------------------------------------------------
# 6. ZCFDInputModelBase integration
# ---------------------------------------------------------------------------


class TestZCFDInputModelBaseIntegration:
    def test_cross_block_check_registered_as_after_model_validator(self):
        decorators = ZCFDInputModelBase.__pydantic_decorators__.model_validators
        assert "check_across_blocks" in decorators
        assert decorators["check_across_blocks"].info.mode == "after"

    def test_minimal_valid_config_still_validates(self):
        """The applicability validator must not reject a real config that
        contains no ConditionalOptional fields at all — find_applicability_
        violations() should simply return no violations and get out of the
        way.
        """
        data = _minimal_valid_config()
        model = ZCFDInputModelBase.model_validate(data)
        assert model is not None
