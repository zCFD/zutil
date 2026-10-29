"""Tests for union resolution in the reference generator."""

from typing import Annotated, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field, Tag

from zutil.docs.introspect import (
    UnresolvedUnionError,
    Variant,
    resolve_union,
    shape_of,
)
from zutil.docs.tree import Walker
from zutil.validate.schemas.base_types import zCFD_Field


class Alpha(BaseModel):
    type: Literal["alpha"] = "alpha"
    a: int = 1


class Beta(BaseModel):
    type: Literal["beta"] = "beta"
    b: float = 2.0


class GammaOne(BaseModel):
    type: Literal["gamma"] = "gamma"
    kind: Literal["one"] = "one"


class GammaTwo(BaseModel):
    type: Literal["gamma"] = "gamma"
    kind: Literal["two"] = "two"


Gamma = Annotated[Union[GammaOne, GammaTwo], Field(discriminator="kind")]


class TestResolve:
    def test_literal_discriminator(self):
        disc, leaves = resolve_union(
            Annotated[Union[Alpha, Beta], Field(discriminator="type")], "x"
        )
        assert disc == "type"
        assert [(v.model, v.tag) for v in leaves] == [(Alpha, "alpha"), (Beta, "beta")]

    def test_tagged_members(self):
        tagged = Annotated[
            Union[Annotated[Alpha, Tag("first")], Annotated[Beta, Tag("second")]],
            Field(discriminator="type"),
        ]
        _, leaves = resolve_union(tagged, "x")
        assert [v.tag for v in leaves] == ["first", "second"]

    def test_two_stage_union_flattens_to_leaves(self):
        two = Annotated[
            Union[Annotated[Alpha, Tag("alpha")], Annotated[Gamma, Tag("gamma")]],
            Field(discriminator="type"),
        ]
        _, leaves = resolve_union(two, "x")
        assert [v.when for v in leaves] == [
            {"type": "alpha"},
            {"type": "gamma", "kind": "one"},
            {"type": "gamma", "kind": "two"},
        ]

    def test_shared_literal_on_undeclared_union(self):
        disc, leaves = resolve_union(Union[Alpha, Beta], "x")
        assert disc == "type"
        assert [v.model for v in leaves] == [Alpha, Beta]

    def test_nested_union_member_without_tag_fails(self):
        with pytest.raises(UnresolvedUnionError) as err:
            resolve_union(Union[Gamma, Alpha], "x.field")
        assert err.value.location == "x.field"
        assert err.value.members == ["Union[GammaOne, GammaTwo]"]

    def test_routed_union_resolves_from_variants(self):
        class Holder(BaseModel):
            value: Union[Alpha, Beta] = zCFD_Field(
                variants=[({"solver.x": "a"}, Alpha), ({"solver.x": "b"}, Beta)]
            )

        shape = shape_of(Holder.model_fields["value"], "value")
        assert shape.unresolved is None
        assert [(v.model, v.when) for v in shape.variants] == [
            (Alpha, {"solver.x": "a"}),
            (Beta, {"solver.x": "b"}),
        ]

    def test_routing_that_misses_a_member_fails(self):
        class Holder(BaseModel):
            value: Union[Alpha, Beta] = zCFD_Field(
                variants=[({"solver.x": "a"}, Alpha)]
            )

        shape = shape_of(Holder.model_fields["value"], "value")
        assert shape.unresolved is not None
        assert shape.unresolved.members == ["Beta"]

    def test_scalar_alternative_is_not_a_member(self):
        class Holder(BaseModel):
            value: Optional[Union[str, Alpha]] = None

        assert shape_of(Holder.model_fields["value"], "value").kind == "block"


class TestRealTree:
    def test_fsi_resolves(self):
        walker = Walker()
        walker.build()
        assert not [e for e in walker.unresolved if e.location.endswith(".fsi")]

    def test_every_union_resolves(self):
        walker = Walker()
        walker.build()
        assert walker.unresolved == []

    def test_unresolved_entry_is_marked_not_dropped(self):
        class Holder(BaseModel):
            value: Union[Gamma, Alpha] = Alpha()

        walker = Walker()
        entries = walker.entries(
            [Variant(Holder, None, {})], [], frozenset({Holder}), "x"
        )
        entry = next(e for e in entries if e["key"] == "value")
        assert entry["unresolved"]["members"] == ["Union[GammaOne, GammaTwo]"]


def _type_entry(holder: type) -> dict:
    walker = Walker()
    entries = walker.entries([Variant(holder, None, {})], [], frozenset({holder}), "x")
    value = next(e for e in entries if e["key"] == "value")
    return next(e for e in value["fields"] if e["key"] == "type")


class TestDiscriminatorEntry:
    def test_no_option_is_default_without_a_union_default(self):
        class Holder(BaseModel):
            value: Annotated[Union[Alpha, Beta], Field(discriminator="type")]

        entry = _type_entry(Holder)
        assert [o["default"] for o in entry["options"]] == [False, False]
        assert entry["facts"]["default"]["show"] is False

    def test_union_default_marks_only_its_tag(self):
        class Holder(BaseModel):
            value: Annotated[Union[Alpha, Beta], Field(discriminator="type")] = Field(
                default_factory=Beta
            )

        entry = _type_entry(Holder)
        assert {o["value"]: o["default"] for o in entry["options"]} == {
            "alpha": False,
            "beta": True,
        }
        assert entry["facts"]["default"] == {"value": "beta", "show": True}

    def test_options_carry_no_variant_label(self):
        class Holder(BaseModel):
            value: Annotated[Union[Alpha, Beta], Field(discriminator="type")]

        entry = _type_entry(Holder)
        assert [o["variants"] for o in entry["options"]] == [None, None]

    def test_bare_restatement_inherits_the_base_description(self):
        class Base(BaseModel):
            type: Literal["alpha", "beta"] = zCFD_Field(
                ..., description="Which flavour to use."
            )

        class A(Base):
            type: Literal["alpha"] = "alpha"

        class B(Base):
            type: Literal["beta"] = "beta"

        class Holder(BaseModel):
            value: Annotated[Union[A, B], Field(discriminator="type")]

        entry = _type_entry(Holder)
        assert entry["description"] == "Which flavour to use."
        assert entry["gap"] is False

    def test_equations_type_has_a_description_and_no_default(self):
        walker = Walker()
        page = next(
            p for p in walker.build() if p.route == "reference/solver/equations"
        )
        entry = next(e for e in page.entries if e["key"] == "type")
        assert entry["description"]
        assert not any(o["default"] for o in entry["options"])
