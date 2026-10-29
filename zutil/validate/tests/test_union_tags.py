"""Each set of type tags is written once, on the members of its union.

Three guards keep it so:

* a model outside a union that offers its tags as choices (a UI base
  model such as ``BoundaryConditionBase``) offers exactly those tags;
* a ``Tag(...)`` written on a union member names the same value as the
  member's own ``Literal``;
* no schema module spells a union's full tag set out by hand, in a
  ``Literal``, tuple, list or set. A dict keyed by tags (an ``options``
  table of descriptions) is allowed, and its keys are checked against
  the union instead.
"""

import ast
import importlib
import pathlib
import pkgutil
import typing
from typing import Annotated, Literal, Union

import pytest
from pydantic import BaseModel, Field, Tag

import zutil.validate.schemas as schemas_package
from zutil.validate.schemas import union_tags
from zutil.docs.introspect import zcfd_extra
from zutil.validate.schemas.boundary_conditions_schemas import (
    BoundaryCondition,
    BoundaryConditionBase,
    InflowBoundary,
    InterfaceBoundaryBase,
)
from zutil.validate.schemas.convergence_schemas import (
    EXPLICIT_SCHEME_NAMES,
    IMPLICIT_SCHEME_NAMES,
    TimeScheme,
)
from zutil.validate.schemas.fluid_zones_schemas import (
    FluidZoneBase,
    Fluid_Zone_Base_Model,
)
from zutil.validate.schemas.union_tags import literal_of, tags_of

SCHEMAS_DIR = pathlib.Path(schemas_package.__file__).parent

KNOWN_RESTATEMENTS: dict = {}

DIFFERENT_CHOICES = {
    ("GenericFSIModel", "transform_type"): (
        "the generic coupling offers every mesh morpher the solver has, plain "
        "'rbf' included; ModalModel has a version for two of them"
    ),
}


def _schema_modules():
    for info in pkgutil.iter_modules(schemas_package.__path__):
        yield importlib.import_module(f"{schemas_package.__name__}.{info.name}")


def _unions():
    """Every discriminated union bound to a module-level name, by qualified name."""
    found = {}
    for module in _schema_modules():
        for name, value in vars(module).items():
            try:
                field = union_tags.discriminator_of(value)
            except TypeError:
                continue
            if field is not None:
                found[f"{module.__name__.rsplit('.', 1)[-1]}.{name}"] = value
    return found


UNIONS = _unions()


def _models():
    seen = {}
    for module in _schema_modules():
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, BaseModel):
                seen[value] = None
    return list(seen)


def _strip(hint):
    while typing.get_origin(hint) is Annotated:
        hint = typing.get_args(hint)[0]
    return hint


def _leaf_models(union):
    for member in union_tags.members_of(union):
        inner = _strip(member)
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            yield inner
        else:
            yield from _leaf_models(member)


def _ui_base_fields():
    """(union name, model, discriminator) for every model outside a union that
    offers two or more of its tags on the union's discriminator field."""
    out = []
    models = _models()
    for name, union in UNIONS.items():
        field = union_tags.discriminator_of(union)
        tags = set(tags_of(union))
        leaves = set(_leaf_models(union))
        for model in models:
            if model in leaves or field not in model.model_fields:
                continue
            offered = typing.get_args(_strip(model.model_fields[field].annotation))
            if (model.__name__, field) in DIFFERENT_CHOICES:
                continue
            if len(tags & set(offered)) > 1:
                out.append((name, model, field))
    return out


class TestTagsOf:
    def test_reads_member_literals_in_order(self):
        assert tags_of(TimeScheme) == (
            "euler",
            "runge kutta",
            "implicit euler",
            "lu-sgs",
            "mf-gmres",
        )

    def test_nested_union_member_contributes_its_shared_tag(self):
        tags = tags_of(BoundaryCondition)
        assert "inflow" in tags and "pressure" not in tags
        assert len(tags) == len(set(tags))

    def test_kinds_of_a_nested_union(self):
        assert tags_of(InflowBoundary) == ("pressure", "massflow", "default")

    def test_bare_union_needs_the_field(self):
        class A(BaseModel):
            kind: Literal["a"] = "a"

        class B(BaseModel):
            kind: Literal["b"] = "b"

        with pytest.raises(TypeError):
            tags_of(Union[A, B])
        assert tags_of(Union[A, B], "kind") == ("a", "b")

    def test_non_literal_discriminator_is_refused(self):
        class A(BaseModel):
            kind: str = "a"

        class B(BaseModel):
            kind: Literal["b"] = "b"

        with pytest.raises(TypeError):
            tags_of(Annotated[Union[A, B], Field(discriminator="kind")])

    def test_literal_of_round_trips(self):
        assert typing.get_args(literal_of(TimeScheme)) == tags_of(TimeScheme)


class TestUIBaseModelsOfferTheUnionTags:
    def test_the_known_base_models_are_found(self):
        found = {(model, field) for _, model, field in _ui_base_fields()}
        assert (BoundaryConditionBase, "type") in found
        assert (FluidZoneBase, "type") in found
        assert (InterfaceBoundaryBase, "kind") in found

    @pytest.mark.parametrize(
        "union_name,model,field",
        _ui_base_fields(),
        ids=lambda v: v.__name__ if isinstance(v, type) else str(v),
    )
    def test_choices_equal_union_tags(self, union_name, model, field):
        offered = typing.get_args(_strip(model.model_fields[field].annotation))
        assert offered == tags_of(UNIONS[union_name])
        options = zcfd_extra(model.model_fields[field]).get("options")
        if options:
            assert set(options) == set(offered)

    def test_fluid_zone_choices_are_the_real_tags(self):
        offered = set(typing.get_args(FluidZoneBase.model_fields["type"].annotation))
        assert {"rbf transform", "idw transform"} <= offered
        assert not {"rbf", "rbf_multiscale", "idw", "fsi"} & offered
        assert offered == set(tags_of(Fluid_Zone_Base_Model))


class TestWrittenTagsAgreeWithLiterals:
    @pytest.mark.parametrize("name", sorted(UNIONS))
    def test_every_tag_annotation_matches_its_member(self, name):
        union = UNIONS[name]
        field = union_tags.discriminator_of(union)
        for member in union_tags.members_of(union):
            written = [m.tag for m in typing.get_args(member)[1:] if isinstance(m, Tag)]
            if written:
                assert written == list(union_tags._member_tags(member, field))


class TestSchemeNameSets:
    def test_explicit_and_implicit_partition_the_time_schemes(self):
        assert set(EXPLICIT_SCHEME_NAMES).isdisjoint(IMPLICIT_SCHEME_NAMES)
        assert EXPLICIT_SCHEME_NAMES + IMPLICIT_SCHEME_NAMES == tags_of(TimeScheme)


def _string_collections(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id != "Literal":
                continue
            elts = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
        elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            elts = node.elts
        else:
            continue
        if elts and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elts
        ):
            yield node, frozenset(e.value for e in elts)


def _assigned_name(tree, target):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.value is target:
            return getattr(node.targets[0], "id", None)
        if isinstance(node, ast.Subscript) and node.slice is target:
            return _assigned_name(tree, node)
    return None


def _restatements():
    tag_sets = {
        frozenset(tags_of(union)): name
        for name, union in UNIONS.items()
        if len(tags_of(union)) > 1
    }
    out = []
    for path in sorted(SCHEMAS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node, values in _string_collections(tree):
            if values in tag_sets:
                where = (path.name, _assigned_name(tree, node))
                out.append((where, node.lineno, tag_sets[values]))
    return out


class TestNoTagSetIsSpelledOut:
    def test_no_schema_module_restates_a_union_tag_set(self):
        unexpected = [
            f"{file}:{line} ({name}) restates the tags of {union}"
            for (file, name), line, union in _restatements()
            if (file, name) not in KNOWN_RESTATEMENTS
        ]
        assert not unexpected, "\n".join(unexpected)

    def test_known_restatements_still_exist(self):
        present = {where for where, _, _ in _restatements()}
        assert set(KNOWN_RESTATEMENTS) <= present
