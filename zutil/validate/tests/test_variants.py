"""
Tests for zcfd.variants schema emission — the ``variants=`` kwarg of
zCFD_Field (schemas/base_types.py) that encodes a context-routed union
pydantic cannot discriminate natively: the variant is picked by *other*
settings, resolved by a before-validator, and the same routing table is
serialised into the JSON schema (``zcfd.variants``) so the UI can offer the
identical choice. See zCFD_Field's docstring for the wire format:
``[{"when": {path: value, ...}, "ref": "#/$defs/<ClassName>"}, ...]``.

Section 1 walks the real ZCFDInputModelBase schema and checks the emitted
nodes are well-formed and internally consistent (refs resolve to real
$defs, "when" condition paths resolve to real field chains — reusing the
path-resolution walker from test_applicability.py). Section 2 spot-checks
specific routing tables against the real schema. Section 3 is a toy-model
regression test for the zCFD_Field merge guard: a field using BOTH
ConditionalOptional (annotation-level applies_when) AND
zCFD_Field(variants=..., advanced=...) (field-level extras) must not have
either side clobber the other under the single 'zcfd' key.
"""

from typing import Any, List, Tuple, Union

from pydantic import BaseModel, ConfigDict

from zutil.validate.schemas.base_types import ConditionalOptional, When, zCFD_Field
from zutil.validate.schemas import block_versions as versions
from zutil.validate.tests.test_applicability import _path_resolves
from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _collect_variants_nodes(node: Any, path: str = "root") -> List[Tuple[str, list]]:
    """(path, variants-list) for every JSON-schema node carrying
    ``zcfd.variants``, found by a recursive dict/list walk over the full
    ``model_json_schema()`` output (properties, $defs, anyOf branches,
    array items — a variants field can sit at any depth in there).
    """
    found: List[Tuple[str, list]] = []
    if isinstance(node, dict):
        zcfd = node.get("zcfd")
        if isinstance(zcfd, dict) and "variants" in zcfd:
            found.append((path, zcfd["variants"]))
        for key, value in node.items():
            found.extend(_collect_variants_nodes(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(_collect_variants_nodes(value, f"{path}[{i}]"))
    return found


# ---------------------------------------------------------------------------
# 1. Real-schema discovery and well-formedness
# ---------------------------------------------------------------------------


class TestVariantsNodesDiscoveredInRealSchema:
    def test_at_least_9_variants_nodes_exist_today(self):
        """Guard the guard: the solver block's seven table-driven blocks
        (solver_settings, equations, fluid_properties, reference_conditions,
        time_settings, convergence_control, numerical_scheme) and the two
        per-mesh overrides contribute at least 9 zcfd.variants nodes. If this
        ever drops, the checks below have gone blind, not clean.
        """
        schema = ZCFDInputModelBase.model_json_schema()
        nodes = _collect_variants_nodes(schema)
        assert len(nodes) >= 9

    def test_every_entry_has_a_dict_when_and_a_resolving_def_ref(self):
        schema = ZCFDInputModelBase.model_json_schema()
        defs = schema.get("$defs", {})
        nodes = _collect_variants_nodes(schema)
        assert nodes, "no zcfd.variants nodes found — walker or schema regressed"

        for path, variants in nodes:
            assert isinstance(variants, list) and variants, (
                f"{path}: empty variants list"
            )
            for entry in variants:
                assert "when" in entry, f"{path}: entry missing 'when': {entry}"
                assert isinstance(entry["when"], dict), (
                    f"{path}: 'when' is not a dict: {entry}"
                )
                assert "ref" in entry, f"{path}: entry missing 'ref': {entry}"
                ref = entry["ref"]
                assert isinstance(ref, str) and ref.startswith("#/$defs/"), (
                    f"{path}: malformed ref {ref!r}"
                )
                def_name = ref[len("#/$defs/") :]
                assert def_name in defs, (
                    f"{path}: ref {ref!r} does not resolve to a real $def "
                    f"(entry: {entry})"
                )

    def test_every_when_condition_path_resolves_to_a_real_field_chain(self):
        schema = ZCFDInputModelBase.model_json_schema()
        nodes = _collect_variants_nodes(schema)
        assert nodes

        for path, variants in nodes:
            for entry in variants:
                for cond_path in entry["when"]:
                    assert _path_resolves(cond_path, ZCFDInputModelBase), (
                        f"{path}: variants entry {entry!r} references "
                        f"condition path {cond_path!r} that does not "
                        "resolve to a real field chain from "
                        "ZCFDInputModelBase"
                    )


# ---------------------------------------------------------------------------
# 2. Spot-checks of specific routing tables
# ---------------------------------------------------------------------------


class TestSolverIsOneBlock:
    def test_solver_is_a_single_block(self):
        schema = ZCFDInputModelBase.model_json_schema()
        assert schema["properties"]["solver"]["$ref"] == "#/$defs/SolverBlockModel"


def _solver_variants(field):
    schema = ZCFDInputModelBase.model_json_schema()
    return schema["$defs"]["SolverBlockModel"]["properties"][field]["zcfd"]["variants"]


class TestTimeSettingsVariantsRouting:
    def test_routes_steady_gts_dts_and_unset(self):
        refs = [
            (entry["when"], entry["ref"]) for entry in _solver_variants("time_settings")
        ]
        assert refs == [
            ({"solver.time_settings.type": "steady"}, "#/$defs/SteadyTimeSettings"),
            (
                {
                    "solver.time_settings.type": "unsteady",
                    "solver.time_settings.kind": "global timestepping",
                },
                "#/$defs/GTSTimeSettings",
            ),
            (
                {
                    "solver.time_settings.type": "unsteady",
                    "solver.time_settings.kind": "dual time stepping",
                },
                "#/$defs/DTSTimeSettings",
            ),
            ({"solver.time_settings.type": None}, "#/$defs/SteadyTimeSettings"),
        ]


class TestNumericalSchemeVariantsRouting:
    def test_one_entry_per_table_row(self):
        expected = [
            {"when": when, "ref": f"#/$defs/{cls.__name__}"}
            for when, cls in versions.NUMERICAL_SCHEME
        ]
        assert _solver_variants("numerical_scheme") == expected
        assert len(expected) == 7


# ---------------------------------------------------------------------------
# 3. zCFD_Field variants= / ConditionalOptional merge guard
# ---------------------------------------------------------------------------


class _VariantA(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = 1


class _VariantB(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = "x"


class TestZCFDFieldVariantsMergeGuard:
    """A field carrying BOTH ConditionalOptional (annotation-level
    applies_when, contributed by AppliesWhen.__get_pydantic_json_schema__)
    AND zCFD_Field(variants=..., advanced=...) (field-level extras merged
    in _merge_extra) must end up with applies_when, variants AND advanced
    all present under the single 'zcfd' key — this is exactly the shape
    ModelSettings.convergence_control and SolverBlockModel.numerical_scheme
    use in the real schemas.
    """

    def test_applies_when_variants_and_advanced_all_coexist(self):
        class Toy(BaseModel):
            model_config = ConfigDict(extra="forbid")

            mode: str = "off"
            combo: ConditionalOptional[
                Union[_VariantA, _VariantB], When({"mode": "special"})
            ] = zCFD_Field(
                default=None,
                advanced=True,
                variants=[
                    ({"mode": "special"}, _VariantA),
                    ({}, _VariantB),
                ],
            )

        schema = Toy.model_json_schema()
        node = schema["properties"]["combo"]
        zcfd = node["zcfd"]

        # From AppliesWhen (annotation-level).
        assert zcfd["applies_when"] == {"mode": ["special"]}
        # From zCFD_Field (field-level) — must survive the merge.
        assert zcfd["advanced"] is True
        assert zcfd["variants"] == [
            {"when": {"mode": "special"}, "ref": "#/$defs/_VariantA"},
            {"when": {}, "ref": "#/$defs/_VariantB"},
        ]
