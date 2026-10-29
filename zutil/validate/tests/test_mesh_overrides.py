"""A mesh's override blocks list only what differs from the solver's.

Initialisation, convergence control, numerical scheme and output settings
all follow one rule: a mesh block is laid over the solver block of the same
name, dicts merging key by key, so whatever the mesh leaves out comes from the
solver and not from a schema default. The C++, the Python driver, the log's
report sorter and post-processing read the resolved block.
"""

import copy
import json
from importlib.resources import files

import pytest

from zutil.Parameters import DeckReloader
from zutil.validate.tests.test_buoyancy_placement import (
    _incompressible_buoyancy_config,
)
from zutil.validate.tests.test_output_variable_names import _deck
from zutil.validate.validate_input import validate_zcfd_input
from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive
from zutil.validate.zcfd_schema import (
    MESH_OVERRIDES,
    ZCFDInputModelBase,
    inherited_settings,
    mesh_difference,
    resolve_mesh_settings,
)

CASES = json.loads(
    files("zutil.validate").joinpath("mesh_settings_inheritance.json").read_text()
)


def _mesh(block, settings, **solver):
    deck = _deck(volume_variables=["p", "V"], surface_variables=["p"])
    deck["solver"].update(solver)
    deck["model"]["block"][block] = settings
    cpp = validate_zcfd_input(deck).to_cpp_dict()
    return cpp["model"]["block"][block.replace(" ", "_")]


class TestEveryBlockInherits:
    def test_convergence_control_keeps_the_solver_cfl(self):
        resolved = _mesh("convergence control", {"cycles": 50})
        assert resolved["cycles"] == 50
        assert resolved["cfl"]["cfl"] == 2.0

    def test_numerical_scheme_keeps_the_solver_limiter(self):
        resolved = _mesh(
            "numerical scheme",
            {"order": "first"},
            **{"numerical scheme": {"order": "second", "limiter": "van leer"}},
        )
        assert resolved["order"] == "first"
        assert resolved["limiter"] == "van leer"

    def test_initialisation_keeps_the_solver_restart(self):
        resolved = _mesh(
            "initialisation",
            {"initial conditions": "IC_1"},
            **{"initialisation": {"initial conditions": "IC_1", "restart": True}},
        )
        assert resolved["restart"] is True

    def test_output_settings_keep_the_solver_solution(self):
        resolved = _mesh(
            "output settings", {"report": {"forces": {"FR_1": {"zones": [1]}}}}
        )
        assert resolved["solution"]["volume_variables"] == ["p", "V"]
        assert resolved["solution"]["format"] == "vtk"
        assert resolved["report"]["frequency"] == 10


class TestMerging:
    def test_named_reports_add_to_the_solver_reports(self):
        deck = _deck(volume_variables=["p"])
        deck["solver"]["output settings"]["report"]["monitor"] = {
            "MR_1": {"point": [0, 0, 0], "variables": ["p"]}
        }
        deck["model"]["block"]["output settings"] = {
            "report": {"forces": {"FR_1": {"zones": [1]}}}
        }
        report = validate_zcfd_input(deck).to_cpp_dict()["model"]["block"][
            "output_settings"
        ]["report"]
        assert set(report["monitor"]) == {"mr_1"}
        assert set(report["forces"]) == {"fr_1"}

    def test_a_list_the_mesh_sets_replaces_the_solver_list(self):
        resolved = _mesh("output settings", {"solution": {"volume variables": ["p"]}})
        assert resolved["solution"]["volume_variables"] == ["p"]
        assert resolved["solution"]["surface_variables"] == ["p"]

    def test_a_different_scheme_replaces_the_solver_scheme(self):
        resolved = _mesh(
            "convergence control",
            {"scheme": {"name": "runge kutta", "stage": 5}},
            **{
                "convergence control": {
                    "cycles": 10,
                    "cfl": {"cfl": 2.0},
                    "scheme": {"name": "lu-sgs", "sub_iterations": 4},
                }
            },
        )
        assert resolved["scheme"] == {"name": "runge kutta", "stage": 5}

    def test_the_same_scheme_merges(self):
        resolved = _mesh(
            "convergence control",
            {"scheme": {"name": "lu-sgs", "over_relaxation": 1.2}},
            **{
                "convergence control": {
                    "cycles": 10,
                    "cfl": {"cfl": 2.0},
                    "scheme": {"name": "lu-sgs", "sub_iterations": 4},
                }
            },
        )
        assert resolved["scheme"]["sub_iterations"] == 4
        assert resolved["scheme"]["over_relaxation"] == 1.2

    def test_an_incompressible_mesh_does_not_inherit_the_coupling_scheme(self):
        deck = _incompressible_buoyancy_config()
        mesh = next(iter(deck["model"]))
        deck["model"][mesh]["convergence_control"] = {"cycles": 50}
        cpp = validate_zcfd_input(deck).to_cpp_dict()
        assert "scheme" not in cpp["model"][mesh]["convergence_control"]
        assert cpp["model"][mesh]["convergence_control"]["cycles"] == 50

    def test_the_solver_block_is_untouched(self):
        deck = _deck(volume_variables=["p"])
        deck["model"]["block"]["convergence control"] = {"cycles": 50}
        cpp = validate_zcfd_input(deck).to_cpp_dict()
        assert cpp["solver"]["convergence_control"]["cycles"] == 10


class TestErrors:
    def test_a_mesh_may_request_averages_the_solver_computes(self):
        deck = _deck(volume_variables=["p"])
        deck["solver"]["output settings"]["compute average and rms"] = True
        deck["model"]["block"]["output settings"] = {
            "solution": {"volume variables": ["p", "p_avg"]}
        }
        resolved = validate_zcfd_input(deck).to_cpp_dict()["model"]["block"]
        assert resolved["output_settings"]["compute_average_and_rms"] is True

    def test_a_solver_error_a_mesh_inherits_is_reported_once(self):
        deck = _deck(volume_variables=["p"])
        deck["solver"]["numerical scheme"]["limiter"] = "minmod"
        deck["model"]["block"]["numerical scheme"] = {"order": "first"}
        _, nodes = validate_zcfd_comprehensive(copy.deepcopy(deck))
        paths = [n["field_path"] for n in nodes]
        assert len(paths) == 1
        assert paths[0].startswith("solver")

    def test_a_mesh_error_is_reported_at_the_mesh(self):
        deck = _deck(volume_variables=["p"])
        deck["model"]["block"]["numerical scheme"] = {"limiter": "minmod"}
        _, nodes = validate_zcfd_comprehensive(copy.deepcopy(deck))
        assert [n["field_path"].split(".")[0] for n in nodes] == ["model"]

    def test_a_mesh_that_repeats_a_solver_error_is_reported_at_both(self):
        deck = _deck(volume_variables=["p"])
        deck["solver"]["numerical scheme"]["limiter"] = "minmod"
        deck["model"]["block"]["numerical scheme"] = {"limiter": "minmod"}
        _, nodes = validate_zcfd_comprehensive(copy.deepcopy(deck))
        roots = sorted(n["field_path"].split(".")[0] for n in nodes)
        assert roots == ["model", "solver"]


class TestInheritedSettings:
    def _deck(self, mesh_scheme):
        deck = _deck(volume_variables=["p"])
        deck["solver"]["convergence_control"] = {
            "cycles": 10,
            "scheme": {"name": "lu-sgs", "sub_iterations": 4},
        }
        deck["model"]["block"]["convergence_control"] = {"scheme": mesh_scheme}
        return deck

    def test_a_key_the_mesh_leaves_out_comes_from_the_solver(self):
        inherited = inherited_settings(self._deck({"name": "lu-sgs"}))
        path = ("model", "block", "convergence_control", "scheme", "sub_iterations")
        assert inherited.source(path) == (
            "solver",
            "convergence_control",
            "scheme",
            "sub_iterations",
        )
        assert inherited.source(("model", "block", "convergence_control", "cycles"))

    def test_a_key_the_mesh_sets_is_its_own(self):
        inherited = inherited_settings(self._deck({"name": "lu-sgs"}))
        path = ("model", "block", "convergence_control", "scheme", "name")
        assert inherited.source(path) is None

    def test_a_replaced_sub_block_inherits_nothing(self):
        inherited = inherited_settings(self._deck({"name": "runge kutta"}))
        path = ("model", "block", "convergence_control", "scheme", "sub_iterations")
        assert inherited.source(path) is None


def _case_deck(case, mesh):
    return {
        "solver": {
            "solver_settings": {"type": case.get("solver_type", "compressible")},
            case["block"]: case["solver"],
        },
        "model": {"m": {case["block"]: mesh}},
    }


@pytest.mark.parametrize("case", CASES, ids=[c["case"] for c in CASES])
class TestConformanceCases:
    """The cases zui runs its own copy of the rule against."""

    def test_resolves(self, case):
        resolved = resolve_mesh_settings(_case_deck(case, case["mesh"]))
        assert resolved["model"]["m"][case["block"]] == case["resolved"]

    def test_the_difference_is_what_the_mesh_wrote(self, case):
        assert mesh_difference(case["solver"], case["resolved"]) == case["mesh"]

    def test_the_difference_resolves_back(self, case):
        written = mesh_difference(case["solver"], case["resolved"])
        resolved = resolve_mesh_settings(_case_deck(case, written))
        assert resolved["model"]["m"][case["block"]] == case["resolved"]

    def test_resolving_twice_changes_nothing(self, case):
        once = resolve_mesh_settings(_case_deck(case, case["mesh"]))
        assert resolve_mesh_settings(once) == once


class TestResolvingTwice:
    """validate_zcfd_comprehensive resolves a deck, then the model's before
    validator resolves it again, so a second pass must change nothing."""

    def _deck(self, mesh_scheme):
        deck = _deck(volume_variables=["p"])
        deck["solver"]["convergence_control"] = {
            "cycles": 10,
            "scheme": {"name": "lu-sgs", "sub_iterations": 4},
        }
        deck["model"]["block"]["convergence_control"] = {"scheme": mesh_scheme}
        deck["model"]["block"]["output_settings"] = {
            "solution": {"volume_variables": ["p"]}
        }
        return deck

    def test_a_merged_block_resolves_to_itself(self):
        once = resolve_mesh_settings(self._deck({"name": "LU-SGS"}))
        assert resolve_mesh_settings(once) == once

    def test_a_replaced_sub_block_resolves_to_itself(self):
        once = resolve_mesh_settings(self._deck({"name": "runge kutta", "stage": 5}))
        assert resolve_mesh_settings(once) == once

    def test_an_incompressible_mesh_resolves_to_itself(self):
        deck = _incompressible_buoyancy_config()
        mesh = next(iter(deck["model"]))
        deck["model"][mesh]["convergence_control"] = {"cycles": 50}
        once = resolve_mesh_settings(deck)
        assert resolve_mesh_settings(once) == once


class TestReload:
    KEY = "model:block:convergence_control:cfl:cfl"

    def _reload(self, tmp_path, before, after):
        path = tmp_path / "case.py"
        path.write_text(f"parameters = {before!r}\n")
        reloader = DeckReloader(str(path.with_suffix("")), str(path))
        reloader.prime()
        path.write_text(f"parameters = {after!r}\n")
        return reloader._reread_and_diff()

    def _deck(self, solver_cfl, mesh_block):
        deck = _deck(volume_variables=["p"])
        deck["solver"]["convergence control"]["cfl"] = {"cfl": solver_cfl}
        deck["model"]["block"]["convergence control"] = mesh_block
        return deck

    def test_an_inherited_edit_is_an_echo(self, tmp_path):
        update = self._reload(
            tmp_path,
            self._deck(1.0, {"cycles": 50}),
            self._deck(2.5, {"cycles": 50}),
        )
        assert self.KEY in update.accepted
        assert self.KEY in update.echoes

    def test_dropping_a_mesh_override_is_not_an_echo(self, tmp_path):
        update = self._reload(
            tmp_path,
            self._deck(1.0, {"cycles": 50, "cfl": {"cfl": 0.5}}),
            self._deck(1.0, {"cycles": 50}),
        )
        assert self.KEY in update.accepted
        assert self.KEY not in update.echoes


class TestTheRuleIsSchemaData:
    """A form resolves a mesh block as the user types, so the JSON schema carries the rule."""

    RULE = ZCFDInputModelBase.model_json_schema()["zcfd"]["mesh_settings"]

    def test_names_the_blocks_a_mesh_may_set(self):
        assert self.RULE["blocks"] == list(MESH_OVERRIDES)

    def test_names_the_variant_selectors(self):
        assert self.RULE["variant_selectors"] == ["name", "type", "kind"]

    def test_names_the_incompressible_run_wide_convergence_keys(self):
        assert self.RULE["not_inherited"] == {
            "convergence_control": {
                "when": {"solver.solver_settings.type": ["incompressible"]},
                "keys": ["scheme"],
            }
        }
