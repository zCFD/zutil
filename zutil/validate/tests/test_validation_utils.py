"""
Direct accept/reject coverage for expand_reference_inheritance and
replace_spaces_in_keys (validation_utils.py).
"""

import pytest

from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive
from zutil.validate.validation_utils import (
    expand_reference_inheritance,
    replace_spaces_in_keys,
)


class TestReplaceSpacesInKeys:
    def test_spaced_and_underscored_forms_normalize_identically(self):
        """A deck written with spaced keys and the same deck written with
        underscored keys must normalize to exactly the same structure -
        the two spellings are equally valid, not just individually
        accepted."""
        spaced = {
            "config version": 2,
            "solver": {
                "convergence control": {"cfl": 100.0, "cfl ramp": {"initial": 0.05}},
                "reference conditions": {
                    "IC_1": {"turbulence intensity": 0.01},
                },
            },
        }
        underscored = {
            "config_version": 2,
            "solver": {
                "convergence_control": {"cfl": 100.0, "cfl_ramp": {"initial": 0.05}},
                "reference_conditions": {
                    "IC_1": {"turbulence_intensity": 0.01},
                },
            },
        }
        assert replace_spaces_in_keys(spaced) == replace_spaces_in_keys(underscored)

    def test_collection_entry_names_preserved_verbatim(self):
        """Entry names inside a collection dict (reference_conditions,
        boundary_conditions, fluid_zones) are user-defined identifiers, not
        schema keys - they must never be normalized."""
        data = {"reference_conditions": {"IC_1": {"pressure": 101325}}}
        result = replace_spaces_in_keys(data)
        assert "IC_1" in result["reference_conditions"]

    def test_report_labels_keep_their_case_and_underscore_their_spaces(self):
        """Force, mass-flow and monitor labels are the deck author's names,
        so their case is kept for the engineer, but a label is also the stem
        of its report columns, and a column name carries no spaces. The
        settings inside them are schema keys and are normalised."""
        data = {
            "output settings": {
                "report": {
                    "forces": {"Main Wing": {"reference area": 2.0}},
                    "mass flow": {"Total Inlet": {"zones": [1]}},
                    "monitor": {"MP_1": {"point": [0, 0, 0]}},
                }
            }
        }
        report = replace_spaces_in_keys(data)["output_settings"]["report"]
        assert report["forces"] == {"Main_Wing": {"reference_area": 2.0}}
        assert report["mass_flow"] == {"Total_Inlet": {"zones": [1]}}
        assert list(report["monitor"]) == ["MP_1"]

    @pytest.mark.parametrize(
        "first, second",
        [
            ("Total Inlet", "Total_Inlet"),
            ("Wall", "wall"),
            ("Total Inlet", "total_inlet"),
        ],
    )
    def test_report_labels_that_name_the_same_columns_are_rejected(self, first, second):
        """The solver reads every label in lower case with underscores, so
        two labels that differ only in spacing or case would write the same
        columns and one report would silently replace the other."""
        data = {
            "report": {"mass_flow": {first: {"zones": [1]}, second: {"zones": [2]}}}
        }
        with pytest.raises(ValueError, match=f"'{first}' and '{second}'"):
            replace_spaces_in_keys(data)

    def test_boundary_condition_labels_are_still_verbatim(self):
        data = {"boundary_conditions": {"Inlet Face": {"zones": [1]}}}
        assert list(replace_spaces_in_keys(data)["boundary_conditions"]) == [
            "Inlet Face"
        ]

    def test_a_monitor_outside_the_report_is_not_a_label_collection(self):
        data = {"reference_conditions": {"IC_1": {"monitor": {"Update Report": True}}}}
        result = replace_spaces_in_keys(data)
        assert result["reference_conditions"]["IC_1"]["monitor"] == {
            "update_report": True
        }

    def test_duplicate_key_from_two_spellings_rejected(self):
        """Writing both spellings of the same key in one dict must fail
        loudly rather than let the later one silently win."""
        data = {
            "convergence_control": {"cfl": 50.0},
            "convergence control": {"cfl": 100.0},
        }
        with pytest.raises(ValueError, match="Duplicate key"):
            replace_spaces_in_keys(data)

    def test_duplicate_key_detected_at_nested_level(self):
        data = {
            "solver": {
                "cfl_ramp": {"initial": 0.05},
                "cfl ramp": {"initial": 1.0},
            }
        }
        with pytest.raises(ValueError, match="Duplicate key"):
            replace_spaces_in_keys(data)

    def test_duplicate_key_via_case_difference_also_rejected(self):
        """Normalization also lowercases, so a casing-only clash is caught
        the same way a spacing-only clash is."""
        data = {
            "Convergence_Control": {"cfl": 50.0},
            "convergence_control": {"cfl": 100.0},
        }
        with pytest.raises(ValueError, match="Duplicate key"):
            replace_spaces_in_keys(data)

    def test_real_deck_validates_identically_with_spaced_keys(self):
        """A complete, real deck (matching _minimal_valid_config in
        test_core_orchestration.py) validates successfully through the full
        schema - not just replace_spaces_in_keys - with every multi-word
        key written spaced, and produces the exact same validated model as
        the underscored form."""
        underscored = {
            "config_version": 2,
            "solver": {
                "solver_settings": {"type": "compressible"},
                "initialisation": {"initial_conditions": "IC_1"},
                "equations": {"type": "euler"},
                "numerical_scheme": {"order": "second"},
                "fluid_properties": {},
                "reference_conditions": {
                    "IC_1": {"temperature": 300, "pressure": 101325}
                },
                "output_settings": {
                    "solution": {"frequency": 10},
                    "report": {"frequency": 10},
                },
                "time_settings": {"type": "steady"},
                "convergence_control": {
                    "cycles": 100,
                    "cfl": 1.0,
                    "scheme": {"name": "runge kutta"},
                },
            },
            "model": {
                "base": {
                    "mesh": "base.h5",
                    "boundary_conditions": {
                        "bc1": {
                            "type": "farfield",
                            "condition": "IC_1",
                            "kind": "riemann",
                            "zones": [1],
                        }
                    },
                }
            },
        }
        spaced = {
            "config version": 2,
            "solver": {
                "solver settings": {"type": "compressible"},
                "initialisation": {"initial conditions": "IC_1"},
                "equations": {"type": "euler"},
                "numerical scheme": {"order": "second"},
                "fluid properties": {},
                "reference conditions": {
                    "IC_1": {"temperature": 300, "pressure": 101325}
                },
                "output settings": {
                    "solution": {"frequency": 10},
                    "report": {"frequency": 10},
                },
                "time settings": {"type": "steady"},
                "convergence control": {
                    "cycles": 100,
                    "cfl": 1.0,
                    "scheme": {"name": "runge kutta"},
                },
            },
            "model": {
                "base": {
                    "mesh": "base.h5",
                    "boundary conditions": {
                        "bc1": {
                            "type": "farfield",
                            "condition": "IC_1",
                            "kind": "riemann",
                            "zones": [1],
                        }
                    },
                }
            },
        }

        underscored_model, underscored_errors = validate_zcfd_comprehensive(underscored)
        spaced_model, spaced_errors = validate_zcfd_comprehensive(spaced)

        assert underscored_errors == []
        assert spaced_errors == []
        assert underscored_model is not None
        assert spaced_model is not None
        assert spaced_model.model_dump() == underscored_model.model_dump()

    def test_a_spaced_report_label_reaches_the_solver_underscored(self):
        """The engineer sees 'Total_Inlet'; the solver dict, whose keys the
        C++ looks up in lower case, carries 'total_inlet', so the report
        column is 'total_inlet_massflow'."""
        deck = _viscous_deck_with_condition(
            {
                "temperature": 300,
                "pressure": 101325,
                "v": {"vector": [1.0, 0.0, 0.0]},
                "reynolds_no": 1.0e6,
            }
        )
        deck["solver"]["output_settings"]["report"]["mass_flow"] = {
            "Total Inlet": {"zones": [1]}
        }
        deck["solver"]["output_settings"]["report"]["forces"] = {
            "Main Wing": {"zones": [1]}
        }
        model, errors = validate_zcfd_comprehensive(deck)
        assert errors == []
        report = model.solver.output_settings.report
        assert list(report.mass_flow) == ["Total_Inlet"]
        assert list(report.forces) == ["Main_Wing"]
        solver_report = model.to_cpp_dict()["solver"]["output_settings"]["report"]
        assert list(solver_report["mass_flow"]) == ["total_inlet"]
        assert list(solver_report["forces"]) == ["main_wing"]


def _viscous_deck_with_condition(condition) -> dict:
    return {
        "config_version": 2,
        "solver": {
            "solver_settings": {"type": "compressible"},
            "initialisation": {"initial_conditions": "IC_1"},
            "equations": {"type": "viscous"},
            "numerical_scheme": {"order": "second"},
            "fluid_properties": {},
            "reference_conditions": {
                "IC_1": {
                    "temperature": 300,
                    "pressure": 101325,
                    "v": {"vector": [1.0, 0.0, 0.0]},
                    "reynolds_no": 1.0e6,
                },
                "IC_2": condition,
            },
            "output_settings": {
                "solution": {"frequency": 10},
                "report": {"frequency": 10},
            },
            "time_settings": {"type": "steady"},
            "convergence_control": {
                "cycles": 100,
                "cfl": 1.0,
                "scheme": {"name": "runge kutta"},
            },
        },
        "model": {
            "base": {
                "mesh": "base.h5",
                "boundary_conditions": {
                    "bc1": {
                        "type": "farfield",
                        "condition": "IC_2",
                        "kind": "riemann",
                        "zones": [1],
                    }
                },
            }
        },
    }


class TestConditionKeys:
    def test_spaced_static_entry_is_accepted(self):
        static = {
            "temperature": 300,
            "pressure": 101325,
            "v": {"vector": [1.0, 0.0, 0.0]},
            "reynolds no": 1.0e6,
        }
        model, nodes = validate_zcfd_comprehensive(_viscous_deck_with_condition(static))
        assert model is not None, nodes


class TestExpandReferenceInheritance:
    def test_no_references_returned_unchanged(self):
        data = {
            "IC_1": {"temperature": 300, "pressure": 101325},
            "IC_2": {"temperature": 250, "pressure": 90000},
        }
        assert expand_reference_inheritance(data) == data

    def test_single_level_reference_inherits_and_overrides(self):
        data = {
            "initial": {"temperature": 300, "pressure": 101325},
            "inflow": {"reference": "initial", "velocity": 10},
        }
        result = expand_reference_inheritance(data)
        assert result["inflow"] == {
            "temperature": 300,
            "pressure": 101325,
            "velocity": 10,
        }
        # The base entry itself is untouched.
        assert result["initial"] == {"temperature": 300, "pressure": 101325}

    def test_reference_overrides_a_base_field(self):
        data = {
            "initial": {"temperature": 300, "pressure": 101325},
            "outflow": {"reference": "initial", "pressure": 90000},
        }
        result = expand_reference_inheritance(data)
        assert result["outflow"]["pressure"] == 90000
        assert result["outflow"]["temperature"] == 300

    def test_base_mutation_does_not_leak_into_inherited_entry(self):
        """Nested dict values are copied, not shared, between base and
        inheriting entries."""
        data = {
            "initial": {"temperature": 300, "profile": {"z0": 0.1}},
            "inflow": {"reference": "initial"},
        }
        result = expand_reference_inheritance(data)
        result["inflow"]["profile"]["z0"] = 999.0
        assert result["initial"]["profile"]["z0"] == 0.1

    def test_reference_to_undefined_base_rejected(self):
        data = {"inflow": {"reference": "does_not_exist", "velocity": 10}}
        with pytest.raises(ValueError, match="not defined or not yet expanded"):
            expand_reference_inheritance(data)

    def test_chained_reference_in_dependency_order_resolves(self):
        data = {
            "A": {"z": 3},
            "B": {"reference": "A", "y": 2},
            "C": {"reference": "B", "x": 1},
        }
        result = expand_reference_inheritance(data)
        assert result["C"] == {"z": 3, "y": 2, "x": 1}

    def test_chained_reference_declared_out_of_dependency_order_now_resolves(self):
        """expand_reference_inheritance previously used a single two-pass
        sweep — pass 1 pre-populates entries with NO
        'reference' key, pass 2 expands referencing entries in their
        original dict-iteration order — would resolve a chain longer than
        one hop (C -> B -> A) only when the dict happened to declare B
        before C. The fixed-point sweep repeatedly expands whichever pending
        items have their base already resolved, so the opposite (equally
        legitimate) declaration order below must resolve too.
        """
        data = {
            "C": {"reference": "B", "x": 1},
            "B": {"reference": "A", "y": 2},
            "A": {"z": 3},
        }
        result = expand_reference_inheritance(data)
        assert result["C"] == {"z": 3, "y": 2, "x": 1}
        assert result["B"] == {"z": 3, "y": 2}
        assert result["A"] == {"z": 3}

    def test_longer_reference_chain_declared_in_reverse_order_resolves(self):
        """A 4-hop chain (D -> C -> B -> A) declared in fully-reversed order
        still resolves via the fixed-point sweep."""
        data = {
            "D": {"reference": "C", "w": 4},
            "C": {"reference": "B", "x": 1},
            "B": {"reference": "A", "y": 2},
            "A": {"z": 3},
        }
        result = expand_reference_inheritance(data)
        assert result["D"] == {"z": 3, "y": 2, "x": 1, "w": 4}

    def test_circular_reference_chain_rejected(self):
        """A genuine cycle (A -> B -> A) never resolves and must raise,
        rather than looping forever."""
        data = {
            "A": {"reference": "B", "x": 1},
            "B": {"reference": "A", "y": 2},
        }
        with pytest.raises(ValueError, match="[Cc]ircular"):
            expand_reference_inheritance(data)
