"""A farfield's ``target``: what a deck may say, and the text it gets back."""

import pytest

from zutil.validate.schemas.boundary_conditions_schemas import (
    _TARGET_NEEDS_COMPRESSIBLE,
    _TARGET_NEEDS_STEADY,
)
from zutil.validate.validate_zcfd_schema import validate_zcfd_comprehensive

BC = "model.base.boundary_conditions.BC_3"


def _deck(**target):
    return {
        "config_version": 2,
        "solver": {
            "solver_settings": {"type": "compressible"},
            "initialisation": {"initial_conditions": "IC_1"},
            "equations": {"type": "euler"},
            "numerical_scheme": {"order": "second"},
            "fluid_properties": {},
            "reference_conditions": {
                "IC_1": {
                    "temperature": 300,
                    "pressure": 101325,
                    "v": {"vector": [1.0, 0.0, 0.1], "mach": 0.5},
                }
            },
            "output_settings": {
                "solution": {"frequency": 10},
                "report": {"frequency": 10, "forces": {"FR_1": {"zones": [1]}}},
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
                    "BC_3": {
                        "type": "farfield",
                        "condition": "IC_1",
                        "kind": "riemann",
                        "zones": [1],
                        "target": target or {"alpha": {"cl": 1.0778}},
                    }
                },
            }
        },
    }


def _bc(deck):
    return deck["model"]["base"]["boundary_conditions"]["BC_3"]


def _errors(deck):
    model, nodes = validate_zcfd_comprehensive(deck)
    errors = [(n["field_path"], n["display_message"]) for n in nodes]
    assert model is None, nodes
    return errors


def _valid(deck):
    model, nodes = validate_zcfd_comprehensive(deck)
    assert model is not None, nodes
    assert not [n for n in nodes if n["severity"] == "error"], nodes
    return model, [n for n in nodes if n["severity"] == "warning"]


class TestValidFragments:
    def test_the_contract_example_validates(self):
        deck = _deck(
            alpha={"cl": 1.0778, "hold": 500, "tolerance": 0.001, "of": "FR_1"},
            axes={"forward": "+x", "up": "+z"},
        )
        model, warnings = _valid(deck)
        assert warnings == []
        assert model.to_cpp_dict()["model"]["base"]["boundary_conditions"]["bc_3"][
            "target"
        ] == {
            "alpha": {"cl": 1.0778, "hold": 500, "tolerance": 0.001, "of": "FR_1"},
            "axes": {"forward": "+x", "up": "+z"},
        }

    def test_a_drag_target_validates(self):
        _valid(_deck(alpha={"cd": 0.02}))

    def test_the_axes_default_to_forward_x_and_up_z(self):
        model, _ = _valid(_deck())
        target = model.to_cpp_dict()["model"]["base"]["boundary_conditions"]["bc_3"][
            "target"
        ]
        assert target["axes"] == {"forward": "+x", "up": "+z"}

    def test_the_only_force_report_is_inferred(self):
        _valid(_deck(alpha={"cl": 0.5}))

    def test_of_names_one_of_several_reports(self):
        deck = _deck(alpha={"cl": 0.5, "of": "FR_2"})
        deck["solver"]["output_settings"]["report"]["forces"]["FR_2"] = {"zones": [1]}
        _valid(deck)

    def test_other_axes_validate(self):
        deck = _deck(alpha={"cl": 0.5}, axes={"forward": "-z", "up": "+x"})
        _valid(deck)


class TestRejections:
    def test_cl_and_cd_together(self):
        assert _errors(_deck(alpha={"cl": 0.5, "cd": 0.02})) == [
            (f"{BC}.target.alpha.cl", "Give either 'cl' or 'cd', not both")
        ]

    def test_no_quantity(self):
        assert _errors(_deck(alpha={"hold": 10})) == [
            (f"{BC}.target.alpha.cl", "Give either 'cl' or 'cd'")
        ]

    @pytest.mark.parametrize(
        "key, spelt",
        [
            ("cy", "cy"),
            ("lift", "lift"),
            ("drag", "drag"),
            ("side_force", "side force"),
            ("cm", "cm"),
            ("column", "column"),
        ],
    )
    def test_another_quantity_is_not_yet_supported(self, key, spelt):
        assert _errors(_deck(alpha={key: 0.1, "cl": 0.5})) == [
            (
                f"{BC}.target.alpha.{key}",
                f"'{spelt}' cannot be a target yet: a target trims 'cl' or 'cd', "
                'as "alpha": {"cl": 0.5}.',
            )
        ]

    @pytest.mark.parametrize("key", ["beta", "mach"])
    @pytest.mark.parametrize("with_alpha", [True, False])
    def test_beta_and_mach_are_not_yet_supported(self, key, with_alpha):
        deck = _deck(alpha={"cl": 0.5})
        if not with_alpha:
            del _bc(deck)["target"]["alpha"]
        _bc(deck)["target"][key] = {"cl": 0.5}
        assert _errors(deck) == [
            (
                f"{BC}.target.{key}",
                f"'{key}' cannot be trimmed yet: a target turns 'alpha' only, as "
                '"target": {"alpha": {"cl": 0.5}}.',
            )
        ]

    def test_a_target_needs_alpha(self):
        deck = _deck()
        _bc(deck)["target"] = {"axes": {"forward": "+x", "up": "+z"}}
        assert _errors(deck) == [
            (
                f"{BC}.target",
                'A target names the angle it turns, as "target": {"alpha": '
                '{"cl": 0.5}}.',
            )
        ]

    @pytest.mark.parametrize("hold", [0, -5])
    def test_hold_must_be_positive(self, hold):
        assert _errors(_deck(alpha={"cl": 0.5, "hold": hold})) == [
            (f"{BC}.target.alpha.hold", "Hold must be greater than 0.")
        ]

    def test_tolerance_must_be_positive(self):
        assert _errors(_deck(alpha={"cl": 0.5, "tolerance": 0.0})) == [
            (f"{BC}.target.alpha.tolerance", "Tolerance must be greater than 0.")
        ]

    def test_axes_must_be_at_right_angles(self):
        deck = _deck(alpha={"cl": 0.5}, axes={"forward": "+x", "up": "-x"})
        assert _errors(deck) == [
            (
                f"{BC}.target.axes.up",
                "'up' must be at right angles to 'forward': '+x' and '-x' both lie "
                "along x.",
            )
        ]

    def test_an_axis_must_be_signed(self):
        deck = _deck(alpha={"cl": 0.5}, axes={"up": "z"})
        assert [p for p, _ in _errors(deck)] == [f"{BC}.target.axes.up"]

    def test_of_must_name_a_force_report(self):
        assert _errors(_deck(alpha={"cl": 0.5, "of": "FR_9"})) == [
            (
                f"{BC}.target.alpha.of",
                "There is no force report 'FR_9'. The force reports are 'FR_1'.",
            )
        ]

    def test_of_is_needed_with_several_reports(self):
        deck = _deck()
        deck["solver"]["output_settings"]["report"]["forces"]["FR_2"] = {"zones": [1]}
        assert _errors(deck) == [
            (
                f"{BC}.target.alpha.of",
                "The force reports are 'FR_1' and 'FR_2': name the one the "
                'coefficient is read from with \'of\', as "of": "FR_1".',
            )
        ]

    def test_a_target_needs_a_force_report(self):
        deck = _deck()
        del deck["solver"]["output_settings"]["report"]["forces"]
        assert _errors(deck) == [
            (
                f"{BC}.target",
                "A target reads its coefficient from a force report, and none is "
                "defined: add one under output settings › report › forces, naming "
                "the zones of the body.",
            )
        ]

    def test_a_mesh_report_adds_to_the_solver_ones(self):
        deck = _deck()
        deck["model"]["base"]["output_settings"] = {
            "report": {"forces": {"FR_2": {"zones": [1]}}},
        }
        assert _errors(deck) == [
            (
                f"{BC}.target.alpha.of",
                "The force reports are 'FR_1' and 'FR_2': name the one the "
                'coefficient is read from with \'of\', as "of": "FR_1".',
            )
        ]
        _bc(deck)["target"]["alpha"]["of"] = "FR_2"
        _valid(deck)

    @pytest.mark.parametrize(
        "of, path", [(None, "target"), ("FR_1", "target.alpha.of")]
    )
    def test_a_dimensional_report_is_refused(self, of, path):
        deck = _deck(alpha={"cl": 0.5, "of": of})
        deck["solver"]["output_settings"]["report"]["forces"]["FR_1"]["dimensional"] = (
            True
        )
        assert _errors(deck) == [
            (
                f"{BC}.{path}",
                "Force report 'FR_1' writes dimensional forces, and a target trims "
                "a coefficient. Set \"dimensional\": False on 'FR_1', or name a "
                "coefficient report with 'of'.",
            )
        ]

    def test_the_condition_needs_flow_in_the_forward_up_plane(self):
        deck = _deck()
        _bc(deck)["target"]["axes"] = {"forward": "+z", "up": "+y"}
        deck["solver"]["reference_conditions"]["IC_1"]["v"] = {"vector": [1, 0, 0]}
        assert _errors(deck) == [
            (
                f"{BC}.condition",
                "Reference condition 'IC_1' has velocity [1, 0, 0], which has no "
                "component along the target's forward axis (+z) or up axis (+y), so "
                "there is no angle of attack to trim from. Check the target's "
                "'axes', or give 'IC_1' a 'v' vector in the forward–up plane.",
            )
        ]


ADVANCED = {
    "max_step": 2.0,
    "relaxation": 1.0,
    "average_over": 10,
    "max_wait": 1500,
    "slope": 0.11,
}


def _target(model):
    return model.to_cpp_dict()["model"]["base"]["boundary_conditions"]["bc_3"]["target"]


class TestAdvanced:
    def test_the_full_block_passes_through(self):
        model, warnings = _valid(
            _deck(alpha={"cl": 1.0778, "hold": 500, "advanced": ADVANCED})
        )
        assert warnings == []
        assert _target(model)["alpha"]["advanced"] == ADVANCED

    def test_a_partial_block_passes_through_as_written(self):
        model, _ = _valid(_deck(alpha={"cd": 0.02, "advanced": {"max_step": 1.0}}))
        assert _target(model)["alpha"]["advanced"] == {"max_step": 1.0}

    def test_an_unset_block_is_omitted(self):
        model, _ = _valid(_deck())
        assert "advanced" not in _target(model)["alpha"]

    @pytest.mark.parametrize(
        "key, value, message",
        [
            ("max_step", 0.0, "Max step must be greater than 0."),
            ("max_step", 7.0, "Max step must be less than or equal to 6."),
            ("relaxation", 0.0, "Relaxation must be greater than 0."),
            ("relaxation", 2.5, "Relaxation must be less than or equal to 2."),
            ("average_over", 0, "Average over must be greater than or equal to 1."),
            ("max_wait", 0, "Max wait must be greater than 0."),
        ],
    )
    def test_each_bound(self, key, value, message):
        deck = _deck(alpha={"cl": 0.5, "advanced": {key: value}})
        assert _errors(deck) == [(f"{BC}.target.alpha.advanced.{key}", message)]

    def test_an_unknown_key_is_rejected(self):
        deck = _deck(alpha={"cl": 0.5, "advanced": {"gain": 1.0}})
        assert [p for p, _ in _errors(deck)] == [f"{BC}.target.alpha.advanced.gain"]


class TestNotSteadyCompressible:
    def test_dual_time_stepping_warns_and_validates(self):
        deck = _deck()
        deck["solver"]["time_settings"] = {
            "type": "unsteady",
            "kind": "dual time stepping",
            "total_time": 1.0,
            "time_step": 0.1,
        }
        _, warnings = _valid(deck)
        assert [(n["field_path"], n["display_message"]) for n in warnings] == [
            (
                f"{BC}.target",
                "Farfield 'BC_3' has a 'target', but this run is unsteady: a target "
                "is trimmed only in a steady compressible run, so it is ignored and "
                "the run keeps the condition's own angle.",
            )
        ]

    @pytest.mark.parametrize(
        "solver, time, unsteady, incompressible",
        [
            ("compressible", "steady", False, False),
            ("incompressible", "steady", False, True),
            ("compressible", "unsteady", True, False),
        ],
    )
    def test_the_clauses(self, solver, time, unsteady, incompressible):
        deck = {
            "solver": {
                "solver_settings": {"type": solver},
                "time_settings": {"type": time},
            }
        }
        assert _TARGET_NEEDS_STEADY.when.evaluate(deck) is unsteady
        assert _TARGET_NEEDS_COMPRESSIBLE.when.evaluate(deck) is incompressible


class TestRemovedKeysPointAtTarget:
    def test_driver_on_a_farfield(self):
        deck = _deck()
        del _bc(deck)["target"]
        _bc(deck)["driver"] = {"target": "wall_Ftz", "target_value": 1.0}
        assert _errors(deck) == [
            (
                f"{BC}.driver",
                "Farfield 'BC_3' sets 'driver', which is not a setting of a "
                "farfield: a farfield is trimmed to a lift or drag coefficient by "
                'its \'target\', as "target": {"alpha": {"cl": 0.5}}.',
            )
        ]

    def test_a_function_as_a_reference_condition(self):
        def drive(**kwargs):
            return {"temperature": 300.0, "pressure": 101325.0}

        deck = _deck()
        deck["solver"]["reference_conditions"]["IC_2"] = drive
        _bc(deck)["condition"] = "IC_2"
        errors = _errors(deck)
        assert (
            "solver.reference_conditions.IC_2",
            "Reference condition 'IC_2' is a function. A reference condition is a "
            "fixed state. To trim a farfield's angle of attack to a lift or drag "
            "coefficient, give the farfield a 'target', as "
            '"target": {"alpha": {"cl": 0.5}}.',
        ) in errors
