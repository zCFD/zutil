"""Accept/reject coverage for CFLModel and the CFL shorthand."""

import warnings

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.convergence_schemas import (
    CFLModel,
    CompressibleConvergenceControl,
    build_cfl_ramp,
)


def _ramp(*, cycle, current_cfl):
    return current_cfl


def ramp_of(model):
    """The callable solver.py builds, from the same dict it is handed."""
    return build_cfl_ramp(model.model_dump(exclude_none=True).get("cfl_ramp"))


class TestCFLModelTurbulenceKey:
    """'cfl_turbulence' is the only accepted spelling."""

    def test_canonical_key_accepted(self):
        model = CFLModel(cfl=1.0, cfl_turbulence=0.7)
        assert model.cfl_turbulence == 0.7

    def test_pre_rename_key_rejected(self):
        with pytest.raises(ValidationError):
            CFLModel(cfl=1.0, cfl_transport=0.5)


class TestCFLModelInitialLivesInTheRamp:
    """`initial` is where the ramp starts, so the ramp owns it.

    The ramp returns `initial` at its own start cycle; there is no separate
    `cfl_initial` field to keep in step.
    """

    def test_neither_set_accepted(self):
        model = CFLModel(cfl=1.0)
        assert model.cfl_ramp is None

    def test_there_is_no_field_beside_the_ramp(self):
        assert "cfl_initial" not in CFLModel.model_fields

    def test_a_flat_initial_is_rejected(self):
        with pytest.raises(ValidationError, match="Unexpected|extra"):
            CFLModel(cfl=1.0, cfl_ramp=_ramp, cfl_initial=0.1)

    def test_a_callable_ramp_needs_no_initial(self):
        """A callable is handed the current CFL and may return anything, so
        it sets its own starting point (see the x15 testcase)."""
        model = CFLModel(cfl=1.0, cfl_ramp=_ramp)
        assert callable(model.cfl_ramp)


class TestCFLModelValidateCflRamp:
    """CFLModel.validate_cfl_ramp: accepts a callable with a compatible
    (cycle, current_cfl) signature or a recognised ramp-config dict."""

    def test_none_accepted(self):
        model = CFLModel(cfl=1.0, cfl_ramp=None)
        assert model.cfl_ramp is None

    def test_compatible_callable_accepted(self):
        model = CFLModel(cfl=1.0, cfl_ramp=_ramp)
        assert model.cfl_ramp is _ramp

    def test_callable_with_the_pre_v2_arguments_rejected(self):
        def bad_ramp(cycle, real_time_cycle, cfl):
            return cfl

        with pytest.raises(ValidationError, match="cannot take the keywords"):
            CFLModel(cfl=1.0, cfl_ramp=bad_ramp)

    def test_callable_with_too_few_args_rejected(self):
        def bad_ramp(cycle):
            return cycle

        with pytest.raises(ValidationError, match="cannot take the keywords"):
            CFLModel(cfl=1.0, cfl_ramp=bad_ramp)

    def test_kwargs_callable_accepted(self):
        def kwargs_ramp(**kwargs):
            return kwargs["current_cfl"]

        model = CFLModel(cfl=1.0, cfl_ramp=kwargs_ramp)
        assert model.cfl_ramp is kwargs_ramp

    def test_positional_only_varargs_callable_rejected(self):
        def varargs_ramp(*args):
            return args[-1]

        with pytest.raises(ValidationError, match="def varargs_ramp"):
            CFLModel(cfl=1.0, cfl_ramp=varargs_ramp)

    def test_linear_ramp_accepted(self):
        model = CFLModel(cfl=1.0, cfl_ramp=[{"type": "linear", "increment": 0.5}])
        assert callable(ramp_of(model))

    def test_stepped_linear_ramp_accepted(self):
        model = CFLModel(
            cfl=1.0,
            cfl_ramp=[{"type": "stepped_linear", "increment": 1.0, "step_cycles": 5}],
        )
        assert callable(ramp_of(model))

    def test_a_shape_without_its_own_parameter_is_rejected(self):
        with pytest.raises(ValidationError, match="increment"):
            CFLModel(cfl=1.0, cfl_ramp=[{"type": "linear"}])

    def test_exponential_ramp_accepted(self):
        model = CFLModel(cfl=1.0, cfl_ramp=[{"type": "exponential", "factor": 1.1}])
        assert callable(ramp_of(model))

    def test_several_ramps_run_in_order(self):
        model = CFLModel(
            cfl=100.0,
            cfl_ramp=[
                {"type": "exponential", "factor": 2.0, "initial": 1.0, "end_cycle": 4},
                {"type": "linear", "increment": 10.0, "start_cycle": 5},
            ],
        )
        ramp, cur, trace = ramp_of(model), model.cfl, []
        for cycle in range(1, 8):
            cur = min(ramp(cycle=cycle, current_cfl=cur), model.cfl)
            trace.append(cur)
        # doubling inside its window, then +10 a cycle once the second takes over
        assert trace == [1.0, 2.0, 4.0, 8.0, 18.0, 28.0, 38.0]

    def test_an_empty_list_is_no_ramp(self):
        assert ramp_of(CFLModel(cfl=1.0, cfl_ramp=[])) is None

    def test_a_bare_dict_is_rejected(self):
        """One ramp is a list of one. There is no second spelling."""
        with pytest.raises(ValidationError, match="must be a list of ramp"):
            CFLModel(cfl=1.0, cfl_ramp={"type": "exponential", "factor": 1.1})

    def test_there_is_no_sequence_shape(self):
        with pytest.raises(ValidationError, match="expected tags"):
            CFLModel(cfl=1.0, cfl_ramp=[{"type": "sequence", "ramps": []}])

    def test_an_unnamed_shape_is_rejected(self):
        """`growth` implied an exponential without naming one. There is no
        special case for it — it is simply not a shape."""
        with pytest.raises(ValidationError, match="Unable to extract tag"):
            CFLModel(cfl=1.0, cfl_ramp=[{"growth": 1.05, "initial": 0.1}])

    def test_a_non_dict_entry_is_rejected(self):
        with pytest.raises(ValidationError, match="valid dictionary or object"):
            CFLModel(cfl=1.0, cfl_ramp=["exponential"])

    def test_a_bad_entry_names_its_own_index(self):
        """The UI anchors messages on the dotted path."""
        with pytest.raises(ValidationError) as exc:
            CFLModel(
                cfl=1.0,
                cfl_ramp=[
                    {"type": "exponential", "factor": 1.1},
                    {"type": "exponential", "factor": 0.5},
                ],
            )
        locs = [e["loc"] for e in exc.value.errors()]
        assert any(1 in loc and "factor" in loc for loc in locs), locs

    def test_a_callable_is_not_offered_as_an_alternative(self):
        """As a union member it would report "input should be callable"
        beside every real mistake in the list."""
        with pytest.raises(ValidationError) as exc:
            CFLModel(cfl=1.0, cfl_ramp=[{"type": "not_a_real_ramp"}])
        assert not any(e["type"] == "callable_type" for e in exc.value.errors())

    def test_the_form_sees_the_list_and_not_the_callable(self):
        """The list is described; the callable stays out of the schema."""
        prop = CFLModel.model_json_schema()["properties"]["cfl_ramp"]
        array = next(m for m in prop["anyOf"] if m.get("type") == "array")
        assert set(array["items"]["discriminator"]["mapping"]) == {
            "exponential",
            "linear",
            "stepped_linear",
        }
        assert not any("callable" in str(m).lower() for m in prop["anyOf"])

    def test_the_ramp_starts_at_its_own_initial(self):
        model = CFLModel(
            cfl=100.0,
            cfl_ramp=[{"type": "exponential", "factor": 1.1, "initial": 0.05}],
        )
        # CFL() starts current_cfl at max_cfl; the ramp pulls it down.
        assert ramp_of(model)(cycle=1, current_cfl=100.0) == 0.05

    def test_initial_is_not_offered_as_a_form_field(self):
        assert "cfl_initial" not in CFLModel.model_json_schema()["properties"]

    def test_a_migrated_ramp_behaves_exactly_as_the_legacy_one_did(self):
        """Guards the corpus migration: same trace, cycle for cycle."""
        model = CFLModel(
            cfl=100.0,
            cfl_ramp=[{"type": "exponential", "factor": 1.1, "initial": 0.05}],
        )
        ramp, cur, trace = ramp_of(model), model.cfl, []
        for cycle in range(1, 9):
            cur = min(ramp(cycle=cycle, current_cfl=cur), model.cfl)
            trace.append(round(cur, 9))
        assert trace == [
            0.05,
            0.055,
            0.0605,
            0.06655,
            0.073205,
            0.0805255,
            0.08857805,
            0.097435855,
        ]

    @pytest.mark.parametrize(
        "ramp_type,extra",
        [
            ("linear", {"increment": 2.0}),
            ("stepped_linear", {"increment": 2.0, "step_cycles": 5}),
            ("exponential", {"factor": 1.1}),
        ],
    )
    def test_every_shape_honours_initial_at_its_start_cycle(self, ramp_type, extra):
        model = CFLModel(
            cfl=100.0,
            cfl_ramp=[{"type": ramp_type, "initial": 0.5, "start_cycle": 10, **extra}],
        )
        ramp = ramp_of(model)
        assert ramp(cycle=9, current_cfl=100.0) == 100.0  # before the window, untouched
        assert ramp(cycle=10, current_cfl=100.0) == 0.5  # at the start, the initial

    def test_unknown_ramp_type_rejected(self):
        with pytest.raises(ValidationError, match="expected tags"):
            CFLModel(cfl=1.0, cfl_ramp=[{"type": "not_a_real_ramp"}])

    def test_neither_a_list_nor_a_callable_rejected(self):
        with pytest.raises(ValidationError, match="must be a list of ramp"):
            CFLModel(cfl=1.0, cfl_ramp=42)


class TestCompressibleConvergenceControlValidateCfl:
    """A scalar `cfl` is shorthand for `{"cfl": <value>}` and inherits
    CFLModel.cfl's own gt=0 constraint."""

    def test_positive_float_accepted(self):
        cc = CompressibleConvergenceControl(cycles=100, cfl=2.0)
        assert cc.cfl.cfl == 2.0

    def test_zero_float_rejected(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            CompressibleConvergenceControl(cycles=100, cfl=0.0)

    def test_negative_float_rejected(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            CompressibleConvergenceControl(cycles=100, cfl=-1.0)

    def test_cfl_model_dict_still_accepted(self):
        cc = CompressibleConvergenceControl(cycles=100, cfl={"cfl": 1.5})
        assert isinstance(cc.cfl, CFLModel)
        assert cc.cfl.cfl == 1.5


class TestCompressibleConvergenceControlCflShape:
    """`cfl` is a simple number or an advanced dict — and nothing else.

    CFL settings belong inside `cfl`. Written beside it they are rejected by
    extra="forbid" rather than folded in, so there is one shape to learn.
    """

    def test_advanced_dict_holds_every_setting(self):
        cc = CompressibleConvergenceControl.model_validate(
            {
                "cycles": 100,
                "cfl": {
                    "cfl": 2.0,
                    "cfl_ramp": [
                        {
                            "type": "exponential",
                            "factor": 1.05,
                            "initial": 1.0,
                        }
                    ],
                    "cfl_turbulence": 0.5,
                    "cfl_viscous_factor": 0.8,
                },
            }
        )
        assert isinstance(cc.cfl, CFLModel)
        assert cc.cfl.cfl == 2.0
        assert ramp_of(cc.cfl)(cycle=1, current_cfl=100.0) == 1.0
        assert cc.cfl.cfl_turbulence == 0.5
        assert cc.cfl.cfl_viscous_factor == 0.8

    @pytest.mark.parametrize(
        "sibling",
        [
            "cfl_ramp",
            "cfl_turbulence",
            "cfl_coarse",
            "cfl_initial",
            "cfl_viscous_factor",
        ],
    )
    def test_setting_written_beside_cfl_is_rejected(self, sibling):
        with pytest.raises(ValidationError, match="[Ee]xtra"):
            CompressibleConvergenceControl.model_validate(
                {"cycles": 100, "cfl": 2.0, sibling: 1.0}
            )

    def test_no_siblings_leaves_cfl_as_the_bare_shorthand(self):
        cc = CompressibleConvergenceControl.model_validate({"cycles": 100, "cfl": 2.0})
        assert cc.cfl.cfl == 2.0
        assert cc.cfl.cfl_ramp is None
        assert cc.cfl.cfl_turbulence is None


class TestCallableRampSerialises:
    """A callable ramp has to survive model_dump, silently.

    The annotation describes only the list form, so without CFLModel's own
    field serializer pydantic treats the callable escape hatch as a type
    mismatch and warns on every dump — including the dump inside the
    applicability validator, which made merely validating a deck like the
    x15 testcase emit PydanticSerializationUnexpectedValue.
    """

    def test_callable_is_dumped_unchanged(self):
        dumped = CFLModel(cfl=1.0, cfl_ramp=_ramp).model_dump(exclude_none=True)
        assert dumped["cfl_ramp"] is _ramp

    def test_dumping_a_callable_warns_about_nothing(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CFLModel(cfl=1.0, cfl_ramp=_ramp).model_dump(exclude_none=True)
        assert [str(w.message) for w in caught] == []

    def test_the_list_form_still_dumps_to_plain_dicts(self):
        model = CFLModel(cfl=1.0, cfl_ramp=[{"type": "linear", "increment": 0.5}])
        dumped = model.model_dump(exclude_none=True)
        assert dumped["cfl_ramp"] == [
            {"start_cycle": 1, "type": "linear", "increment": 0.5}
        ]
