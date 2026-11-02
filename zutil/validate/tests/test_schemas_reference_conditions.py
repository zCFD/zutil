"""
Direct accept/reject coverage for
ViscousReferenceConditions.validate_reynolds_viscosity_mutual_exclusivity
(reference_conditions_schemas.py). ABLProfile's own validators are covered
separately in test_schemas_abl_profile.py.
"""

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.reference_conditions_schemas import (
    ViscousReferenceConditions,
)
from zutil.validate.tests.test_core_orchestration import _minimal_valid_config
from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _base(**overrides):
    data = {"temperature": 300.0, "pressure": 101325.0}
    data.update(overrides)
    return data


class TestViscousReferenceConditionsMutualExclusivity:
    def test_reynolds_no_only_accepted(self):
        model = ViscousReferenceConditions(**_base(reynolds_no=1.0e6))
        assert model.reynolds_no == 1.0e6
        assert model.viscosity is None

    def test_viscosity_only_accepted(self):
        model = ViscousReferenceConditions(**_base(viscosity=1.8e-5))
        assert model.viscosity == 1.8e-5
        assert model.reynolds_no is None

    def test_both_set_rejected(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            ViscousReferenceConditions(**_base(reynolds_no=1.0e6, viscosity=1.8e-5))

    def test_neither_set_rejected(self):
        with pytest.raises(ValidationError, match="Must specify either"):
            ViscousReferenceConditions(**_base())

    def test_model_validate_on_instance_now_enforces_the_check(self):
        """validate_reynolds_viscosity_mutual_exclusivity is a
        model_validator(mode="wrap"). Inspecting `data` only when
        `isinstance(data, dict)` would let an already-constructed instance
        (via model_construct(), or any caller passing an instance to
        model_validate()) skip the mutual-exclusivity check, so it must also
        read the fields off an `isinstance(data, cls)` instance.
        """
        instance = ViscousReferenceConditions.model_construct(
            **_base(reynolds_no=1.0e6, viscosity=1.8e-5)
        )
        # Constructing via model_construct() also bypasses validation, so
        # both fields really are set on the unvalidated instance.
        assert instance.reynolds_no == 1.0e6
        assert instance.viscosity == 1.8e-5

        with pytest.raises(ValidationError, match="mutually exclusive"):
            ViscousReferenceConditions.model_validate(instance)

    def test_model_validate_on_valid_instance_still_accepted(self):
        """A valid (single-field) instance passed to model_validate() must
        still pass through unrejected — the fix only closes the bypass for
        the invalid combination, it doesn't reject valid instances."""
        instance = ViscousReferenceConditions(**_base(reynolds_no=1.0e6))
        revalidated = ViscousReferenceConditions.model_validate(instance)
        assert revalidated.reynolds_no == 1.0e6
        assert revalidated.viscosity is None


class TestVelocityShapeCanonicalisation:
    """The solver reads reference-condition velocity through
    getDict("v").getList("vector") -- one shape, unconditionally
    (src/solver/physics/Reference.cxx, src/solver/incomp/Initialiser.cxx).
    The schema deliberately accepts BOTH a bare [Vx, Vy, Vz] and the full
    {"vector": ..., "mach": ...} form, so _wrap_bare_velocity_vector must
    canonicalise the bare list at validation time. Before it existed, a deck
    using the bare-list branch of the Union validated cleanly and then died
    at solver initialisation with "Parameters lookup failed" -- the
    atmosphere decks were the first to hit this because nothing else in the
    repo uses the bare-list form.
    """

    def _dump(self, model):
        # The exact dump the resolved (C++-facing) dict is built with
        # (zcfd_schema.py).
        return model.model_dump(exclude_none=True, mode="python")

    def test_bare_list_is_wrapped_to_the_vector_dict(self):
        model = ViscousReferenceConditions(
            **_base(reynolds_no=1.0e6, v=[0.0, 10.0, 0.0])
        )
        assert self._dump(model)["v"] == {"vector": [0.0, 10.0, 0.0]}

    def test_bare_tuple_is_wrapped_too(self):
        model = ViscousReferenceConditions(
            **_base(reynolds_no=1.0e6, v=(1.0, 2.0, 3.0))
        )
        assert self._dump(model)["v"] == {"vector": [1.0, 2.0, 3.0]}

    def test_vector_dict_form_is_preserved(self):
        model = ViscousReferenceConditions(
            **_base(reynolds_no=1.0e6, v={"vector": [1.0, 0.0, 0.0], "mach": 0.2})
        )
        assert self._dump(model)["v"] == {"vector": [1.0, 0.0, 0.0], "mach": 0.2}

    def test_omitted_v_defaults_to_the_vector_dict_shape(self):
        """The default must ALSO be dict-shaped: pydantic does not run
        before-validators on defaults, so a bare-list default would put the
        unreadable shape back into the resolved dict for any deck that
        omits v."""
        model = ViscousReferenceConditions(**_base(reynolds_no=1.0e6))
        assert self._dump(model)["v"] == {"vector": [0.0, 0.0, 0.0]}

    def test_mach_only_dict_without_vector_is_rejected(self):
        with pytest.raises(ValidationError):
            ViscousReferenceConditions(**_base(reynolds_no=1.0e6, v={"mach": 0.2}))


def _abl_deck(**abl):
    deck = _minimal_valid_config()
    deck["solver"]["reference_conditions"]["IC_1"]["profile"] = {
        "abl": {"roughness_length": 0.1, **abl}
    }
    return deck


class TestABLProfileNeedsFrictionVelocityOrSurfaceLayerHeight:
    """The compressible initialiser aborts on an ABL profile that gives
    neither 'friction velocity' nor 'surface layer height'."""

    def test_compressible_with_neither_rejected(self):
        with pytest.raises(ValidationError) as info:
            ZCFDInputModelBase.model_validate(_abl_deck())
        assert (
            "Set 'friction velocity' or 'surface layer height'; one is required "
            "when solver.solver settings.type is 'compressible'."
        ) in str(info.value)

    @pytest.mark.parametrize(
        "abl",
        [
            {"friction_velocity": 0.4},
            {"surface_layer_height": 50.0},
            {"friction_velocity": 0.4, "surface_layer_height": 50.0},
        ],
    )
    def test_compressible_with_either_accepted(self, abl):
        ZCFDInputModelBase.model_validate(_abl_deck(**abl))
