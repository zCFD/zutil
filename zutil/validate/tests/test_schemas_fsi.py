"""
ModalModelBase.harmonic_initial_state requires location/force/frequency, and
GenericFSIModel carries a strict field set rather than extra="allow".
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from zutil.validate.schemas.fsi_schemas import (
    FSIModel,
    GenericFSIModel,
    HarmonicInitialStateModel,
    RBFModalModel,
)


def _force(*args, **kwargs):
    return [1.0, 0.0, 0.0]


class TestHarmonicInitialStateModel:
    def test_requires_location_force_frequency(self):
        with pytest.raises(ValidationError):
            HarmonicInitialStateModel()

    def test_vector_force_passes(self):
        state = HarmonicInitialStateModel(
            location=[0.0, 0.0, 0.0], force=[1.0, 0.0, 0.0], frequency=10.0
        )
        assert state.frequency == 10.0

    def test_callable_force_rejected(self):
        with pytest.raises(ValidationError):
            HarmonicInitialStateModel(
                location=[0.0, 0.0, 0.0], force=_force, frequency=10.0
            )

    def test_initial_state_wired_into_modal_model(self):
        model = RBFModalModel(
            file_format="nastran",
            nastran_casename="case1",
            mode_mapping_max_distance=1.0,
            fluid_force_scaling=1.0,
            mode_list=[1, 2, 3],
            modal_damping=[0.01, 0.01, 0.01],
            base_point_fraction=0.5,
            harmonic_initial_state={
                "location": [0.0, 0.0, 0.0],
                "force": [1.0, 0.0, 0.0],
                "frequency": 5.0,
            },
        )
        assert isinstance(model.harmonic_initial_state, HarmonicInitialStateModel)

    def test_initial_state_still_optional(self):
        model = RBFModalModel(
            file_format="nastran",
            nastran_casename="case1",
            mode_mapping_max_distance=1.0,
            fluid_force_scaling=1.0,
            mode_list=[1, 2, 3],
            modal_damping=[0.01, 0.01, 0.01],
            base_point_fraction=0.5,
        )
        assert model.harmonic_initial_state is None

    def test_malformed_initial_state_rejected(self):
        with pytest.raises(ValidationError):
            RBFModalModel(
                file_format="nastran",
                nastran_casename="case1",
                mode_mapping_max_distance=1.0,
                fluid_force_scaling=1.0,
                mode_list=[1, 2, 3],
                modal_damping=[0.01, 0.01, 0.01],
                base_point_fraction=0.5,
                harmonic_initial_state={
                    "location": [0.0, 0.0, 0.0]
                },  # missing force/frequency
            )


class TestGenericFSIModel:
    def test_minimal_required_fields(self):
        model = GenericFSIModel(zone=[0], transform_type="idw")
        assert model.type == "generic_fsi"
        assert model.support_radius == 1.0

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            GenericFSIModel(zone=[0], transform_type="idw", made_up_field=1)

    def test_missing_required_transform_type_rejected(self):
        with pytest.raises(ValidationError):
            GenericFSIModel(zone=[0])

    def test_user_variables_open_ended_dict_accepted(self):
        model = GenericFSIModel(
            zone=[0],
            transform_type="rbf",
            user_variables={"anything": 1, "goes": [1, 2, 3]},
        )
        assert model.user_variables == {"anything": 1, "goes": [1, 2, 3]}

    def test_resolves_via_fsi_model_union(self):
        ta = TypeAdapter(FSIModel)
        model = ta.validate_python(
            {"type": "generic_fsi", "zone": [0], "transform_type": "idw"}
        )
        assert isinstance(model, GenericFSIModel)
