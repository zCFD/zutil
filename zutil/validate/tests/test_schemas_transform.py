"""
Direct accept/reject coverage for TransformsModel (transform_schemas.py):
'scale', 'transform' as a matrix or a placement function, and the order
statement every field description carries; and for
IDWTransformConfig.validate_idw_constraints.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from zutil.validate.schemas.transform_schemas import IDWTransformConfig, TransformsModel

ORDER = "Coordinates go through 'scale', then 'transform', then 'motion'."


class TestTransformsModelTransform:
    """'transform' is a 4x4 matrix, coerced to a numpy array, or a function
    def place(*, location) returning three numbers."""

    _IDENTITY_4X4 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    _SHAPE = (
        "'transform' must be a 4x4 matrix or a function def place(*, location), "
        "returning the placed position as three numbers; it has shape"
    )

    def test_none_accepted(self):
        model = TransformsModel()
        assert model.transform is None
        assert model.motion is None

    def test_valid_4x4_matrix_accepted_and_coerced_to_ndarray(self):
        model = TransformsModel(transform=self._IDENTITY_4X4)
        assert isinstance(model.transform, np.ndarray)
        assert model.transform.shape == (4, 4)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            TransformsModel(transform=[[1.0, 0.0], [0.0, 1.0]])
        assert f"{self._SHAPE} (2, 2)" in str(excinfo.value)

    def test_wrong_row_count_rejected(self):
        with pytest.raises(ValidationError, match="4x4"):
            TransformsModel(transform=self._IDENTITY_4X4[:3])

    def test_placement_function_accepted(self):
        def place(*, location):
            x, y, z = location
            return (x + 2.0, y, z)

        assert TransformsModel(transform=place).transform is place

    def test_placement_function_with_the_wrong_keywords_rejected(self):
        def place(*, position):
            return position

        with pytest.raises(ValidationError) as excinfo:
            TransformsModel(transform=place)
        assert (
            "The transform function place(*, position) cannot take the keywords "
            "zCFD calls it with: location. Write it as def place(*, location)."
        ) in str(excinfo.value)

    def test_placement_function_returning_the_wrong_shape_rejected(self):
        def place(*, location):
            return 1.0

        with pytest.raises(ValidationError) as excinfo:
            TransformsModel(transform=place)
        assert (
            "The transform function place returned 1.0 (float) when called with "
            "location=(0.0, 0.0, 0.0); it must return the placed position as "
            "three numbers."
        ) in str(excinfo.value)

    def test_transform_matrix_is_not_a_setting(self):
        with pytest.raises(ValidationError, match="transform_matrix"):
            TransformsModel(transform_matrix=self._IDENTITY_4X4)


class TestTransformsModelScale:
    def test_default_is_unit(self):
        assert TransformsModel().scale == [1.0, 1.0, 1.0]

    def test_three_positive_numbers_accepted(self):
        assert TransformsModel(scale=[0.0254] * 3).scale == [0.0254] * 3

    @pytest.mark.parametrize("scale", [[1.0, 0.0, 1.0], [1.0, -1.0, 1.0]])
    def test_non_positive_rejected(self, scale):
        with pytest.raises(ValidationError, match="greater than 0"):
            TransformsModel(scale=scale)

    @pytest.mark.parametrize("scale", [[1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    def test_three_factors_required(self, scale):
        with pytest.raises(ValidationError):
            TransformsModel(scale=scale)


@pytest.mark.parametrize("field", ["scale", "transform", "motion"])
def test_every_description_states_the_order(field):
    assert ORDER in TransformsModel.model_fields[field].description


class TestIDWTransformConfigValidateIdwConstraints:
    """IDWTransformConfig.validate_idw_constraints: support_radius and
    stencil_size are alternative ways of bounding the IDW stencil, so
    explicitly setting both is rejected as ambiguous.

    Leaving both unset is accepted, not the other XOR arm: the real IDW
    solver path reads only `n_nearest`, never
    `support_radius`/`stencil_size`, and real control decks
    (testcases/testcases/naca0012/n0012_449_a10p0_idw.py,
    testcases/testcases/shockplatefsi/shockplatefsi_idw.py) rely on exactly
    that: setting only `n_nearest` and neither of these two fields.
    `support_radius` also carries a required, non-None default inherited
    from MeshDeformationConfig (needed by the RBF sibling's own unguarded
    C++ read), so "explicitly set" is checked via `model_fields_set`
    rather than `is not None`.
    """

    def test_neither_set_accepted(self):
        cfg = IDWTransformConfig(n_nearest=350)
        assert cfg.support_radius is not None  # required default, not None
        assert cfg.stencil_size is None

    def test_stencil_size_only_accepted(self):
        cfg = IDWTransformConfig(stencil_size=20)
        assert cfg.stencil_size == 20

    def test_support_radius_only_accepted(self):
        cfg = IDWTransformConfig(support_radius=0.5)
        assert cfg.support_radius == 0.5
        assert cfg.stencil_size is None

    def test_both_explicitly_set_rejected(self):
        with pytest.raises(ValidationError, match="only one"):
            IDWTransformConfig(support_radius=0.5, stencil_size=20)
