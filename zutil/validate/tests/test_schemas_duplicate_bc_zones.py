"""
Duplicate BC zone-id check.

A zone id appearing in more than one boundary-condition block is an error,
enforced by a ModelSettings-level model validator.
"""

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.model_schemas import ModelSettings


def _base_kwargs(boundary_conditions):
    return dict(
        mesh="mesh.h5",
        boundary_conditions=boundary_conditions,
    )


class TestDuplicateBcZoneCheck:
    def test_no_duplicate_zones_passes(self):
        model = ModelSettings(
            **_base_kwargs(
                {
                    "BC_1": {"zones": [0, 1], "type": "symmetry"},
                    "BC_2": {"zones": [2, 3], "type": "symmetry"},
                }
            )
        )
        assert set(model.boundary_conditions.keys()) == {"BC_1", "BC_2"}

    def test_duplicate_zone_across_two_bcs_raises(self):
        with pytest.raises(ValidationError, match="Zone 2"):
            ModelSettings(
                **_base_kwargs(
                    {
                        "BC_1": {"zones": [0, 1, 2], "type": "symmetry"},
                        "BC_2": {"zones": [2, 3], "type": "symmetry"},
                    }
                )
            )

    def test_error_names_both_offending_bcs(self):
        with pytest.raises(ValidationError, match="BC_1.*BC_2|BC_2.*BC_1"):
            ModelSettings(
                **_base_kwargs(
                    {
                        "BC_1": {"zones": [5], "type": "symmetry"},
                        "BC_2": {"zones": [5], "type": "symmetry"},
                    }
                )
            )

    def test_single_bc_repeating_its_own_zone_is_fine(self):
        """A BC listing the same zone id is not a cross-BC duplicate."""
        model = ModelSettings(
            **_base_kwargs(
                {
                    "BC_1": {"zones": [0, 0], "type": "symmetry"},
                }
            )
        )
        assert model.boundary_conditions["BC_1"].zones == [0, 0]

    def test_legacy_zone_key_rejected(self):
        """The legacy singular `zone` spelling is not accepted at parse
        time; `extra="forbid"` on BoundaryConditionBase rejects it outright.
        """
        with pytest.raises(ValidationError):
            ModelSettings(
                **_base_kwargs(
                    {
                        "BC_1": {"zone": [0, 1], "type": "symmetry"},
                    }
                )
            )
