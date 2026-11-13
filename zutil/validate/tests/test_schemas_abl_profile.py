"""
ABLProfile coverage.

ABLProfile (reference_conditions_schemas.py) must expose every key the C++
consumers read: roughness_length,
friction_velocity, ground_level, surface_layer_height, monin_obukhov_length,
tke and up. Anything missing hard-fails a deck against extra="forbid".
"""

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.reference_conditions_schemas import ABLProfile


def _minimal(**overrides):
    data = {
        "roughness_length": 0.1,
        "friction_velocity": 0.4,
        "ground_level": 0.0,
    }
    data.update(overrides)
    return data


class TestNewFieldsAccepted:
    def test_all_new_fields_accepted(self):
        profile = ABLProfile(
            roughness_length=0.1,
            friction_velocity=0.4,
            ground_level=1.5,
            surface_layer_height=500.0,
            monin_obukhov_length=-200.0,
            tke=0.5,
            up=[0.0, 0.0, 1.0],
        )
        assert profile.surface_layer_height == 500.0
        assert profile.monin_obukhov_length == -200.0
        assert profile.tke == 0.5
        assert list(profile.up) == [0.0, 0.0, 1.0]

    def test_new_fields_optional(self):
        profile = ABLProfile(**_minimal())
        assert profile.surface_layer_height is None
        assert profile.monin_obukhov_length is None
        assert profile.tke is None
        assert profile.up is None

    def test_dumped_keys_are_snake_case(self):
        profile = ABLProfile(
            roughness_length=0.1,
            friction_velocity=0.4,
            ground_level=0.0,
            surface_layer_height=500.0,
            monin_obukhov_length=-200.0,
            tke=0.5,
            up=[0.0, 0.0, 1.0],
        )
        dumped = profile.model_dump(exclude_none=True)
        assert set(dumped.keys()) == {
            "roughness_length",
            "friction_velocity",
            "ground_level",
            "surface_layer_height",
            "monin_obukhov_length",
            "tke",
            "up",
        }

    def test_surface_layer_height_must_be_positive(self):
        with pytest.raises(ValidationError):
            ABLProfile(**_minimal(surface_layer_height=-1.0))

    def test_tke_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            ABLProfile(**_minimal(friction_velocity=0.4, tke=-1.0))


class TestGroundLevelRename:
    def test_ground_level_round_trips(self):
        profile = ABLProfile(**_minimal(ground_level=12.5))
        assert profile.ground_level == 12.5
        dumped = profile.model_dump(exclude_none=True)
        assert dumped["ground_level"] == 12.5
        assert "z0" not in dumped

    def test_legacy_z0_key_accepted_and_normalised(self):
        profile = ABLProfile(roughness_length=0.1, friction_velocity=0.4, z0=3.0)
        assert profile.ground_level == 3.0
        assert "z0" not in profile.model_dump(exclude_none=True)

    def test_both_z0_and_ground_level_rejected(self):
        with pytest.raises(ValidationError):
            ABLProfile(
                roughness_length=0.1,
                friction_velocity=0.4,
                z0=3.0,
                ground_level=3.0,
            )

    def test_z0_no_longer_a_declared_field(self):
        # Confirms the rename went all the way: 'z0' is accepted only as an
        # input alias (via the before-validator), not as a real field name.
        assert "z0" not in ABLProfile.model_fields
        assert "ground_level" in ABLProfile.model_fields


class TestMoninObukhovLength:
    def test_legacy_raw_key_accepted(self):
        profile = ABLProfile(
            **_minimal(),
            **{"Monin-Obukhov length": -150.0},
        )
        assert profile.monin_obukhov_length == -150.0

    def test_post_normalisation_key_accepted(self):
        # What replace_spaces_in_keys() actually produces from the raw v1
        # spelling before pydantic sees it (spaces -> underscore, lowercased,
        # hyphen untouched).
        profile = ABLProfile(
            **_minimal(),
            **{"monin-obukhov_length": -150.0},
        )
        assert profile.monin_obukhov_length == -150.0

    def test_zero_rejected(self):
        with pytest.raises(ValidationError):
            ABLProfile(**_minimal(monin_obukhov_length=0.0))

    def test_negative_and_positive_accepted(self):
        assert (
            ABLProfile(**_minimal(monin_obukhov_length=-1.0)).monin_obukhov_length
            == -1.0
        )
        assert (
            ABLProfile(**_minimal(monin_obukhov_length=1.0)).monin_obukhov_length == 1.0
        )

    def test_both_legacy_and_canonical_rejected(self):
        with pytest.raises(ValidationError):
            ABLProfile(
                **_minimal(),
                **{"Monin-Obukhov length": -150.0, "monin_obukhov_length": -150.0},
            )


class TestTkeWithoutFrictionVelocity:
    def test_tke_without_friction_velocity_warns(self):
        with pytest.warns(UserWarning, match="friction_velocity"):
            profile = ABLProfile(
                roughness_length=0.1,
                surface_layer_height=100.0,
                ground_level=0.0,
                tke=0.5,
            )
        assert profile.tke == 0.5

    def test_tke_with_friction_velocity_no_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ABLProfile(**_minimal(friction_velocity=0.4, tke=0.5))

    def test_friction_velocity_now_optional(self):
        # friction_velocity is optional: the C++ accepts either it or
        # surface_layer_height.
        profile = ABLProfile(
            roughness_length=0.1,
            surface_layer_height=100.0,
            ground_level=0.0,
        )
        assert profile.friction_velocity is None


class TestUnknownKeysStillRejected:
    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError):
            ABLProfile(**_minimal(bogus_field=1.0))
