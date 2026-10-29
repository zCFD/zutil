"""
Enum canonical spellings and same-name-collision coverage.

- limiter_types carries one canonical spelling, "van albada"; a
  BeforeValidator normalises the legacy no-space "vanalbada" at parse time.
- turbulence_limiter is a separate vocabulary ("zero"/"no update"): it selects
  the post-update treatment of invalid turbulence variables, not a
  reconstruction limiter.
- WallBoundary.velocity is distinct from ReferenceConditionsBase.v
  (freestream velocity).
"""

import pytest
from pydantic import ValidationError

from zutil.validate.schemas.equations_schemas import TurbulenceModel
from zutil.validate.schemas.numerical_schemes_schemas import (
    FVCompressibleEulerNumericalScheme,
    FVCompressibleRANSNumericalScheme,
    FVIncompressibleRANSNumericalScheme,
    cell_limiter_types,
    limiter_types,
    sweby_limiter_types,
    turbulence_limiter_types,
)
from zutil.validate.schemas.boundary_conditions_schemas import WallBoundary
from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _rans_config(turbulence_limiter):
    """Minimal compressible RANS config carrying a turbulence_limiter."""
    return {
        "config_version": 2,
        "solver": {
            "solver_settings": {"type": "compressible"},
            "equations": {"type": "rans", "turbulence": {"model": "sst"}},
            "numerical_scheme": {
                "order": "second",
                "turbulence_limiter": turbulence_limiter,
            },
            "fluid_properties": {"material": "ideal gas"},
            "reference_conditions": {
                "IC_1": {
                    "temperature": 300.0,
                    "pressure": 101325.0,
                    "reynolds_no": 1.0e6,
                },
            },
            "initialisation": {"initial_conditions": "IC_1"},
            "time_settings": {"type": "steady"},
            "convergence_control": {
                "cycles": 10,
                "cfl": 1.0,
                "scheme": {"name": "runge kutta"},
            },
            "output_settings": {
                "solution": {"frequency": 10},
                "report": {"frequency": 10},
            },
        },
        "model": {
            "base": {
                "mesh": "base.h5",
                "boundary_conditions": {
                    "bc1": {
                        "type": "farfield",
                        "kind": "riemann",
                        "condition": "IC_1",
                        "zones": [1],
                    },
                },
            }
        },
    }


class TestLimiterCanonicalSpelling(object):
    """Exactly one canonical spelling ("van albada"); legacy accepted."""

    def test_vanalbada_duplicate_removed_from_literal(self):
        import typing

        values = set(typing.get_args(limiter_types))
        assert "van albada" in values
        assert "vanalbada" not in values

    def test_canonical_spelling_accepted(self):
        scheme = FVCompressibleEulerNumericalScheme(limiter="van albada")
        assert scheme.limiter == "van albada"

    def test_legacy_spelling_normalised_on_limiter(self):
        scheme = FVCompressibleEulerNumericalScheme(limiter="vanalbada")
        assert scheme.limiter == "van albada"

    def test_legacy_spelling_case_insensitive(self):
        scheme = FVCompressibleEulerNumericalScheme(limiter="VanAlbada")
        assert scheme.limiter == "van albada"

    def test_default_is_canonical_spelling(self):
        scheme = FVCompressibleEulerNumericalScheme()
        assert scheme.limiter == "van albada"

    def test_other_limiters_still_validate(self):
        for value in ("barth jespersen", "van leer", "none"):
            scheme = FVCompressibleEulerNumericalScheme(limiter=value)
            assert scheme.limiter == value

    def test_unknown_limiter_rejected(self):
        with pytest.raises(ValidationError):
            FVCompressibleEulerNumericalScheme(limiter="not a limiter")

    def test_every_declared_limiter_validates(self):
        import typing

        for value in typing.get_args(limiter_types):
            scheme = FVCompressibleEulerNumericalScheme(limiter=value)
            assert scheme.limiter == value

    def test_literal_is_exactly_the_two_families(self):
        """The Literal must stay in step with the two family tuples."""
        import typing

        assert set(typing.get_args(limiter_types)) == set(sweby_limiter_types) | set(
            cell_limiter_types
        )

    def test_symmetric_ratio_is_the_renamed_one_dimensional_limiter(self):
        scheme = FVCompressibleEulerNumericalScheme(limiter="symmetric ratio")
        assert scheme.limiter == "symmetric ratio"
        assert "symmetric ratio" in sweby_limiter_types

    def test_barth_jespersen_is_now_a_cell_limiter(self):
        """The name changed meaning: it is no longer the Sweby-type function."""
        assert "barth jespersen" in cell_limiter_types
        assert "barth jespersen" not in sweby_limiter_types

    def test_cell_limiters_rejected_on_turbulence_limiter(self):
        """turbulence_limiter has its own vocabulary ("zero"/"no update");
        the cell-based limiter names are not part of it."""
        import typing

        assert not set(cell_limiter_types) & set(
            typing.get_args(turbulence_limiter_types)
        )
        for value in ("michalak gooch", "venkatakrishnan"):
            with pytest.raises(ValidationError):
                FVCompressibleRANSNumericalScheme(turbulence_limiter=value)


class TestLimiterCoefficient(object):
    """K in the grid-scaled smoothness threshold of the cell-based limiters."""

    def test_default(self):
        assert FVCompressibleEulerNumericalScheme().limiter_coefficient == 5.0

    def test_accepts_zero(self):
        scheme = FVCompressibleEulerNumericalScheme(limiter_coefficient=0.0)
        assert scheme.limiter_coefficient == 0.0

    def test_rejects_negative(self):
        with pytest.raises(ValidationError):
            FVCompressibleEulerNumericalScheme(limiter_coefficient=-1.0)

    def test_available_on_incompressible_schemes_too(self):
        """It lives on FVNumericalSchemeBase, which both families inherit."""
        assert FVIncompressibleRANSNumericalScheme().limiter_coefficient == 5.0


class TestTurbulenceLimiterVocabulary(object):
    """turbulence_limiter selects the post-update treatment of invalid
    turbulence variables (ExplicitSolver::initialise compares against "zero"),
    so it does NOT share the reconstruction limiter vocabulary."""

    def test_literal_is_the_solver_vocabulary(self):
        import typing

        assert set(typing.get_args(turbulence_limiter_types)) == {"zero", "no update"}

    def test_zero_accepted_compressible(self):
        scheme = FVCompressibleRANSNumericalScheme(turbulence_limiter="zero")
        assert scheme.turbulence_limiter == "zero"

    def test_zero_accepted_incompressible(self):
        scheme = FVIncompressibleRANSNumericalScheme(turbulence_limiter="zero")
        assert scheme.turbulence_limiter == "zero"

    def test_no_update_accepted(self):
        scheme = FVCompressibleRANSNumericalScheme(turbulence_limiter="no update")
        assert scheme.turbulence_limiter == "no update"

    def test_default_is_unset_so_solver_default_applies(self):
        # to_cpp_dict() is exclude_none, so an unset field leaves the key absent
        # and the solver keeps TurbulenceLimiter::NOUPDATE.
        assert FVCompressibleRANSNumericalScheme().turbulence_limiter is None

    def test_reconstruction_limiter_spellings_rejected(self):
        for value in ("van albada", "vanalbada", "barth jespersen", "van leer"):
            with pytest.raises(ValidationError):
                FVCompressibleRANSNumericalScheme(turbulence_limiter=value)

    def test_zero_reaches_the_dict_the_solver_reads(self):
        """ExplicitSolver::initialise reads
        parameters.numericalScheme().get<std::string>("turbulence_limiter")
        and compares it against "zero"; anything that fails to arrive there
        verbatim leaves TurbulenceLimiter::ZERO unreachable.
        """
        config = _rans_config(turbulence_limiter="zero")
        cpp_dict = ZCFDInputModelBase.model_validate(config).to_cpp_dict()

        assert cpp_dict["solver"]["numerical_scheme"]["turbulence_limiter"] == "zero"


class TestOrderLegacyIntegerNormalisation(object):
    """FVNumericalSchemeBase.normalise_order: legacy integer 'order' values
    (1/2) are accepted and mapped to the canonical string spelling."""

    def test_canonical_string_accepted(self):
        scheme = FVCompressibleEulerNumericalScheme(order="first")
        assert scheme.order == "first"

    def test_legacy_int_one_normalised_to_first(self):
        scheme = FVCompressibleEulerNumericalScheme(order=1)
        assert scheme.order == "first"

    def test_legacy_int_two_normalised_to_second(self):
        scheme = FVCompressibleEulerNumericalScheme(order=2)
        assert scheme.order == "second"

    def test_unknown_int_order_rejected(self):
        with pytest.raises(ValidationError):
            FVCompressibleEulerNumericalScheme(order=3)


class TestInviscidFluxSchemeCaseNormalisation(object):
    """FVNumericalSchemeBase.normalise_inviscid_flux_scheme: legacy
    mixed-case flux scheme names are lowercased before the Literal check."""

    def test_canonical_lowercase_accepted(self):
        scheme = FVCompressibleEulerNumericalScheme(inviscid_flux_scheme="hllc")
        assert scheme.inviscid_flux_scheme == "hllc"

    def test_mixed_case_normalised_to_lowercase(self):
        scheme = FVCompressibleEulerNumericalScheme(inviscid_flux_scheme="Roe")
        assert scheme.inviscid_flux_scheme == "roe"

    def test_unknown_flux_scheme_rejected_after_lowercasing(self):
        with pytest.raises(ValidationError):
            FVCompressibleEulerNumericalScheme(inviscid_flux_scheme="Not_A_Scheme")


class TestTurbulenceModelProductionLegacyIntegers(object):
    """TurbulenceModel.normalise_production: legacy integer 'production'
    values (0-3) are accepted and mapped to the canonical string spelling."""

    def test_canonical_string_accepted(self):
        model = TurbulenceModel(model="sst", production="kato-launder")
        assert model.production == "kato-launder"

    def test_legacy_int_zero_normalised_to_vorticity(self):
        model = TurbulenceModel(model="sst", production=0)
        assert model.production == "vorticity"

    def test_legacy_int_three_normalised_to_kato_launder(self):
        model = TurbulenceModel(model="sst", production=3)
        assert model.production == "kato-launder"

    def test_unknown_int_production_rejected(self):
        with pytest.raises(ValidationError):
            TurbulenceModel(model="sst", production=99)

    def test_production_left_unset_by_default(self):
        model = TurbulenceModel(model="sst")
        assert model.production is None


class TestWallVelocityRename(object):
    """WallBoundary exposes `velocity`, not `v`, end-to-end."""

    def test_velocity_field_accepts_linear_spec(self):
        wall = WallBoundary(
            zones=[0],
            type="wall",
            kind="no slip",
            velocity={"type": "linear", "vector": [1.0, 0.0, 0.0]},
        )
        assert wall.velocity.type == "linear"
        assert wall.velocity.vector == [1.0, 0.0, 0.0]

    def test_velocity_field_accepts_rotating_spec(self):
        wall = WallBoundary(
            zones=[0],
            type="wall",
            kind="wall function",
            velocity={
                "type": "rotating",
                "axis": [0.0, 0.0, 1.0],
                "origin": [0.0, 0.0, 0.0],
                "omega": 10.0,
            },
        )
        assert wall.velocity.type == "rotating"
        assert wall.velocity.omega == 10.0
        assert wall.velocity.origin == [0.0, 0.0, 0.0]

    def test_v_field_no_longer_accepted(self):
        """Old field name 'v' must be rejected — extra='forbid' on WallBoundary."""
        with pytest.raises(ValidationError):
            WallBoundary(
                zones=[0],
                type="wall",
                kind="no slip",
                v={"type": "linear", "vector": [1.0, 0.0, 0.0]},
            )

    def test_dumped_dict_uses_velocity_key(self):
        wall = WallBoundary(
            zones=[0],
            type="wall",
            kind="no slip",
            velocity={"type": "linear", "vector": [1.0, 0.0, 0.0]},
        )
        dumped = wall.model_dump(exclude_none=True)
        assert "velocity" in dumped
        assert "v" not in dumped
