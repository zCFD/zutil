"""Default coverage for CellReorderingOptions.

The reordering defaults are not cosmetic: they decide whether a case gets the
space-filling-curve cell ordering and colour-wise face sort that #5568 added
for cache locality and colouring balance. #5709 moved validation to pydantic
and, in doing so, dropped `sfc` and `face_sort` to False, silently disabling
both on every deck that does not set them explicitly.

Nothing caught it. The C++ reads in Mesh.cxx fall back to off when a key is
absent, so the solver could not tell an intended False from a lost True, and
the byte-identical dump pin in tests/unit/test_byte_identical_dump.py compared
pydantic against pydantic, never against the voluptuous schema that shipped
the True values. These tests pin the defaults themselves.
"""

from zutil.validate.schemas.solver_settings_schemas import CellReorderingOptions


class TestDefaults:
    def test_sfc_on_by_default(self) -> None:
        assert CellReorderingOptions().sfc is True

    def test_face_sort_on_by_default(self) -> None:
        assert CellReorderingOptions().face_sort is True

    def test_hilbert_is_the_default_curve(self) -> None:
        assert CellReorderingOptions().sfc_curve == "hilbert"

    def test_profile_reducing_orderings_off_by_default(self) -> None:
        """rcm and sloan are alternatives to sfc, not additions to it — only
        the best-by-profile ordering is kept, so leave them for a deck to ask
        for."""
        defaults = CellReorderingOptions()
        assert defaults.rcm is False
        assert defaults.sloan is False


class TestExplicitValuesSurvive:
    """A deck that turns the defaults off must still get False, so the
    defaults stay overridable rather than becoming forced-on."""

    def test_sfc_can_be_disabled(self) -> None:
        assert CellReorderingOptions(sfc=False).sfc is False

    def test_face_sort_can_be_disabled(self) -> None:
        assert CellReorderingOptions(face_sort=False).face_sort is False
