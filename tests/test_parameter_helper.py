"""Tests for ``zutil.Parameters.ParameterHelper``'s block lookup.

Two lookups, deliberately different:

- ``convergence_control`` (and the other overrideable blocks) is
  **model-first**: a mesh's own block wins over the solver default, matching
  ``ParameterHelper::convergenceControl`` in ``src/util/Parameters.h``.
- ``solver_convergence_control`` is **solver-only**, for the keys that are
  global to the run by construction. The incompressible coupling scheme is
  the one that matters: it selects the C++ solver class for the whole run, so
  the schema forbids a per-mesh ``scheme`` and the runtime must not go looking
  for one in a mesh block that legitimately overrides the sibling keys.
"""

from zutil.Parameters import ParameterHelper


def _params(mesh_convergence_control=None):
    """Two-mesh deck whose solver and mesh blocks disagree, so a test can tell
    which block a lookup actually read."""
    mesh = {"mesh": "wing.h5"}
    if mesh_convergence_control is not None:
        mesh["convergence_control"] = mesh_convergence_control
    return {
        "solver": {
            "solver_settings": {"type": "incompressible"},
            "convergence_control": {
                "cycles": 5000,
                "scheme": {"name": "simple", "implicit_relax": 0.7},
            },
        },
        "model": {"wing": mesh},
    }


class TestOverrideableLookup:
    def test_convergence_control_falls_back_to_solver(self):
        ph = ParameterHelper(_params(), "wing")
        assert ph.convergence_control["cycles"] == 5000

    def test_convergence_control_prefers_the_mesh_block(self):
        ph = ParameterHelper(_params({"cycles": 100}), "wing")
        assert ph.convergence_control["cycles"] == 100


class TestSolverLevelLookup:
    def test_reads_the_solver_block_when_no_override_exists(self):
        ph = ParameterHelper(_params(), "wing")
        assert ph.solver_convergence_control["scheme"]["name"] == "simple"

    def test_a_mesh_override_does_not_shadow_the_scheme(self):
        """The failure this guards: a mesh overriding cycles (legal) must not
        hide the solver-level scheme, or the run would silently fall back to
        the coupled solver."""
        ph = ParameterHelper(_params({"cycles": 100}), "wing")
        scheme = ph.solver_convergence_control["scheme"]

        assert scheme["name"] == "simple"
        assert scheme["implicit_relax"] == 0.7

    def test_missing_solver_block_returns_an_empty_dict(self):
        """Callers do .get("scheme", {}) on the result, so it is never None."""
        ph = ParameterHelper({"solver": {}, "model": {"wing": {}}}, "wing")
        assert ph.solver_convergence_control == {}

    def test_null_convergence_control_returns_an_empty_dict(self):
        params = {"solver": {"convergence_control": None}, "model": {"wing": {}}}
        assert ParameterHelper(params, "wing").solver_convergence_control == {}
