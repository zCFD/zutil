"""
The mf-gmres pseudo-time scheme (MFGMRESScheme) solves each implicit step
with restarted GMRES applying the analytic flux Jacobian matrix-free.

What the schema has to hold, and where each rule comes from:

* Every tuning key is optional. The solver reads each one only if present
  (``scheme.hasKey(...)`` in SolverBase.cxx), so an unset key must stay out
  of the dict the solver receives and the C++ default applies.
* Counts and tolerances are strictly positive: a zero-length Krylov
  subspace, a zero tolerance, a zero iteration cap or zero sweeps are not a
  solve, and a zero over-relaxation zeroes the preconditioner diagonal.
* The preconditioner and orthogonalisation names are the strings the C++
  compares against. An unrecognised preconditioner name would fall through
  to block Jacobi without a word, so the schema must reject it.
* "lu-sgs nonlinear" needs ``flexible: true``: the unlinearised sweep is not
  a linear operator, so plain right-preconditioned GMRES has no fixed
  operator to build a Krylov space for. SolverBase.cxx throws on it at
  start-up; the schema states the same rule so it is caught at validation.
* mf-gmres cannot run with SAS (``turbulence.model: "sas"`` or
  ``turbulence.hybrid: "SAS"``). The Jacobian twins of the viscous kernel
  are given a scratch time-step array that is zeroed before every pass
  (ExplicitSolver.cxx), while under SAS the viscous kernel reads slot [0]
  of that array (the eddy viscosity the SST source kernel stored in the real
  array) to decide whether the WALE floor applies (Viscous.h). The twin
  therefore always sees zero and applies the floor where the residual does
  not, so the Jacobian is not the derivative of the residual being solved.
  WALE only writes that slot and DES/DDES/IDDES never read it in the
  viscous kernel, so those stay allowed.
"""

import copy
import typing

import pytest
from pydantic import ValidationError

from zutil.docs.anchors import reference_url
from zutil.validate.error_handling import ValidationErrorHandler
from zutil.validate.schemas.convergence_schemas import MFGMRESScheme
from zutil.validate.schemas.equations_schemas import (
    hybrid_types,
    turbulence_model_types,
)
from zutil.validate import validate_zcfd_input
from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _config(scheme=None, turbulence=None, mesh_scheme=None):
    """Steady compressible RANS config marching with mf-gmres."""
    config = {
        "config_version": 2,
        "solver": {
            "solver_settings": {"type": "compressible"},
            "equations": {
                "type": "rans",
                "turbulence": turbulence or {"model": "sst"},
            },
            "numerical_scheme": {"order": "second"},
            "fluid_properties": {"material": "air"},
            "reference_conditions": {
                "IC_1": {
                    "temperature": 300.0,
                    "pressure": 101325.0,
                    "v": [10.0, 0.0, 0.0],
                    "reynolds_no": 1.0e6,
                },
            },
            "initialisation": {"initial_conditions": "IC_1"},
            "time_settings": {"type": "steady"},
            "convergence_control": {
                "cycles": 10,
                "scheme": {"name": "mf-gmres", **(scheme or {})},
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
                    "bc1": {"zones": [1], "type": "wall", "kind": "no slip"},
                },
            },
        },
    }
    if mesh_scheme is not None:
        config["model"]["base"]["convergence_control"] = {
            "cycles": 10,
            "scheme": mesh_scheme,
        }
    return config


def _rejection_text(config):
    """The text an engineer sees when their deck is rejected."""
    with pytest.raises(ValidationErrorHandler) as excinfo:
        validate_zcfd_input(copy.deepcopy(config))
    return str(excinfo.value)


class TestDefaults:
    def test_every_tuning_key_defaults_to_unset(self):
        scheme = MFGMRESScheme()
        assert scheme.name == "mf-gmres"
        assert {k: v for k, v in scheme.model_dump().items() if k != "name"} == {
            "krylov_subspace": None,
            "linear_tolerance": None,
            "max_iterations": None,
            "preconditioner": None,
            "preconditioner_sweeps": None,
            "orthogonalisation": None,
            "preconditioner_over_relaxation": None,
            "turbulence_solver": None,
            "turbulence_sweeps": None,
            "flexible": None,
        }

    def test_unset_keys_never_reach_the_solver(self):
        # The solver reads each key under hasKey(), so its own default only
        # applies if the key is absent from the dict it is handed.
        model = ZCFDInputModelBase.model_validate(_config())
        cpp = model.to_cpp_dict()
        assert cpp["solver"]["convergence_control"]["scheme"] == {"name": "mf-gmres"}


class TestRoundTrip:
    def test_every_key_the_solver_reads_arrives_unchanged(self):
        # Key names and types as SolverBase.cxx reads them from the scheme dict.
        scheme = {
            "krylov_subspace": 30,
            "linear_tolerance": 0.05,
            "max_iterations": 90,
            "preconditioner": "lu-sgs",
            "preconditioner_sweeps": 2,
            "orthogonalisation": "cgs2",
            "preconditioner_over_relaxation": 1.2,
            "turbulence_solver": "amg",
            "turbulence_sweeps": 3,
            "flexible": True,
        }
        model = ZCFDInputModelBase.model_validate(_config(scheme))
        cpp = model.to_cpp_dict()
        assert cpp["solver"]["convergence_control"]["scheme"] == {
            "name": "mf-gmres",
            **scheme,
        }


class TestBounds:
    @pytest.mark.parametrize(
        "key",
        [
            "krylov_subspace",
            "max_iterations",
            "preconditioner_sweeps",
            "turbulence_sweeps",
        ],
    )
    def test_counts_must_be_at_least_one(self, key):
        assert getattr(MFGMRESScheme(**{key: 1}), key) == 1
        for bad in (0, -1):
            with pytest.raises(ValidationError):
                MFGMRESScheme(**{key: bad})

    @pytest.mark.parametrize(
        "key", ["linear_tolerance", "preconditioner_over_relaxation"]
    )
    def test_real_parameters_must_be_strictly_positive(self, key):
        assert getattr(MFGMRESScheme(**{key: 1.0e-12}), key) == 1.0e-12
        for bad in (0.0, -0.1):
            with pytest.raises(ValidationError):
                MFGMRESScheme(**{key: bad})

    def test_an_unknown_key_is_rejected(self):
        with pytest.raises(ValidationError):
            MFGMRESScheme(sub_iterations=2)


class TestEnumerations:
    # The accepted names are the strings SolverBase.cxx compares against.
    @pytest.mark.parametrize("preconditioner", ["block jacobi", "lu-sgs", "none"])
    def test_linear_preconditioners_are_accepted(self, preconditioner):
        ZCFDInputModelBase.model_validate(_config({"preconditioner": preconditioner}))

    @pytest.mark.parametrize(
        "preconditioner", ["ilu", "LU-SGS", "lu-sgs-nonlinear", "jacobi", ""]
    )
    def test_an_unknown_preconditioner_is_rejected(self, preconditioner):
        with pytest.raises(ValidationError):
            MFGMRESScheme(preconditioner=preconditioner)

    @pytest.mark.parametrize("ortho", ["mgs", "cgs2"])
    def test_orthogonalisations_are_accepted(self, ortho):
        assert MFGMRESScheme(orthogonalisation=ortho).orthogonalisation == ortho

    @pytest.mark.parametrize("ortho", ["cgs", "gram-schmidt", "MGS"])
    def test_an_unknown_orthogonalisation_is_rejected(self, ortho):
        with pytest.raises(ValidationError):
            MFGMRESScheme(orthogonalisation=ortho)

    @pytest.mark.parametrize("solver", ["amg", "point-implicit"])
    def test_turbulence_solvers_are_accepted(self, solver):
        assert MFGMRESScheme(turbulence_solver=solver).turbulence_solver == solver

    def test_an_unknown_turbulence_solver_is_rejected(self):
        with pytest.raises(ValidationError):
            MFGMRESScheme(turbulence_solver="gmres")


class TestNonlinearPreconditionerNeedsFlexible:
    def test_accepted_with_flexible(self):
        ZCFDInputModelBase.model_validate(
            _config({"preconditioner": "lu-sgs nonlinear", "flexible": True})
        )

    @pytest.mark.parametrize("flexible", [{}, {"flexible": False}])
    def test_rejected_without_flexible(self, flexible):
        text = _rejection_text(
            _config({"preconditioner": "lu-sgs nonlinear", **flexible})
        )
        assert text == (
            "Configuration validation failed with 1 error:\n\n"
            "  solver › convergence control › scheme › preconditioner\n"
            "    'preconditioner': 'lu-sgs nonlinear' only applies when flexible is True\n"
            "    Reference: "
            + reference_url(
                "reference/solver/convergence-control/scheme#preconditioner"
            )
        )

    @pytest.mark.parametrize("preconditioner", ["block jacobi", "lu-sgs", "none"])
    def test_linear_preconditioners_do_not_need_flexible(self, preconditioner):
        ZCFDInputModelBase.model_validate(
            _config({"preconditioner": preconditioner, "flexible": False})
        )

    def test_a_mesh_override_inherits_the_solver_flexible(self):
        ZCFDInputModelBase.model_validate(
            _config(
                {"flexible": True},
                mesh_scheme={"name": "mf-gmres", "preconditioner": "lu-sgs nonlinear"},
            )
        )
        text = _rejection_text(
            _config(
                {"flexible": True},
                mesh_scheme={
                    "name": "mf-gmres",
                    "preconditioner": "lu-sgs nonlinear",
                    "flexible": False,
                },
            )
        )
        assert (
            "  model › base › convergence control › scheme › preconditioner\n"
            "    'preconditioner': 'lu-sgs nonlinear' only applies when flexible is True"
        ) in text


_SAS_REJECTION = (
    "'name': 'mf-gmres' only applies when solver.equations.turbulence.model "
    "is not 'sas' and solver.equations.turbulence.hybrid is not 'SAS'"
)


class TestScaleAdaptiveSimulationIsRejected:
    def test_the_gated_names_are_real_schema_values(self):
        assert "sas" in typing.get_args(turbulence_model_types)
        assert "SAS" in typing.get_args(hybrid_types)

    @pytest.mark.parametrize(
        "turbulence",
        [
            {"model": "sst"},
            {"model": "sa-neg"},
            {"model": "sst", "hybrid": "DES"},
            {"model": "sst", "hybrid": "DDES"},
            {"model": "sa-neg", "hybrid": "IDDES"},
        ],
    )
    def test_models_that_do_not_read_the_slot_are_accepted(self, turbulence):
        ZCFDInputModelBase.model_validate(_config(turbulence=turbulence))

    @pytest.mark.parametrize(
        "turbulence", [{"model": "sas"}, {"model": "sst", "hybrid": "SAS"}]
    )
    def test_sas_is_rejected(self, turbulence):
        text = _rejection_text(_config(turbulence=turbulence))
        assert text == (
            "Configuration validation failed with 1 error:\n\n"
            "  solver › convergence control › scheme › name\n"
            f"    {_SAS_REJECTION}\n"
            "    Reference: "
            + reference_url("reference/solver/convergence-control/scheme#name")
        )

    def test_a_mesh_override_to_mf_gmres_is_rejected_too(self):
        config = _config(turbulence={"model": "sas"}, mesh_scheme={"name": "mf-gmres"})
        config["solver"]["convergence_control"]["scheme"] = {"name": "lu-sgs"}
        text = _rejection_text(config)
        assert (
            f"  model › base › convergence control › scheme › name\n    {_SAS_REJECTION}"
        ) in text
        assert "  solver › convergence control › scheme › name" not in text

    def test_sas_under_another_scheme_is_accepted(self):
        config = _config(turbulence={"model": "sas"})
        config["solver"]["convergence_control"]["scheme"] = {"name": "lu-sgs"}
        ZCFDInputModelBase.model_validate(config)

    def test_wale_les_is_accepted(self):
        config = _config()
        config["solver"]["equations"] = {"type": "les", "turbulence": {"les": "wale"}}
        ZCFDInputModelBase.model_validate(config)
