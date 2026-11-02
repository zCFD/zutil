# type: ignore
from colorama import Fore
from voluptuous import (
    Schema,
    All,
    Any,
    Required,
    Length,
    Object,
    Range,
    Number,
    Coerce,
    IsFile,
    truth,
    error,
    Extra,
)
from contextlib import contextmanager
from copy import deepcopy
import os
from typing import Callable, Generator
import numpy as np
import warnings


def custom_formatwarning(msg, *args, **kwargs):
    return f"Warning: {msg}\n"


warnings.simplefilter("always", UserWarning)
warnings.formatwarning = custom_formatwarning


@truth
def iscallable(f: Callable) -> bool:
    return callable(f)


def transform_logic_check(d: dict) -> dict:
    if d["type"] == "rbf multiscale":
        if "base point fraction" not in d:
            raise error.Invalid(
                "base point fraction must be set for multiscale transforms",
                path=["base point fraction"],
            )
        if "fixed" in d:
            raise error.Invalid(
                "fixed nodes cannot currently be used with multiscale transforms",
                path=["fixed"],
            )
    elif d["type"] == "rbf":
        if "tol" not in d:
            raise error.Invalid("tol must be set for greedy transforms", path=["tol"])
    elif d["type"] == "idw":
        if "n nearest" not in d:
            raise error.Invalid(
                "n nearest must be set for idw transforms", path=["n nearest"]
            )
        if "power" not in d:
            raise error.Invalid("power must be set for idw transforms", path=["power"])

    return d


def unique_name_check(name_list: list, k1: str, k2: str, k3: str, d: dict) -> None:
    name = d[k1][k2][k3].get("name")
    if name:
        if name in name_list:
            raise error.Invalid(
                "{} {} name must be unique".format(k2, k1), path=d[k1][k2][k3]
            )

        name_list.append(name)


@truth
def has_read_perms(fname: str) -> bool:
    try:
        if fname:
            fname = str(fname)
            exists = os.path.exists(fname)
            if not exists:
                raise error.FileInvalid("File does not exist")
            readable = os.access(fname, os.R_OK)
            if not readable:
                raise error.FileInvalid("File does not have read permission")
            return readable
        else:
            raise error.FileInvalid("Not a file")
    except IOError:
        raise error.FileInvalid("Not a file")


@truth
def is_readable_nastran(case_name: str) -> bool:
    status = []
    for fname in [case_name + ".bdf", case_name + ".op2"]:
        readable = has_read_perms(fname)
        status.append(readable)

    if False not in status:
        return True
    else:
        return False


@contextmanager
def try_except_pass() -> Generator:
    try:
        yield
    except Exception:
        pass


@contextmanager
def try_except_func(func: Callable, *args: None) -> Generator:
    try:
        yield
    except Exception:
        func(*args)


@contextmanager
def try_except_cond_blocks(
    k: str, d: dict, all_errors: error.MultipleInvalid
) -> Generator:
    try:
        yield
    except Exception as e:
        if isinstance(e, error.MultipleInvalid):
            for err in e.errors:
                if err.path[0] != k:
                    err.path.insert(0, k)

                all_errors.add(err)

            all_errors.add(
                error.Invalid(
                    Fore.RED
                    + "Error in configuration for key %s: field %s did not validate (either missing or invalid value)"
                    % (k, e.path)
                    + Fore.RESET,
                    path=[k],
                )
            )
        else:
            all_errors.add(error.Invalid(Fore.RED + repr(e) + Fore.RESET, path=[k]))


@contextmanager
def try_except(all_errors: error.MultipleInvalid, k: str = None) -> Generator:  # type: ignore
    try:
        yield
    except Exception as e:
        if isinstance(e, error.MultipleInvalid):
            for err in e.errors:
                if k and err.path[0] != k:
                    err.path.insert(0, k)

                all_errors.add(err)

        elif isinstance(e, error.Invalid):
            if k and e.path != k:
                e.path.insert(0, k)

            all_errors.add(e)

        elif isinstance(e, KeyError):
            all_errors.add(
                error.Invalid(
                    Fore.RED + "parameters must contain '%s' key" % k + Fore.RESET,
                    path=[k],
                )
            )
        else:
            all_errors.add(error.Invalid(Fore.RED + repr(e) + Fore.RESET, path=[k]))


def normalise_vector_with_warning(v):
    """Normalise a vector and warn if it wasn't already normalised."""

    v_array = np.array(v)
    magnitude = np.linalg.norm(v_array)

    if not np.isclose(magnitude, 1.0, rtol=1e-9):
        with warnings.catch_warnings():
            # custom formatting so not to dump out the whole stack trace
            warnings.simplefilter("always", UserWarning)
            warnings.formatwarning = custom_formatwarning
            warnings.warn(
                f"{Fore.YELLOW}Vector {v} is not normalised (magnitude={magnitude:.6f}). "
                f"Automatically normalising to {(v_array / magnitude).tolist()}.{Fore.RESET}",
            )
        return (v_array / magnitude).tolist()

    return v


def validate_update(keys: dict, solver_type: str) -> None:
    base_keys = [
        "time marching:cfl",
        "time marching:cfl transport",
        "time marching:cfl ramp factor:growth",
        "time marching:cfl ramp factor:initial",
        "time marching:ramp func",
        "time marching:cycles",
        "time marching:unsteady:total time",
        "time marching:unsteady:time step",
        "time marching:unsteady:order",
        "time marching:unsteady:start",
        "time marching:scheme:kind",
        "time marching:cycles per overset cycle",
        "write output:surface variables",
        "write output:volume variables",
        "write output:frequency",
        "write output:frequency:volume data",
        "write output:frequency:surface data",
        "write output:frequency:fwh wall data",
        "write output:frequency:fwh interpolate",
        "write output:frequency:volume interpolate",
        "write output:frequency:checkpoint",
        "write output:compute average and rms",
        "write output:average start time cycle",
        "report:frequency",
        "safe",
    ]

    updatable_keys = {
        "fv": base_keys
        + [
            "time marching:cfl coarse",
            "time marching:multigrid",
            "time marching:multigrid cycles",
            "time marching:multigrid ramp",
            "time marching:prolong factor",
            "time marching:prolong transport factor",
            "time marching:moving mesh",
            "time marching:cycles per overset cycle",
            "rans:order",
            "euler:order",
            "viscous:order",
            "rans:first order cycles",
            "euler:first order cycles",
            "viscous:first order cycles",
        ],
    }

    non_updatable_keys = []
    for key in keys:
        if key.lower() not in updatable_keys[solver_type]:
            non_updatable_keys.append(
                error.Invalid(
                    Fore.RED
                    + "Parameters key: %s cannot be dynamically updated" % key
                    + Fore.RESET
                )
            )

    if non_updatable_keys:
        raise error.MultipleInvalid(errors=non_updatable_keys)


def validate(d: dict, case_name: str) -> dict:
    Vector3d = All(
        Any([Number(), Number(), Number()], (Number(), Number(), Number())),
        Length(min=3, max=3),
    )

    NormalisedVector3d = All(Vector3d, normalise_vector_with_warning)
    IntVector3d = All(Any([int, int, int], (int, int, int)), Length(min=3, max=3))

    if "equations" in d:
        d["equations"] = d["equations"].lower()

    modal_model_schema = {
        Required("zone"): list,
        Required("file format", default="nastran"): Any("nastran"),
        Required("transform type", default="rbf multiscale"): Any(
            "rbf multiscale", "rbf", "idw"
        ),
        Required("alpha", default=1.0): float,
        Required("fluid force scaling", default=1.0): float,
        Required("basis", default="c2"): Any("c4", "c2", "c0", "tps", "gaussian"),
        Required("base point fraction", default=0.1): float,
        Required("tol", default=0.01): float,
        Required("max error", default=0.1): float,
        Required("recalculate rbfs alpha fraction", default=0.0): float,
        "nastran casename": All(str, is_readable_nastran),
        "forcing": {
            Required("location"): Vector3d,
            Required("force"): Any(Vector3d, iscallable),
            Required("frequency"): float,
        },
        "mode mapping max distance": float,
        "mesh transform matrix": np.ndarray,
        "modal damping": list,
        Required("mode list", default=[]): list,
        "fixed zones": list,
        Required("mode mapping power", default=1.0): float,
        Required("mode mapping pureidw", default=True): bool,
        Required("use n nearest neighbours", default=4): int,
        Required("deformation distance", default=1.0): float,
        Required("blending stiffness", default=0.5): float,
        Required("power", default=[3.0, 5.0, 10.0]): list,
        Required("n nearest", default=100): int,
        Required("scale", default=Vector3d([1.0, 1.0, 1.0])): Any(Vector3d),
        Required("update frequency", default=1): int,
        Required("use local rbf nodes", default=False): bool,
        Required("newmark alpha", default=0.25): float,
        Required("newmark delta", default=0.5): float,
        "relaxation factor": float,
    }

    large_int = 1e8

    fsi_schema = {
        Required("fsi"): {
            Required("type"): str,
            Required("zone"): list,
            "fixed zones": list,
            Required("transform type"): Any("rbf multiscale", "rbf", "idw"),
            Required("alpha", default=1.0): float,
            Required("basis", default="c2"): Any("c4", "c2", "c0", "tps", "gaussian"),
            Required("base point fraction", default=0.1): float,
            Required("tol", default=0.01): float,
            Required("max error", default=0.1): float,
            Required("n nearest", default=100): int,
            Required("deformation distance", default=1.0): float,
            Required("blending stiffness", default=0.5): float,
            Required("power", default=[3.0, 5.0, 10.0]): list,
            "user variables": dict,
        }
    }

    restart_schema = {
        Required("restart", default=False): bool,
        "restart casename": All(str, Length(min=1)),
        Required("restart ignore history", default=False): bool,
    }

    interpolate_restart_schema = {
        Required("restart", default=False): bool,
        Required("interpolate restart", default=False): bool,
        Required("restart meshname"): All(str, Length(min=1)),
        Required("restart casename"): All(str, Length(min=1)),
        Required("solution smooth cycles", default=0): All(int, Range(min=0)),
    }

    all_cfd_schema = {
        Required("material", default="air"): All(str, Length(min=1)),
        Required("reference"): All(str, Length(min=1)),
        "initial": Any(str, {Required("name"): str, "func": iscallable}),
        "modal model": {
            key: modal_model_schema
            for key in list(d.get("modal model", {}).keys())
            if key.startswith("mm_")
        },
        "reset face area": All(Number(), Range(min=0.0)),
        "sponge layer damping": {
            Required("distance"): Number(),
            Required("damping factor"): Number(),
            Required("condition"): All(str, Length(min=1)),
        },
    }

    all_fv_schema = {
        d.get("equations", "euler"): {
            Required("limiter", default="van albada"): Any(
                "vanalbada", "van albada", "van leer", "barth jespersen", "none"
            ),
            Required("reconstruction", default="muscl"): Any(
                "muscl", "umuscl", "lpumuscl"
            ),
            Required(
                "permit backward difference during reconstruction", default=False
            ): bool,
            Required("order"): Any("first", "second", "euler_second"),
            Required("first order cycles", default=0): int,
            "linear gradients": bool,
            "leastsq gradients": bool,
            Required("entropy fix", default=0.0): All(
                Number(), Range(min=0.0, max=0.2)
            ),
        },
        Required("create overset halos", default=False): bool,
        Required("mesh quality remediation angle threshold", default=-1.0): Number(),
    }

    base_globaltimemarching_schema = Schema(
        {
            Required("unsteady"): {
                Required("total time"): Number(),
                Required("time step"): Number(),
            },
        }
    )

    fv_globaltimemarching_schema = base_globaltimemarching_schema.extend(
        {
            Required("scheme", default={}): {
                Required("name", default="runge kutta"): Any("euler", "runge kutta"),
                "stage": Any(1, "rk third order tvd"),
                Required("kind", default="global timestepping"): Any(
                    "global timestepping"
                ),
            }
        }
    )

    if d.get("equations", "euler") in ["incomp viscous", "incomp rans", "incomp les"]:
        base_explicit_timemarching_schema = {}
    else:
        base_explicit_timemarching_schema = {
            Required("scheme"): {
                "name": Any("euler", "runge kutta"),
                "stage": Any(1, "rk third order tvd", 4, 5),
                "class": Object,
                Required("kind", default="local timestepping"): Any(
                    "local timestepping"
                ),  # Turn off global time stepping for dual time stepping for time being, 'global timestepping'),
            },
            Required("cycles per overset cycle", default=1): All(
                Coerce(int), Range(min=1)
            ),
        }

    base_implicit_timemarching_schema = {
        Required("scheme"): {
            "name": Any("implicit euler"),
            "stage": Any(1),
            Required("kind", default="local timestepping"): Any(
                "local timestepping"
            ),  # Turn off global time stepping for dual time stepping for time being, 'global timestepping'),
        },
        Required("cycles per overset cycle", default=1): All(Coerce(int), Range(min=1)),
    }

    linear_solver_schema = {
        Required("linear solver options", default={}): {
            Required(
                "flow",
                default={
                    "-ksp_min_it": "2",
                    "-ksp_type": "fgmres",
                    "-ksp_rtol": "1.0e-3",
                    "-ksp_monitor": "",
                    "-ksp_converged_reason": "",
                },
            ): dict,
            Required(
                "turbulence",
                default={
                    "-ksp_min_it": "2",
                    "-ksp_type": "fgmres",
                    "-ksp_rtol": "1.0e-5",
                    "-ksp_monitor": "",
                    "-ksp_converged_reason": "",
                },
            ): dict,
            Required(
                "rbf",
                default={
                    "-ksp_type": "fgmres",
                    "-ksp_rtol": "1.0e-5",
                    "-ksp_monitor": "",
                },
            ): dict,
            Required("double precision", default=False): bool,
        }
    }

    base_dualtimemarching_schema = Schema(
        {
            Required("unsteady"): {
                Required("total time"): Number(),
                Required("time step"): Number(),
                Required("order", default=2): Any("first", "second", 1, 2),
                "start": Coerce(int),
            },
            "ramp func": iscallable,
            Required("cycles"): All(Coerce(int), Range(min=1)),
            Required("moving mesh", default=False): bool,
        }
    )

    if d.get("equations", "euler") not in [
        "incomp viscous",
        "incomp rans",
        "incomp les",
    ]:
        base_dualtimemarching_schema = base_dualtimemarching_schema.extend(
            {
                Required("cfl", default=1.0): Number(),
                "cfl transport": Number(),
                "cfl coarse": Number(),
                "cfl ramp factor": {
                    Required("growth"): Number(),
                    Required("initial"): Number(),
                },
                Required("cfl viscous factor", default=1.0): Number(),
            },
        )

    fv_dualtimemarching_schema = base_dualtimemarching_schema

    with try_except_pass():
        if d["time marching"]["scheme"].get("name", "euler") == "implicit euler":
            fv_dualtimemarching_schema = base_dualtimemarching_schema.extend(
                base_implicit_timemarching_schema
            )
        else:
            base_dualtimemarching_schema = base_dualtimemarching_schema.extend(
                base_explicit_timemarching_schema
            )
            if d.get("equations", "euler") in [
                "incomp viscous",
                "incomp rans",
                "incomp les",
            ]:
                fv_dualtimemarching_schema = base_dualtimemarching_schema
            else:
                fv_dualtimemarching_schema = base_dualtimemarching_schema.extend(
                    {
                        "multigrid": All(Coerce(int), Range(min=1)),
                        "multigrid cycles": All(Coerce(int), Range(min=1)),
                        "multigrid ramp": Number(),
                        "prolong factor": Number(),
                        "prolong transport factor": Number(),
                    }
                )

    #
    #  Equations specific
    #
    euler_schema = {
        d.get("equations", "euler"): {
            Required("inviscid flux scheme", default="Roe"): Any(
                "HLLC", "Rusanov", "Roe", "Roe low diss"
            ),
            "roe low dissipation sensor": Any("FD", "NTS", "DES", "NONE"),
            "roe dissipation sensor minimum": All(Number(), Range(min=0.05, max=1.0)),
            Required("roe entropy fix coefficient", default=0.001): Number(),
            Required("freeze limiter cycle", default=-1): Number(),
        }
    }

    les_schema = {
        d.get("equations", "les"): {
            Required("inviscid flux scheme", default="Roe"): Any(
                "HLLC", "Rusanov", "Roe", "Roe low diss"
            ),
            "roe low dissipation sensor": Any("FD", "NTS", "DES", "NONE"),
            "roe dissipation sensor minimum": All(Number(), Range(min=0.05, max=1.0)),
            Required("roe entropy fix coefficient", default=0.001): Number(),
            Required("turbulence", default={}): {
                Required("les", default="none"): Any("none", "WALE"),
            },
        }
    }

    rans_schema = {
        d.get("equations", "rans"): {
            Required("inviscid flux scheme", default="Roe"): Any(
                "HLLC", "Rusanov", "Roe", "Roe low diss"
            ),
            "roe low dissipation sensor": Any("FD", "NTS", "DES", "NONE"),
            "roe dissipation sensor minimum": All(Number(), Range(min=0.05, max=1.0)),
            Required("roe entropy fix coefficient", default=0.001): Number(),
            Required("freeze limiter cycle", default=-1): Number(),
            Required("turbulence", default={}): {
                Required("model"): Any("sst", "sas", "sa-neg", "sst-transition"),
                Required("les", default="none"): Any(
                    "none", "DES", "DDES", "IDDES", "SAS"
                ),
                Required("betastar", default=0.09): Number(),
                "approximate sa from sst results": bool,
                "limit mut": bool,
                "cdes_kw": Number(),
                "cdes_keps": Number(),
                "cd1": Number(),
                "cd2": Coerce(int),
                "cw": Number(),
                "a1": Number(),
                "production": Coerce(int),
                "rotation correction": bool,
                "cdes": Number(),
                "ca1": Number(),
                "ca2": Number(),
                "ce1": Number(),
                "ce2": Number(),
                "cthetat": Number(),
                "sigmagamma": Number(),
                "sigmathetat": Number(),
                "limit gradient k": All(Number(), Range(min=0.01, max=1.0)),
                "separation correction": bool,
                "qcr": Any(2000, 2020, 2021),
            },
        }
    }

    incomp_linear_solver_schema = {
        Required("linear solver options", default={}): {
            Required(
                "flow",
                default={
                    "-ksp_min_it": "2",
                    "-ksp_type": "fgmres",
                    "-ksp_rtol": "1.0e-7",
                    "-ksp_atol": "1.0e-16",
                    "-ksp_max_it": "20",
                    "-ksp_converged_reason": "",
                    "-ksp_monitor": "",
                    # "-pc_type": "hypre",
                    # "-pc_hypre_type": "boomeramg",
                },
            ): dict,
            Required(
                "turbulence",
                default={
                    "-ksp_min_it": "2",
                    "-ksp_type": "fgmres",
                    "-ksp_rtol": "1.0e-12",
                    "-ksp_atol": "1.0e-16",
                    "-ksp_converged_reason": "",
                    "-ksp_monitor": "",
                    "-ksp_max_it": "20",
                    # "-pc_type": "hypre",
                    # "-pc_hypre_type": "boomeramg",
                },
            ): dict,
            Required("rbf", default={}): dict,
            Required("double precision", default=True): bool,
        }
    }

    incomp_viscous_schema = {
        d.get("equations", "incomp viscous"): {
            Required("relax", default=0.5): All(Number(), Range(min=0.1, max=1.0)),
            Required("correction", default="over relaxed"): Any(
                "none", "orthogonal", "minimum", "over relaxed"
            ),
            Required("correction limiter", default=0.5): All(
                Number(), Range(min=0.0, max=1.0)
            ),
            Required("gradient limiter type", default="cellmd"): Any("cellmd", "cell"),
            Required("gradient limiter", default=0.5): All(
                Number(), Range(min=0.0, max=1.0)
            ),
            Required("max second order non orthogonal angle", default=90.0): All(
                Number(), Range(min=0.0, max=90.0)
            ),
            Required("order"): Any("first", "second", "euler_second", "central"),
            Required("central blend", default=0.8): Number(),
        }
    }

    incomp_viscous_schema.update(incomp_linear_solver_schema)

    incomp_rans_schema = {
        d.get("equations", "incomp rans"): {
            Required("inviscid flux scheme", default="Roe"): Any(
                "HLLC", "Rusanov", "Roe", "Roe low diss"
            ),
            "roe low dissipation sensor": Any("FD", "NTS", "DES", "NONE"),
            "roe dissipation sensor minimum": All(Number(), Range(min=0.05, max=1.0)),
            Required("roe entropy fix coefficient", default=0.001): Number(),
            Required("freeze limiter cycle", default=-1): Number(),
            Required("turbulence", default={}): {
                Required("model"): Any("sst", "sas", "sa-neg", "sst-transition"),
                Required("les", default="none"): Any(
                    "none", "DES", "DDES", "IDDES", "SAS"
                ),
                Required("betastar", default=0.09): Number(),
                Required("relax", default=0.5): Number(),
                "approximate sa from sst results": bool,
                "limit mut": bool,
                "cdes_kw": Number(),
                "cdes_keps": Number(),
                "cd1": Number(),
                "cd2": Coerce(int),
                "cw": Number(),
                "a1": Number(),
                "production": Coerce(int),
                "rotation correction": bool,
                "cdes": Number(),
                "ca1": Number(),
                "ca2": Number(),
                "ce1": Number(),
                "ce2": Number(),
                "cthetat": Number(),
                "sigmagamma": Number(),
                "sigmathetat": Number(),
                "limit gradient k": All(Number(), Range(min=0.01, max=1.0)),
                "separation correction": bool,
                "qcr": Any(2000, 2020, 2021),
            },
        }
    }

    incomp_rans_schema[d.get("equations", "incomp rans")].update(
        incomp_viscous_schema[d.get("equations", "incomp viscous")]
    )
    incomp_rans_schema.update(incomp_linear_solver_schema)

    incomp_les_schema = {
        d.get("equations", "incomp les"): {
            Required("inviscid flux scheme", default="Roe"): Any(
                "HLLC", "Rusanov", "Roe", "Roe low diss"
            ),
            "roe low dissipation sensor": Any("FD", "NTS", "DES", "NONE"),
            "roe dissipation sensor minimum": All(Number(), Range(min=0.05, max=1.0)),
            Required("roe entropy fix coefficient", default=0.001): Number(),
            Required("turbulence", default={}): {
                Required("les", default="none"): Any("none", "WALE"),
                Required("relax", default=0.5): Number(),
                Required("les", default="wale"): "wale",
            },
        }
    }

    incomp_les_schema[d.get("equations", "incomp les")].update(
        incomp_viscous_schema[d.get("equations", "incomp viscous")]
    )

    incomp_les_schema.update(incomp_linear_solver_schema)

    precondition_schema = {
        d.get("equations", "euler"): {
            Required("precondition", default=False): bool,
            "preconditioner": {Required("minimum mach number"): Number()},
        }
    }

    equations_to_schema = {
        "euler": euler_schema,
        "rans": rans_schema,
        "viscous": euler_schema,
        "les": les_schema,
        "incomp viscous": incomp_viscous_schema,
        "incomp rans": incomp_rans_schema,
        "incomp les": incomp_les_schema,
    }

    tr_schema = All(
        {
            Required("type", default="rbf"): Any("rbf", "rbf multiscale", "idw"),
            "base point fraction": Number(),
            Required("zone"): list,
            Required("func"): iscallable,
            "alpha": Number(),
            "basis": Any("c0", "c2", "tps", "gaussian"),
            "tol": Number(),
            "fixed": {Required("zone"): list},
            Required("power", default=[3.0, 5.0, 10.0]): list,
            "fixed zones": list,
            Required("n nearest", default=100): int,
            Required("deformation distance", default=1.0): float,
            Required("blending stiffness", default=0.5): float,
        },
        transform_logic_check,
    )

    #  Keys corresponding to dictionaries for further testing

    all_errors = error.MultipleInvalid()

    global_time_stepping = False
    with try_except_pass():
        if d["time marching"]["scheme"]["kind"] == "global timestepping":
            global_time_stepping = True

    num_eq = 5
    if "equations" in d:
        equations = d["equations"]
        if equations.endswith("rans"):
            num_eq = 7
            with try_except_pass():
                if d[equations]["turbulence"]["model"] == "sa-neg":
                    num_eq = 6
            with try_except_pass():
                if d[equations]["turbulence"]["model"] == "sst-transition":
                    num_eq = 9

    with try_except_pass():
        if d[equations]["inviscid flow"] and d[equations]["shock sensing"]:
            all_errors.add(
                error.Invalid(
                    Fore.RED
                    + "equations: shock sensing requires inviscid flow to be set to False"
                    + Fore.RESET
                )
            )

    base_variables = (
        ["var_%i" % ii for ii in range(1, num_eq + 1)]
        + ["resvar_%i" % ii for ii in range(1, num_eq + 1)]
        + ["vargrad_%i" % ii for ii in range(1, num_eq + 1)]
    )

    base_variables += [
        "walldist",
        "timestep",
        "pressure",
        "pressuregrad",
        "gauge_pressure",
        "density",
        "densitygrad",
        "velocity",
        "velocitynd",
        "velocitynearnd",
        "velocitygrad",
        "mach",
        "temperature",
        "temperaturegrad",
        "potentialtemperature",
        "viscosity",
        "mu",
        "nu",
        "V",
        "v",
        "p",
        "rho",
        "m",
        "t",
        "T",
        "cp",
        "totalcp",
        "ek",
        "enstrophy",
        "overset",
        "densitylimiter",
        "klimiter",
        "omegalimiter",
    ]

    output_volume_variables = base_variables + [
        "cell_centre",
        "cell_velocity",
        "cell_volupdate",
        "cellzone",
        "cellcolour",
        "sponge",
        "zone",
        "centre",
        "parent",
        "vorticity",
        "vorticitynd",
        "Qcriterion",
        "Qcriterionnd",
        "helicity",
        "helicitynd",
        "FailCell",
        "MomentumSource",
        "roelowdissipation",
        "cellvolume",
        "badcells",
    ]

    les_volume_variables = ["eddy", "reynoldstress"]

    rans_volume_variables = [
        "viscouslengthscale",
        "lesregion",
        "ti",
        "turbulencekineticenergy",
        "turbulencekineticenergygrad",
        "turbulenceeddyfrequency",
        "turbulenceeddyfrequencygrad",
        "nuTilde",
        "walldistancezone",
        "sagradientlimiter",
        "turbulencetimestep",
        "menterf1",
    ]

    transition_volume_variables = [
        "turbulenceintermittency",
        "transitionreynoldsnumber",
    ]

    output_surface_variables = base_variables + [
        "pressureforce",
        "pressuremoment",
        "pressuremomentx",
        "pressuremomenty",
        "pressuremomentz",
        "centre",
        "densitylimiter",
        "klimiter",
        "omegalimiter",
        "immersed boundary yplus",
        "immersed boundary vector",
        "immersed boundary normal",
    ]

    viscous_surface_variables = [
        "yplus",
        "ut",
        "frictionforce",
        "frictionmoment",
        "frictionmomentx",
        "frictionmomenty",
        "frictionmomentz",
        "cf",
        "eddy",
    ]

    rans_surface_variables = [
        "roughness",
        "walldistancezone",
        "walldist",
        "walldistance",
        "LESRegion",
        "turbulencekineticenergy",
        "turbulenceeddyfrequency",
        "nuTilde",
    ]

    default_volume_variables = ["V", "T", "p", "rho", "mach"]

    default_surface_variables = ["p", "T", "rho"]

    default_viscous_surface_variables = ["yplus", "ut"]

    modal_model_surface_variables = [
        "modeshapedisplacement_{}".format(ii) for ii in range(1, 15)
    ]
    modal_model_surface_variables += [
        "modeshaperotation_{}".format(ii) for ii in range(1, 11)
    ]
    modal_model_surface_variables += ["modaldeformation", "meshvelocity", "cell_centre"]

    modal_model_volume_variables = ["meshvelocity"]

    time_averaged_surface_variables = ["V_avg", "T_avg", "p_avg"]
    rms_surface_variables = ["V_rms", "T_rms", "p_rms"]
    time_averaged_volume_variables = ["V_avg", "T_avg", "p_avg"]
    rms_volume_variables = ["V_rms", "T_rms", "p_rms"]

    default_output_freq_dict = {
        "volume data": large_int,
        "surface data": large_int,
        "volume interpolate": large_int,
        "fwh wall data": large_int,
        "fwh interpolate": large_int,
        "checkpoint": large_int,
        "volume data start": 1,
        "surface data start": 1,
        "volume interpolate start": 1,
        "fwh interpolate start": 1,
        "fwh wall data start": 1,
        "checkpoint start": 1,
    }

    if "equations" in d:
        if d["equations"].endswith("viscous") or d["equations"] == "incomp viscous":
            output_surface_variables += viscous_surface_variables
            default_surface_variables += default_viscous_surface_variables

        if d["equations"].endswith("les"):
            output_volume_variables += les_volume_variables
            output_surface_variables += viscous_surface_variables
            default_volume_variables += les_volume_variables
            default_surface_variables += default_viscous_surface_variables

        if d["equations"].endswith("rans"):
            output_surface_variables += (
                viscous_surface_variables + rans_surface_variables
            )
            output_volume_variables += (
                les_volume_variables + les_volume_variables + rans_volume_variables
            )
            with try_except_pass():
                if d[equations]["turbulence"]["model"] == "sst-transition":
                    output_volume_variables += transition_volume_variables

            default_volume_variables += les_volume_variables
            default_surface_variables += default_viscous_surface_variables

    if "modal model" in d:
        output_surface_variables += modal_model_surface_variables
        output_volume_variables += modal_model_volume_variables

    if "write output" in d:
        if "compute average and rms" in d["write output"]:
            if d["write output"]["compute average and rms"]:
                output_surface_variables += (
                    time_averaged_surface_variables + rms_surface_variables
                )
                output_volume_variables += (
                    time_averaged_volume_variables + rms_volume_variables
                )

    readable_file = All(str, IsFile(), has_read_perms)

    base_output_schema = {
        Required("frequency", default=default_output_freq_dict): Any(
            All(Coerce(int), Range(min=1)),
            {
                Required("volume data", default=large_int): All(
                    Coerce(int), Range(min=1)
                ),
                Required("surface data", default=large_int): All(
                    Coerce(int), Range(min=1)
                ),
                Required("volume interpolate", default=large_int): All(
                    Coerce(int), Range(min=1)
                ),
                Required("fwh interpolate", default=large_int): All(
                    Coerce(int), Range(min=1)
                ),
                Required("fwh wall data", default=large_int): All(
                    Coerce(int), Range(min=1)
                ),
                Required("checkpoint", default=large_int): All(
                    Coerce(int), Range(min=1)
                ),
                Required("volume data start", default=1): All(
                    Coerce(int), Range(min=1)
                ),
                Required("surface data start", default=1): All(
                    Coerce(int), Range(min=1)
                ),
                Required("volume interpolate start", default=1): All(
                    Coerce(int), Range(min=1)
                ),
                Required("fwh interpolate start", default=1): All(
                    Coerce(int), Range(min=1)
                ),
                Required("fwh wall data start", default=1): All(
                    Coerce(int), Range(min=1)
                ),
                Required("checkpoint start", default=1): All(Coerce(int), Range(min=1)),
            },
        ),
        Required("format", default="vtk"): Any(
            "none", "vtk", "ensight", "fwensight", "native"
        ),
        "scripts": [readable_file],
        "variable_name_alias": {str: str},
        Required(
            "surface variables", default=default_surface_variables
        ): output_surface_variables,
        Required(
            "volume variables", default=default_volume_variables
        ): output_volume_variables,
        "volume interpolate": [readable_file],
        "immersed boundary surface": [
            {
                Required("zone"): int,
                Required("stl"): All(str, IsFile(), has_read_perms),
            }
        ],
        "fwh interpolate": [readable_file],
        Required("fwh wall data", default=False): bool,
        "compute average and rms": bool,
        "average start time cycle": int,
    }

    all_solver_schema = {
        Required("units", default="SI"): Any("SI"),
        Required("scale", default=Vector3d([1.0, 1.0, 1.0])): Any(Vector3d),
        Required("equations"): Any(
            "euler",
            "rans",
            "viscous",
            "les",
            "incomp viscous",
            "incomp rans",
            "incomp les",
        ),
        Required("partitioner", default="scotch"): Any("metis", "scotch"),
        Required("safe", default=False): bool,
        Required("time marching", default={}): dict,
        "mesh transform matrix": np.ndarray,
        "append mesh name": str,
        Required("mapping", default={}): {
            Required("idw from cells order", default=0): Any(0, 1),
            Required("number of additional overset active layers", default=0): All(
                Coerce(int), Range(min=0)
            ),
        },
        "reference": str,
        Required("report", default={}): dict,
        Required("write output", default={}): base_output_schema,
        Required(d.get("material", "air"), default={}): {
            Required("sutherlands const", default=110.4): Number(),
            Required("prandtl no", default=0.72): Number(),
            Required("gas constant", default=287.0): Number(),
            Required("gamma", default=1.4): Number(),
            Required("turbulent prandtl no", default=0.9): Number(),
            "gravity": Vector3d,
        },
        "zone_id_to_name": dict,
        "variable_name_alias": dict,
    }

    bc_keys = {key: dict for key in list(d.keys()) if key.startswith("bc_")}
    ic_keys = {
        key: Any(dict, iscallable) for key in list(d.keys()) if key.startswith("ic_")
    }
    all_solver_schema.update(
        {key: tr_schema for key in list(d.keys()) if key.startswith("tr_")}
    )
    fz_schema = {key: dict for key in list(d.keys()) if key.startswith("fz_")}

    all_solver_schema.update(bc_keys)
    all_solver_schema.update(ic_keys)
    all_solver_schema.update(fz_schema)

    if "time marching" in d:
        if "scheme" in d["time marching"]:
            if d["time marching"]["scheme"].get("name", "euler") == "implicit euler":
                all_solver_schema.update(linear_solver_schema)

    if "modal model" in d or "fsi" in d:
        all_solver_schema.update(linear_solver_schema)

    mesh_transform = True
    # if "tr_1" in d:
    #     mesh_transform = True

    if "equations" in d:
        if d["equations"].startswith("dg") or any(
            k.startswith("frpm") or k == "caa" for k in d.keys()
        ):
            all_errors.add(
                error.Invalid(
                    Fore.RED
                    + "high order and CAA codepaths are deprecated and no longer supported."
                    + Fore.RESET
                )
            )

        this_schema = Schema(all_solver_schema)
        time_marching_schema = base_globaltimemarching_schema

        if d["equations"] in [
            "euler",
            "viscous",
            "rans",
            "les",
            "incomp viscous",
            "incomp rans",
            "incomp les",
        ]:
            time_marching_schema = fv_globaltimemarching_schema
            this_schema = (
                this_schema.extend(all_cfd_schema)
                .extend(all_fv_schema)
                .extend(equations_to_schema[d["equations"]])
            )

            if d.get("interpolate restart", False):
                this_schema = this_schema.extend(interpolate_restart_schema)
            else:
                this_schema = this_schema.extend(restart_schema)

            if not global_time_stepping:
                time_marching_schema = fv_dualtimemarching_schema
                this_schema = this_schema.extend(precondition_schema)

            if "fsi" in d:
                this_schema = this_schema.extend(fsi_schema)

        with try_except(all_errors):
            d = this_schema(d)

        with try_except(all_errors, k="time marching"):
            d["time marching"] = time_marching_schema(d["time marching"])

    else:
        all_errors.add(
            error.Invalid(
                Fore.RED
                + "equations key has not been specified in parameters dictionary"
                + Fore.RESET
            )
        )

    fsi_keys = ["modal model", "fsi"]
    for k in fsi_keys:
        if k in d and (d["equations"].startswith("incomp")):
            all_errors.add(
                error.Invalid(
                    Fore.RED
                    + "{} not available with incompressible solvers".format(k)
                    + Fore.RESET,
                    path=["modal model"],
                )
            )

    if fsi_keys[0] in d and fsi_keys[1] in d:
        all_errors.add(
            error.Invalid(
                Fore.RED + "Cannot specify multiple structural models" + Fore.RESET,
                path=["modal model"],
            )
        )

    rd = d.get("report", {})
    fr_schema = {
        key: dict for key in list(rd.get("forces", {}).keys()) if key.startswith("fr_")
    }
    mr_schema = {
        key: dict for key in list(rd.get("monitor", {}).keys()) if key.startswith("mr_")
    }
    mf_schema = {
        key: dict
        for key in list(rd.get("massflow", {}).keys())
        if key.startswith("mf_")
    }

    base_report_schema = Schema(
        {
            Required("frequency"): All(Coerce(int), Range(min=1)),
            "monitor": mr_schema,
        }
    )

    cfd_report_schema = {
        "scale residuals by volume": bool,
        "forces": fr_schema,
        "massflow": mf_schema,
    }

    report_schema = base_report_schema
    report_schema = report_schema.extend(cfd_report_schema)

    with try_except(all_errors, k="report"):
        d["report"] = report_schema(d["report"])

    abl_profile_schema = {
        Required("profile"): {
            Required("abl"): {
                Required("roughness length"): Number(),
                Required("friction velocity"): Number(),
                "surface layer height": Number(),
                "Monin-Obukhov length": Number(),
                "tke": Number(),
                "z0": Number(),
                Required("up", default=[0.0, 0.0, 1.0]): Vector3d,
            }
        }
    }

    abl_field_schema = {
        Required("profile"): {
            Required("field"): All(str, IsFile(), has_read_perms),
            Required("use wall distance", default=False): bool,
        }
    }

    base_ic_schema = Schema(
        {
            Required("pressure"): Number(),
            Required("temperature"): Number(),
            Required("v"): {
                Required("vector"): Vector3d,
                "mach": Number(),
            },
            "reference length": Number(),
            Required(Any("reynolds no", "viscosity")): Number(),
            Required("turbulence intensity", default=0.01): Number(),
            Required("eddy viscosity ratio", default=0.1): Number(),
            Required("ambient turbulence intensity", default=1e-20): Number(),
            Required("ambient eddy viscosity ratio", default=1e-20): Number(),
            "location": Vector3d,
        }
    )

    Schema(Any(base_ic_schema, iscallable))
    reference_ic_schema = Schema(
        {
            Any(
                "static pressure ratio",
                "total pressure ratio",
                "total temperature ratio",
            ): Any(iscallable, Number(), All(str, IsFile(), has_read_perms)),
            Any("mass flow ratio", "mass flow rate"): Number(),
            "vector": Any(iscallable, Vector3d),
            "mach": Number(),
            Required("reference"): str,
        }
    )

    fv_wall_schema = Schema(
        {
            "ref": 3,
            Required("type"): Any("wall", "immersed wall"),
            "kind": Any("slip", "noslip", "wallfunction"),
            "zone": list,
            "roughness": {
                "type": Any("height", "length"),
                "scalar": Number(),
                "field": All(str, IsFile(), has_read_perms),
            },
            "v": Any(
                {Required("linear"): {Required("vector"): Vector3d, "mach": Number()}},
                {
                    Required("rotating"): {
                        Required("axis"): NormalisedVector3d,
                        Required("origin"): Vector3d,
                        Required("omega"): Number(),
                    }
                },
            ),
            "temperature": Any(
                {"scalar": Number()},
                {"field": All(str, IsFile(), has_read_perms)},
            ),
            "immersed normal type": Any("STL", "immersed boundary vector"),
        }
    )

    farfield_schema = Schema(
        {
            "ref": 9,
            "zone": list,
            Required("type"): "farfield",
            Required("kind"): Any(
                "riemann", "pressure", "supersonic", "preconditioned"
            ),
            Required("condition"): str,
        }
    )

    inflow_schema = Schema(
        {
            "ref": 4,
            "zone": list,
            Required("type"): "inflow",
            Required("kind"): Any("default", "massflow"),
            "condition": str,
        }
    )

    outflow_schema = Schema(
        {
            "ref": 5,
            "zone": list,
            Required("type"): "outflow",
            Required("kind"): Any("pressure", "massflow", "radial pressure gradient"),
            "reference radius": Number(),
            "condition": str,
        }
    )

    symmetry_schema = Schema(
        {
            "ref": 7,
            "zone": list,
            Required("type"): "symmetry",
        }
    )

    overset_schema = Schema(
        {
            "ref": 10,
            "zone": list,
            Required("type"): "overset",
        }
    )

    periodic_schema = Schema(
        {
            "zone": list,
            Required("type"): "periodic",
            Required("kind"): Any(
                {
                    Required("rotated"): {
                        Required("theta"): Number(),
                        Required("axis"): NormalisedVector3d,
                        Required("origin"): Vector3d,
                    }
                },
                {Required("linear"): {Required("vector"): Vector3d}},
            ),
        }
    )

    src_schema = Schema(
        {
            "zone": list,
            Required("type"): "wallsrc",
        }
    )

    bc_to_schema = {
        "wall": fv_wall_schema,
        "periodic": periodic_schema,
        "symmetry": symmetry_schema,
        "inflow": inflow_schema,
        "outflow": outflow_schema,
        "farfield": farfield_schema,
        "wallsrc": src_schema,
        "overset": overset_schema,
        "immersed wall": fv_wall_schema,
    }

    fz_base_schema = Schema(
        {
            Required("type"): Any(
                "momentum",
                "disc",
                "rotating",
                "canopy",
                "translating",
                "abl",
                "porous",
                "affine transform",
            ),
            "update frequency": int,
            "output": bool,
        }
    )

    momentum_source_schema = fz_base_schema.extend(
        {
            Required("func"): iscallable,
            Required("def"): All(str, IsFile(), has_read_perms),
        }
    )

    actuator_disk_schema = fz_base_schema.extend(
        {
            "reference plane": bool,
            "reference point": Vector3d,
            Required("discretisation"): {
                Required("type"): Any("disc"),
                Required("number of elements"): Number(),
            },
            Required("model"): {
                Required("type"): Any("simple", "BET"),
                "kind": Any("propellor", "turbine"),
                "power model": Any("curve", "fixed"),
                "thrust coefficient": Number(),
                "thrust coefficient curve": list,
                "power": Number(),
                "power coefficient": Number(),
                "power curve": list,
                "blade twist": list,
                "tilt": Number(),
                "yaw": Number(),
                "blade chord": list,
                "tip loss correction": Any(
                    "elliptic", "acos-fit", "acos shift-fit", "f-fit", "none", "rstar"
                ),
                "aerofoils": dict,
                "aerofoil positions": list,
                "tip loss correction radius": float,
                "number of blades": int,
                "number of sections": int,
            },
            Required("controller"): {
                Required("type"): Any("none", "tsr curve", "fixed", "schedule"),
                "tip speed ratio": Number(),
                "tip speed ratio curve": list,
                "omega": Number(),
                "pitch": Number(),
                "blade pitch tol": Number(),
                "blade pitch range": list,
                "blade pitch step": Number(),
            },
            Required("geometry"): {
                Required("inner radius"): Number(),
                Required("outer radius"): Number(),
                Required("centre"): Vector3d,
                Required("normal"): Vector3d,
                Required("up", default=[0.0, 0.0, 1.0]): Vector3d,
            },
            Required("name"): str,
            "inertia": bool,
            "rotor moment of inertia": bool,
            "mean blade material density": Number(),
            "def": All(str, IsFile(), has_read_perms),
            "verbose": bool,
            "auto yaw": bool,
            "rated power": Number(),
            "tip speed limit": Number(),
            "old yaw": Number(),
            "old omega": Number(),
            "dt": Number(),
            "damage ti": Number(),
            "damage speed": Number(),
            "friction loss": Number(),
            "cut in speed": Number(),
            "cut out speed": Number(),
            Required("rotation direction"): Any("clockwise", "anticlockwise"),
            "status": str,
            "rpm ramp": list,
            "thrust factor": Number(),
            "tip loss correction": Any(
                "elliptic", "acos-fit", "acos shift-fit", "f-fit", "none"
            ),
            "tip loss correction radius": Number(),
            "omega": Number(),
        }
    )

    canopy_schema = fz_base_schema.extend(
        {
            "def": All(str, IsFile(), has_read_perms),
            "field": All(str, IsFile(), has_read_perms),
            Required("func"): iscallable,
            Required("beta_p", default=0.17): Number(),
            Required("beta_d", default=3.37): Number(),
            Required("ceps_4", default=0.9): Number(),
            Required("ceps_5", default=0.9): Number(),
            Required("cd", default=0.25): Number(),
        }
    )

    porous_schema = fz_base_schema.extend(
        {
            "def": All(str, IsFile(), has_read_perms),
            "zone": list,
            Required("alpha", default=1.0e12): Number(),
            Required("c2", default=0.0): Number(),
        }
    )

    abl_schema = momentum_source_schema

    rotating_schema = fz_base_schema.extend(
        {
            Required("zone"): list,
            Required("axis"): NormalisedVector3d,
            Required("origin"): Vector3d,
            Required("omega"): Number(),
        }
    )

    translating_schema = fz_base_schema.extend(
        {
            Required("zone"): list,
            Required("vector"): Vector3d,
            "mach": Number(),
            "translation function": iscallable,
        }
    )

    affine_schema = fz_base_schema.extend(
        {
            Required("zone"): list,
            Required("transform function"): iscallable,
        }
    )

    fz_to_schema = {
        "momentum": momentum_source_schema,
        "disc": actuator_disk_schema,
        "canopy": canopy_schema,
        "abl": abl_schema,
        "rotating": rotating_schema,
        "translating": translating_schema,
        "porous": porous_schema,
        "affine transform": affine_schema,
    }

    force_report_schema = Schema(
        {
            Required("name"): str,
            Required("zone"): list,
            "transform": iscallable,
            "reference area": Number(),
            "reference length": Number(),
            "reference point": Vector3d,
            "reference pressure": Number(),
            Required("dimensional", default=False): bool,
        }
    )

    monitor_point_schema = Schema(
        {
            Required("name"): str,
            Required("point"): Vector3d,
            Required(
                "variables", default=default_volume_variables
            ): output_volume_variables,
        }
    )

    massflow_report_schema = Schema({Required("name"): str, Required("zone"): list})

    for k in ic_keys.keys():
        with try_except_cond_blocks(k, d, all_errors):
            if callable(d[k]):
                ic_dict = d[k](**{"RealTimeStep": 0, "Cycle": 0})
                extra_allowed = True
            else:
                ic_dict = d[k]
                extra_allowed = False
            if "reference" in ic_dict:
                ic_schema = reference_ic_schema
            else:
                ic_schema = base_ic_schema
                if "profile" in ic_dict:
                    if "abl" in ic_dict["profile"]:
                        ic_schema = base_ic_schema.extend(abl_profile_schema)
                    else:
                        ic_schema = base_ic_schema.extend(abl_field_schema)

            if extra_allowed:
                ic_schema_with_extras = ic_schema.extend(
                    {"monitor": {Extra: Any(str, Number(), list, bool)}}
                )  # Allows extra keys to exist so that things like AoA can be tracked.
                _ = ic_schema_with_extras(ic_dict)
                # try:
                #     _ = ic_schema(ic_dict)
                # except error.MultipleInvalid as e:
                #     for extra_key_error in e.errors:
                #         print(
                #             Fore.YELLOW
                #             + "Extra key {} in driven IC {} is not a standard IC condition.\n".format(extra_key_error.path, k)
                #             + "This key will not be used by the solver, and will instead only be passively reported in the report.csv file.\n"
                #             + Fore.RESET
                #         )
            else:
                _ = ic_schema(ic_dict)

    for k in bc_keys.keys():
        with try_except_cond_blocks(k, d, all_errors):
            d[k] = bc_to_schema[d[k]["type"]](d[k])
            if (
                d[k]["type"] in ["immersed wall"]
                and d["time marching"].get("multigrid", 1) > 1
            ):
                all_errors.add(
                    error.Invalid(
                        Fore.RED
                        + "Immersed boundaries cannot be used with multigrid"
                        + Fore.RESET,
                        path=[k],
                    )
                )

    for k in fz_schema.keys():
        with try_except_cond_blocks(k, d, all_errors):
            d[k] = fz_to_schema[d[k]["type"]](d[k])
            # if d[k]['type'] in ('momentum', 'rotating', 'canopy', 'translating', 'abl'):
            #     all_errors.add(error.Invalid(Fore.RED + "%s fluid zones are not currently supported" % d[k]['type'] + Fore.RESET, path=[k]))

    fz_names: list[str] = []
    for k in fr_schema.keys():
        with try_except_cond_blocks(k, d["report"]["forces"], all_errors):
            unique_name_check(fz_names, "report", "forces", k, d)
            d["report"]["forces"][k] = force_report_schema(d["report"]["forces"][k])

    mr_names: list[str] = []
    for k in mr_schema.keys():
        with try_except_cond_blocks(k, d["report"]["monitor"], all_errors):
            unique_name_check(mr_names, "report", "monitor", k, d)
            d["report"]["monitor"][k] = monitor_point_schema(d["report"]["monitor"][k])

    mf_names: list[str] = []
    for k in mf_schema.keys():
        with try_except_cond_blocks(k, d["report"]["massflow"], all_errors):
            unique_name_check(mf_names, "report", "massflow", k, d)
            d["report"]["massflow"][k] = massflow_report_schema(
                d["report"]["massflow"][k]
            )

    if d.get("restart", False) or d.get("interpolate restart", False):
        key = "restart"
        if d.get("restart casename", None):
            case_name = d["restart casename"]
            key = "restart casename"
        case_name += "_results.h5"
        with try_except(all_errors, k=key):
            has_read_perms(case_name)

    if d.get("interpolate restart", False):
        d["restart"] = False
        if d.get("restart meshname", False):
            key = "restart meshname"
            mesh_name = d["restart meshname"]
            if mesh_name.endswith(".h5"):
                mesh_name = mesh_name[:-3]
                d["restart meshname"] = mesh_name
            mesh_name += ".h5"
            with try_except(all_errors, k=key):
                has_read_perms(mesh_name)

    if d.get("write output", None):
        keys = ["volume interpolate", "fwh interpolate"]
        for key in keys:
            if d["write output"].get(key, None):
                with try_except(all_errors, k=key):
                    for item in d["write output"][key]:
                        has_read_perms(item)
        if d["write output"].get("immersed boundary surface", None):
            key = "immersed boundary surface"
            with try_except(all_errors, k=key):
                for item in d["write output"][key]:
                    has_read_perms(item["stl"])

    if all_errors.errors:
        raise all_errors

    # Dynamic type behaviour, therefore need mypy to ignore it

    if isinstance(d["write output"]["frequency"], int):
        f = d["write output"]["frequency"]
        d["write output"]["frequency"] = default_output_freq_dict
        d["write output"]["frequency"]["volume data"] = f  # type: ignore
        d["write output"]["frequency"]["surface data"] = f  # type: ignore
        d["write output"]["frequency"]["volume interpolate"] = f  # type: ignore
        d["write output"]["frequency"]["fwh interpolate"] = f  # type: ignore
        d["write output"]["frequency"]["fwh wall data"] = f  # type: ignore
        d["write output"]["frequency"]["checkpoint"] = f  # type: ignore

    return d
