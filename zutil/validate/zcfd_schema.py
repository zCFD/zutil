"""The whole deck: the solver block, the meshes, and the rules that span them.

Each block checks its own settings. The rules that read settings from more
than one block are declared on the fields where possible
(``ConditionalOptional``, ``RequiredWhen``, ``Exclusive``, ``Connects``)
and the rest are listed in ``_CROSS_BLOCK_RULES``. Every rule reports at the
setting it concerns, so a deck gets one error per mistake, each with its own
path.
"""

import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

from pydantic import ConfigDict, Field, ValidationError, model_validator
from pydantic_core import InitErrorDetails, PydanticCustomError

from .applicability import find_applicability_violations, find_applicability_warnings
from .schemas import model_schemas as mod
from .schemas import solver_schemas as ss
from .schemas.base_types import When, zCFD_Field
from .schemas.convergence_schemas import (
    EXPLICIT_SCHEME_NAMES,
    IMPLICIT_SCHEME_NAMES,
    IncompressibleConvergenceControl,
    IncompressibleMeshConvergenceControl,
)
from .schemas.output_schemas import _validate_output_variables
from .schemas.variants import VariantRoot
from .validation_utils import force_reports, lowercase_keys_recursive, normalise_key

_logger = logging.getLogger("zutil.validate")

Loc = Tuple[str, ...]

_WARNINGS: ContextVar = ContextVar("zcfd_deck_warnings", default=None)


@contextmanager
def collecting_warnings() -> Iterator[List[Tuple[Loc, str]]]:
    """Gather the warnings a deck raises while it validates, instead of logging them."""
    collected: List[Tuple[Loc, str]] = []
    token = _WARNINGS.set(collected)
    try:
        yield collected
    finally:
        _WARNINGS.reset(token)


def _warn(loc: Loc, message: str) -> None:
    collected = _WARNINGS.get()
    if collected is None:
        _logger.warning(message)
    else:
        collected.append((loc, message))


class ZCFDInputModelBase(VariantRoot):
    """A whole deck: how to solve (`solver`) and what to solve (`model`)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    config_version: Literal[2] = Field(
        default=2,
        description="Configuration schema version. Only version 2 is supported.",
    )

    solver: ss.SolverBlockModel = zCFD_Field(
        title="Solver",
        description="Global solver configuration and overrideable defaults",
    )

    model: Dict[str, mod.ModelSettings] = zCFD_Field(
        title="Model",
        description=(
            "The meshes to solve, each under a name of the engineer's choosing. "
            "Each entry names its mesh file and carries that mesh's boundary "
            "conditions, fluid zones, motion and any overrides of the solver "
            "defaults. Several entries run together in one simulation, coupled "
            "through overset or sliding interfaces."
        ),
        min_length=1,
    )

    @model_validator(mode="before")
    @classmethod
    def inherit_mesh_settings(cls, data):
        """Lay each mesh's override blocks over the solver's before either is checked."""
        return resolve_mesh_settings(data)

    @model_validator(mode="after")
    def check_across_blocks(self):
        """Report every rule that reads more than one block, each at its own setting."""
        deck = self.model_dump(exclude_unset=True, warnings=False)
        unused = _unused_reference_conditions(deck)
        if unused:
            _warn(("solver", "reference_conditions"), unused)
        for path, message in find_applicability_warnings(self):
            _warn(_loc(path), message)
        errors = [
            (_loc(path), "not_applicable", message)
            for path, message in find_applicability_violations(self)
        ]
        for rule in _CROSS_BLOCK_RULES:
            kind = rule.__name__.lstrip("_")
            errors.extend((loc, kind, message) for loc, message in rule(self, deck))
        if errors:
            raise ValidationError.from_exception_data(
                type(self).__name__,
                [
                    InitErrorDetails(
                        type=PydanticCustomError(kind, "{message}", {"message": text}),
                        loc=loc,
                        input=None,
                    )
                    for loc, kind, text in errors
                ],
            )
        return self

    def to_cpp_dict(self) -> dict:
        """Produce a serialised dict from the validated model.

        Serialises to a plain dict with lowercase keys.  Callable
        fields and numpy arrays are preserved — C++ accesses them
        via boost::python.

        A mesh's override blocks arrive already resolved against the
        solver's, so the C++ ParameterHelper reads a mesh's block whole when
        the mesh has one and the solver's otherwise.
        """
        d = self.model_dump(exclude_none=True, mode="python")
        d = lowercase_keys_recursive(d)
        return d

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict:
        """The form JSON Schema, with each documented property's reference anchor under ``zcfd``.

        The root's ``zcfd.mesh_settings`` is the rule :func:`resolve_mesh_settings`
        applies: the ``blocks`` a mesh may set, the ``variant_selectors`` whose
        change replaces a sub-block whole, and, per block, the ``keys`` a mesh
        does not inherit while the ``when`` clause holds.
        """
        from zutil.docs.form_anchors import annotate

        schema = annotate(super().model_json_schema(*args, **kwargs), cls.__name__)
        schema.setdefault("zcfd", {})["mesh_settings"] = _mesh_settings_rule()
        return schema


MESH_OVERRIDES = ss.MESH_OVERRIDES
_VARIANT_SELECTORS = ("name", "type", "kind")
_NOT_INHERITED = {
    "convergence_control": (
        When({"solver.solver_settings.type": "incompressible"}),
        sorted(
            set(IncompressibleConvergenceControl.model_fields)
            - set(IncompressibleMeshConvergenceControl.model_fields)
        ),
    ),
}


def _mesh_settings_rule() -> dict:
    return {
        "blocks": list(MESH_OVERRIDES),
        "variant_selectors": list(_VARIANT_SELECTORS),
        "not_inherited": {
            block: {"when": when.to_schema(), "keys": keys}
            for block, (when, keys) in _NOT_INHERITED.items()
        },
    }


class InheritedSettings:
    """The mesh settings a deck takes from its solver block, by path."""

    def __init__(self, roots: Iterable[Loc]) -> None:
        self._roots = frozenset(tuple(map(normalise_key, root)) for root in roots)

    def source(self, path: Sequence) -> Optional[Loc]:
        """The solver path *path* takes its value from, or None when the mesh sets it."""
        path = tuple(map(normalise_key, path))
        if any(path[:end] in self._roots for end in range(4, len(path) + 1)):
            return ("solver",) + path[2:]
        return None


def resolve_mesh_settings(deck):
    """*deck* with each mesh's override blocks resolved against the solver's.

    A mesh block lists only what differs on that mesh. Dicts merge key by
    key, any other value replaces, and a sub-block that selects a different
    variant (``name``, ``type`` or ``kind``) replaces the solver's whole.
    Keys only the solver may set are never inherited.
    """
    return _resolve(deck)[0]


def inherited_settings(deck: object) -> InheritedSettings:
    """The settings :func:`resolve_mesh_settings` copies into *deck*'s meshes."""
    return InheritedSettings(_resolve(deck)[1])


def _resolve(deck: object) -> tuple:
    roots: List[Loc] = []
    if not isinstance(deck, dict):
        return deck, roots
    solver, meshes = deck.get("solver"), deck.get("model")
    if not isinstance(solver, dict) or not isinstance(meshes, dict):
        return deck, roots
    resolved = {}
    for name, mesh in meshes.items():
        if isinstance(mesh, dict):
            mesh = dict(mesh)
            for block in MESH_OVERRIDES:
                if isinstance(mesh.get(block), dict) and isinstance(
                    solver.get(block), dict
                ):
                    mesh[block] = _overlay(
                        inherited_block(deck, block),
                        mesh[block],
                        ("model", name, block),
                        roots,
                    )
        resolved[name] = mesh
    return {**deck, "model": resolved}, roots


def inherited_block(deck: dict, block: str) -> dict:
    """The part of the solver's *block* that a mesh in *deck* inherits."""
    settings = deck["solver"][block]
    when, keys = _NOT_INHERITED.get(block, (None, ()))
    if when is None or not when.evaluate(deck):
        return settings
    return {k: v for k, v in settings.items() if k not in keys}


def mesh_difference(solver_block: object, resolved_block: object) -> object:
    """What a mesh must write so that its block resolves to *resolved_block*.

    The inverse of the overlay :func:`resolve_mesh_settings` applies:
    resolving the result over *solver_block*, the block the mesh inherits,
    gives back *resolved_block*. A sub-block that selects a different
    variant is kept whole.
    """
    if not isinstance(solver_block, dict) or not isinstance(resolved_block, dict):
        return resolved_block
    if _chooses_other_variant(solver_block, resolved_block):
        return resolved_block
    out = {}
    for key, value in resolved_block.items():
        base = solver_block.get(key)
        if isinstance(value, dict) and isinstance(base, dict):
            inner = mesh_difference(base, value)
            if not isinstance(inner, dict) or inner:
                out[key] = inner
        elif key not in solver_block or base != value:
            out[key] = value
    return out


def _chooses_other_variant(base: dict, top: dict) -> bool:
    return any(_differs(base.get(k), top[k]) for k in _VARIANT_SELECTORS if k in top)


def _overlay(base: dict, top: dict, path: Loc, roots: List[Loc]) -> dict:
    if _chooses_other_variant(base, top):
        return top
    roots.extend(path + (key,) for key in base if key not in top)
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _overlay(out[key], value, path + (key,), roots)
        else:
            out[key] = value
    return out


def _differs(base, top) -> bool:
    if isinstance(base, str) and isinstance(top, str):
        return base.lower() != top.lower()
    return base != top


def _loc(path: str) -> Loc:
    return tuple(
        part
        for segment in path.split(".")
        for part in re.split(r"\[(\d+)\]", segment)
        if part
    )


def _listed(names, joiner: str = "and") -> str:
    quoted = [f"'{n}'" for n in names]
    if len(quoted) < 2:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} {joiner} {quoted[-1]}"


def _meshes(deck: dict) -> Iterator[Tuple[str, dict]]:
    return iter((deck.get("model") or {}).items())


def _boundaries(deck: dict) -> Iterator[Tuple[Loc, dict]]:
    for mesh, settings in _meshes(deck):
        for name, bc in (settings.get("boundary_conditions") or {}).items():
            yield ("model", mesh, "boundary_conditions", name), bc


def _condition_names(deck: dict) -> Iterator[Tuple[Loc, str]]:
    """Every place the deck names a reference condition, and the name it gives."""
    initialisations = [
        (("solver", "initialisation"), deck["solver"].get("initialisation"))
    ]
    initialisations += [
        (("model", mesh, "initialisation"), settings.get("initialisation"))
        for mesh, settings in _meshes(deck)
    ]
    for loc, block in initialisations:
        for key in ("initial_conditions", "reference"):
            yield loc + (key,), (block or {}).get(key)
    for loc, bc in _boundaries(deck):
        yield loc + ("condition",), bc.get("condition")
        yield loc + ("reference",), bc.get("reference")
        sponge = bc.get("sponge_layer_damping") or {}
        yield loc + ("sponge_layer_damping", "condition"), sponge.get("condition")


def _unused_reference_conditions(deck: dict) -> str:
    defined = deck["solver"].get("reference_conditions") or {}
    used = {name for _, name in _condition_names(deck) if isinstance(name, str)}
    unused = sorted(set(defined) - used)
    if not unused:
        return ""
    if len(unused) == 1:
        return (
            f"Reference condition {_listed(unused)} is not used anywhere in the deck."
        )
    return f"Reference conditions {_listed(unused)} are not used anywhere in the deck."


def _undefined_reference(
    model: ZCFDInputModelBase, deck: dict
) -> Iterator[Tuple[Loc, str]]:
    defined = deck["solver"].get("reference_conditions") or {}
    known = (
        f"The reference conditions are {_listed(defined)}."
        if defined
        else "No reference conditions are defined."
    )
    for loc, name in _condition_names(deck):
        if isinstance(name, str) and name not in defined:
            yield loc, f"There is no reference condition '{name}'. {known}"


def _in_plane(vector: list, axes: dict) -> List[float]:
    def along(axis: str) -> float:
        return (1.0 if axis[0] == "+" else -1.0) * vector["xyz".index(axis[1])]

    return [along(axes.get("forward", "+x")), along(axes.get("up", "+z"))]


def _target(model: ZCFDInputModelBase, deck: dict) -> Iterator[Tuple[Loc, str]]:
    conditions = deck["solver"].get("reference_conditions") or {}
    for loc, bc in _boundaries(deck):
        target = bc.get("target")
        if target is None:
            continue
        alpha = target.get("alpha")
        if alpha is None:
            yield (
                loc + ("target",),
                "A target names the angle it turns, as "
                '"target": {"alpha": {"cl": 0.5}}.',
            )
            continue
        forces = force_reports(deck, loc[1])
        named = alpha.get("of")
        chosen = None
        if named is None and not forces:
            yield (
                loc + ("target",),
                "A target reads its coefficient from a force report, and none is "
                "defined: add one under output settings › report › forces, naming "
                "the zones of the body.",
            )
        elif named is None and len(forces) > 1:
            first = next(iter(forces))
            yield (
                loc + ("target", "alpha", "of"),
                f"The force reports are {_listed(forces)}: name the one the "
                f'coefficient is read from with \'of\', as "of": "{first}".',
            )
        elif named is None:
            chosen = next(iter(forces))
        else:
            chosen = next(
                (k for k in forces if normalise_key(k) == normalise_key(named)), None
            )
            if chosen is None:
                known = (
                    f"The force reports are {_listed(forces)}."
                    if forces
                    else "No force reports are defined in output settings › "
                    "report › forces."
                )
                yield (
                    loc + ("target", "alpha", "of"),
                    f"There is no force report '{named}'. {known}",
                )
        if chosen is not None and (forces[chosen] or {}).get("dimensional"):
            yield (
                loc + (("target", "alpha", "of") if named else ("target",)),
                f"Force report '{chosen}' writes dimensional forces, and a target "
                f"trims a coefficient. Set \"dimensional\": False on '{chosen}', or "
                "name a coefficient report with 'of'.",
            )
        name = bc.get("condition")
        if name not in conditions:
            continue
        vector = ((conditions[name] or {}).get("v") or {}).get("vector")
        axes = target.get("axes") or {}
        if vector is None:
            yield (
                loc + ("condition",),
                f"Reference condition '{name}' has no velocity vector, so there is "
                "no angle of attack to trim from. Give it a 'v' vector.",
            )
        elif not any(_in_plane(vector, axes)):
            yield (
                loc + ("condition",),
                f"Reference condition '{name}' has velocity {_shown(vector)}, which "
                "has no component along the target's forward axis "
                f"({axes.get('forward', '+x')}) or up axis ({axes.get('up', '+z')}), "
                "so there is no angle of attack to trim from. Check the target's "
                f"'axes', or give '{name}' a 'v' vector in the forward–up plane.",
            )


def _shown(vector: list) -> str:
    return "[" + ", ".join(f"{float(c):g}" for c in vector) + "]"


def _output_variables(
    model: ZCFDInputModelBase, deck: dict
) -> Iterator[Tuple[Loc, str]]:
    equations = deck["solver"].get("equations") or {}
    if not equations.get("type"):
        return
    blocks = [(("solver", "output_settings"), deck["solver"].get("output_settings"))]
    blocks += [
        (("model", mesh, "output_settings"), settings.get("output_settings"))
        for mesh, settings in _meshes(deck)
    ]
    for loc, block in blocks:
        if block is None:
            continue
        try:
            _validate_output_variables(block, equations.get("type"), equations, deck)
        except ValueError as e:
            yield loc, str(e)


def _scheme_name(convergence_control) -> str:
    return getattr(getattr(convergence_control, "scheme", None), "name", None) or ""


def _explicit_with_coupled_meshes(scheme: str, coupled: List[str]) -> str:
    return (
        f"The explicit pseudo-time scheme '{scheme}' cannot be used when meshes are "
        f"coupled, and overset or sliding interfaces are declared on "
        f"{_listed(coupled)}. Use {_listed(sorted(IMPLICIT_SCHEME_NAMES), 'or')}, "
        "and remove multigrid from the numerical scheme at the same time, since "
        "multigrid does not work with the implicit schemes."
    )


def _pseudo_time_scheme(
    model: ZCFDInputModelBase, deck: dict
) -> Iterator[Tuple[Loc, str]]:
    coupled = sorted(
        mesh
        for mesh, settings in model.model.items()
        for bc in (settings.boundary_conditions or {}).values()
        if getattr(bc, "type", None) == "interface"
        and getattr(bc, "kind", None) in ("overset", "sliding")
    )
    inherited = _scheme_name(model.solver.convergence_control)
    if (
        coupled
        and inherited in EXPLICIT_SCHEME_NAMES
        and any(s.convergence_control is None for s in model.model.values())
    ):
        yield (
            ("solver", "convergence_control", "scheme"),
            _explicit_with_coupled_meshes(inherited, coupled),
        )
    for mesh, settings in model.model.items():
        if settings.convergence_control is None:
            continue
        loc = ("model", mesh, "convergence_control", "scheme")
        scheme = _scheme_name(settings.convergence_control)
        if coupled and scheme in EXPLICIT_SCHEME_NAMES:
            yield loc, _explicit_with_coupled_meshes(scheme, coupled)
        numerical = settings.numerical_scheme or model.solver.numerical_scheme
        if (
            scheme in IMPLICIT_SCHEME_NAMES
            and getattr(numerical, "multigrid", None) is not None
        ):
            yield (
                loc,
                (
                    f"Multigrid cannot be used with the '{scheme}' scheme. Multigrid "
                    f"needs {_listed(EXPLICIT_SCHEME_NAMES, 'or')}."
                ),
            )


def _wall_function_solver(
    model: ZCFDInputModelBase, deck: dict
) -> Iterator[Tuple[Loc, str]]:
    if getattr(model.solver.solver_settings, "type", None) != "incompressible":
        return
    for mesh, settings in model.model.items():
        for name, bc in (settings.boundary_conditions or {}).items():
            if getattr(bc, "uses_compressible_wall_function", False):
                yield (
                    ("model", mesh, "boundary_conditions", name),
                    (
                        "This wall asks for the compressible wall function, by a "
                        "temperature, a heat flux or 'compressible wall function', "
                        "which the incompressible solver does not provide."
                    ),
                )


_CROSS_BLOCK_RULES = (
    _undefined_reference,
    _target,
    _output_variables,
    _pseudo_time_scheme,
    _wall_function_solver,
)


ZCFDInputModelBase.model_rebuild()
