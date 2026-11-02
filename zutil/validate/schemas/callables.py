"""Contracts for the Python functions a control dictionary may hold.

zCFD calls every deck function with keyword arguments only. Each contract
names those keywords, carries a representative value for each, and says what
the function must return. Validation binds the function's signature to the
keywords, calls it with the representative values, and checks the shape of
what comes back, so a function the solver cannot use is rejected with the
control file rather than part-way through a run. A call that raises is made
once more at a point off the coordinate axes, so a function singular only on
an axis is not rejected for it.

A three-component vector is returned as any sequence of three numbers: a
tuple, a list or a numpy array. A ``{"v1": x, "v2": y, "v3": z}`` dict is
accepted as well.
"""

import inspect
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Optional

from pydantic import WrapValidator
from pydantic_core.core_schema import ValidatorFunctionWrapHandler

Vector3 = tuple[float, float, float]

_VECTOR_KEYS = ("v1", "v2", "v3")


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _sequence(value: Any) -> Optional[list]:
    if isinstance(value, (str, bytes, Mapping, set, frozenset)):
        return None
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, Sequence):
        return list(value)
    return None


def is_vector3(value: Any) -> bool:
    """True when the solver can read ``value`` as a three-component vector."""
    if isinstance(value, Mapping):
        return set(value) == set(_VECTOR_KEYS) and all(
            _is_number(value[k]) for k in _VECTOR_KEYS
        )
    items = _sequence(value)
    return items is not None and len(items) == 3 and all(map(_is_number, items))


def _describe(value: Any) -> str:
    text = repr(value)
    if len(text) > 80:
        text = text[:77] + "..."
    return f"{text} ({type(value).__name__})"


def _vector3(value: Any) -> Optional[str]:
    return None if is_vector3(value) else ""


def _list_of(
    count: int, is_item: Callable[[Any], bool]
) -> Callable[[Any], Optional[str]]:
    def check(value: Any) -> Optional[str]:
        items = _sequence(value)
        if items is not None and len(items) == count and all(map(is_item, items)):
            return None
        return f"it was given {count} locations"

    return check


def _dict_of(
    entries: Mapping[str, Callable[[Any], bool]],
) -> Callable[[Any], Optional[str]]:
    def check(value: Any) -> Optional[str]:
        if not isinstance(value, Mapping):
            return ""
        for key, item in value.items():
            if key not in entries:
                return f"{key!r} is not one of them"
            if not entries[key](item):
                kind = "three numbers" if entries[key] is is_vector3 else "a number"
                return f"{key!r} must be {kind}"
        return None

    return check


_RAMP_TARGETS = _dict_of(
    dict.fromkeys(("cfl", "cfl_turbulence", "cfl_coarse"), _is_number)
)


def _ramp_result(value: Any) -> Optional[str]:
    if _is_number(value):
        return None
    if isinstance(value, Mapping) and not value:
        return ""
    return _RAMP_TARGETS(value)


@dataclass(frozen=True)
class CallableContract:
    """What zCFD passes to one kind of deck function and what it reads back.

    Attributes:
        label: The function as an engineer knows it, used in every message.
        inputs: The keyword arguments the solver passes, each with a
            representative value used to call the function during validation.
        reads: What the solver reads back from the function, in words. Every
            message, field description and conversion decision quotes it.
        check_result: Returns ``None`` for a result the solver can read,
            otherwise a phrase, possibly empty, saying what is wrong with it.
        off_axis: Inputs that replace some of ``inputs`` for a second call
            when the first raises, placed off the coordinate axes.
        returns: The return type the ``Callable`` annotation carries.
    """

    label: str
    inputs: Mapping[str, Any]
    reads: str
    check_result: Callable[[Any], Optional[str]]
    off_axis: Mapping[str, Any] = field(default_factory=dict)
    returns: Any = field(default=Any)

    def call(self, name: str = "f") -> str:
        """The call zCFD makes to a function called ``name``."""
        return f"{name}(*, {', '.join(self.inputs)})"

    def signature_text(self, name: str) -> str:
        """The ``def`` line an engineer writes for a function called ``name``."""
        return f"def {self.call(name)}"

    def definition(self, name: str) -> str:
        """The ``def`` line for ``name`` and what the function returns."""
        return f"{self.signature_text(name)}, returning {self.reads}"

    def description(self, purpose: str) -> str:
        """Field help text: ``purpose``, the call zCFD makes and what it reads back."""
        return f"{purpose}, called as {self.call()}; it returns {self.reads}"

    def signature_error(self, func: Callable) -> Optional[str]:
        """Why the solver cannot call ``func`` with its keywords, or ``None``."""
        name = getattr(func, "__name__", type(func).__name__)
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return None
        keywords = ", ".join(self.inputs)
        try:
            signature.bind(**self.inputs)
        except TypeError:
            return (
                f"{self.label} {name}{signature} cannot take the keywords zCFD "
                f"calls it with: {keywords}. Write it as "
                f"{self.signature_text(name)}."
            )
        return None

    def check(self, func: Callable) -> Callable:
        """Reject ``func`` unless the solver can call it and read its result.

        A call that raises is made again with ``off_axis`` in place of the
        matching inputs, and ``func`` is rejected only if that raises too.

        Raises:
            ValueError: If ``func`` cannot take the solver's keywords, raises
                at both sample points, or returns something the solver cannot
                read.
        """
        name = getattr(func, "__name__", type(func).__name__)
        error = self.signature_error(func)
        if error is not None:
            raise ValueError(error)
        inputs = dict(self.inputs)
        try:
            result = func(**inputs)
        except Exception as first:
            if not self.off_axis:
                raise ValueError(
                    f"{self.label} {name} raised {type(first).__name__}: {first} "
                    f"when called with {_sample(inputs)}."
                ) from first
            retry = {**inputs, **self.off_axis}
            try:
                result = func(**retry)
            except Exception as second:
                raise ValueError(
                    f"{self.label} {name} raised {type(first).__name__}: {first} "
                    f"when called with {_sample(inputs)}, and again when called "
                    f"with {_sample(self.off_axis)}."
                ) from second
            inputs = retry
        wanted = self.check_result(result)
        if wanted is not None:
            detail = f"; {wanted}" if wanted else ""
            raise ValueError(
                f"{self.label} {name} returned {_describe(result)} when called "
                f"with {_sample(inputs)}; it must return {self.reads}{detail}."
            )
        return func

    def validate(self, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        """Check a callable against this contract; validate anything else normally."""
        if callable(value) and not isinstance(value, type):
            return self.check(value)
        return handler(value)

    @property
    def annotation(self) -> Any:
        """A typed ``Callable`` checked against this contract at validation."""
        return Annotated[Callable[..., self.returns], WrapValidator(self.validate)]


def _sample(inputs: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in inputs.items())


_LOCATIONS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
_OFF_AXIS = (0.3, 0.7, 0.2)
_OFF_AXIS_LOCATIONS = [_OFF_AXIS, (1.3, 0.4, 0.6)]

INITIAL_CONDITIONS = CallableContract(
    label="The initial conditions function",
    inputs={
        "location": (0.0, 0.0, 0.0),
        "wall_distance": 1.0,
        "pressure": 101325.0,
        "temperature": 288.15,
        "velocity": (1.0, 0.0, 0.0),
    },
    reads="a dict of any of 'pressure', 'temperature' and 'velocity'",
    check_result=_dict_of(
        {"pressure": _is_number, "temperature": _is_number, "velocity": is_vector3}
    ),
    off_axis={"location": _OFF_AXIS},
)

FORCE_TRANSFORM = CallableContract(
    label="The forces transform",
    inputs={"vector": (1.0, 0.0, 0.0)},
    reads="the transformed vector as three numbers",
    check_result=_vector3,
    off_axis={"vector": _OFF_AXIS},
    returns=Vector3,
)

MESH_DEFORMATION = CallableContract(
    label="The mesh deformation func",
    inputs={"location": (0.0, 0.0, 0.0)},
    reads="the deformed position as three numbers",
    check_result=_vector3,
    off_axis={"location": _OFF_AXIS},
    returns=Vector3,
)

MOMENTUM_SOURCE = CallableContract(
    label="The momentum source func",
    inputs={"locations": _LOCATIONS},
    reads="a list of one source vector of three numbers per location",
    check_result=_list_of(len(_LOCATIONS), is_vector3),
    off_axis={"locations": _OFF_AXIS_LOCATIONS},
    returns=list[Vector3],
)

LEAF_AREA_DENSITY = CallableContract(
    label="The canopy function",
    inputs={"locations": _LOCATIONS, "wall_distances": [0.5, 1.0]},
    reads="a list of one leaf area density per location",
    check_result=_list_of(len(_LOCATIONS), _is_number),
    off_axis={"locations": _OFF_AXIS_LOCATIONS},
    returns=list[float],
)

PLACEMENT = CallableContract(
    label="The transform function",
    inputs={"location": (0.0, 0.0, 0.0)},
    reads="the placed position as three numbers",
    check_result=_vector3,
    off_axis={"location": _OFF_AXIS},
    returns=Vector3,
)

CFL_RAMP = CallableContract(
    label="The cfl ramp function",
    inputs={"cycle": 1, "current_cfl": 1.0},
    reads=(
        "the new CFL as a number, or a dict of any of 'cfl', 'cfl_turbulence' "
        "and 'cfl_coarse'"
    ),
    check_result=_ramp_result,
)

ForceTransformFunction = FORCE_TRANSFORM.annotation
MeshDeformationFunction = MESH_DEFORMATION.annotation
MomentumSourceFunction = MOMENTUM_SOURCE.annotation
LeafAreaDensityFunction = LEAF_AREA_DENSITY.annotation
PlacementFunction = PLACEMENT.annotation
