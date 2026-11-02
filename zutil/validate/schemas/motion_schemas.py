from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from .base_types import (
    UNSET,
    Advises,
    AnyOf,
    ConditionalChoice,
    ConditionalDefault,
    ConditionalOptional,
    Rotation,
    Vector3d,
    When,
    zCFD_Field,
)

ORDER = "Coordinates go through 'scale', then 'transform', then 'motion'."
ROTATION_SETTINGS = '"axis": [...], "origin": [...], "omega": ...'
TRANSLATION_SETTINGS = '"velocity": [...]'

_MOVING_MESH_NEEDS_UNSTEADY_INTERFACE = When(
    {
        "<mesh>.boundary_conditions.*.kind": ("overset", "sliding"),
        "solver.time_settings.type": ("unsteady",),
        "solver.solver_settings.type": (UNSET, "compressible"),
    },
    text="the model has an overset or sliding interface and the run is unsteady and compressible",
    hint=(
        "For a steady rotor, or an incompressible run, set 'moving mesh': False "
        "to solve the flow in the moving frame."
    ),
)

_ONLY_A_MOVING_MESH = "'moving mesh' is True"

_RAMP_NEEDS_MOVING_MESH = When(
    {"<self>.moving_mesh": (True,)},
    text=_ONLY_A_MOVING_MESH,
    hint="A frame runs at its full rate from the first step: remove 'ramp time'.",
)

_ROTATE_NODES_NEEDS_MOVING_MESH = When(
    {"<self>.moving_mesh": (True,)},
    text=_ONLY_A_MOVING_MESH,
    hint="A frame never moves the nodes: remove 'rotate nodes'.",
)

_IDLE_MOTION = Advises(
    AnyOf(
        When({"<self>.motion.moving_mesh": (True,), "<self>.motion.omega": (0.0,)}),
        When(
            {
                "<self>.motion.moving_mesh": (True,),
                "<self>.motion.velocity": ([0.0, 0.0, 0.0],),
            }
        ),
    ),
    "'motion' has a zero rate, so the mesh never moves but its interfaces are "
    "re-coupled every step, which slows the run: give a non-zero 'omega' or "
    "'velocity', or remove 'motion'.",
)


class _RigidMotion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    moving_mesh: ConditionalChoice[  # type: ignore[valid-type]
        bool, {True: _MOVING_MESH_NEEDS_UNSTEADY_INTERFACE}
    ] = zCFD_Field(
        False,
        title="Moving Mesh",
        description=(
            "True moves the nodes through an unsteady run. False keeps the mesh "
            "still and solves the flow in the moving frame (MRF)."
        ),
    )
    ramp_time: ConditionalOptional[  # type: ignore[valid-type]
        PositiveFloat, _RAMP_NEEDS_MOVING_MESH
    ] = zCFD_Field(
        None,
        title="Ramp Time",
        units="s",
        description=(
            "Physical time over which a moving mesh rises linearly from rest to "
            "its full rate; unset starts at full rate."
        ),
    )


class RotatingMotion(Rotation, _RigidMotion):
    """Rigid rotation of the mesh, or a rotating frame on it."""

    type: Literal["rotating"]
    rotate_nodes: ConditionalDefault[  # type: ignore[valid-type]
        bool, _ROTATE_NODES_NEEDS_MOVING_MESH
    ] = zCFD_Field(
        False,
        advanced=True,
        title="Rotate Nodes",
        description=(
            "Rotate the mesh nodes every time step, instead of carrying the rotation as "
            "an angle that the overset mapper and sliding supermesh apply."
        ),
    )


class TranslatingMotion(_RigidMotion):
    """Rigid translation of the mesh, or a translating frame on it."""

    type: Literal["translating"]
    velocity: Vector3d = zCFD_Field(
        ..., title="Velocity", units="m/s", description="Full translation velocity."
    )


Motion = Annotated[
    Union[RotatingMotion, TranslatingMotion], Field(discriminator="type")
]
