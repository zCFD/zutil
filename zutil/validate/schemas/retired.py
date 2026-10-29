"""Error text for retired and not yet supported types and settings.

Keys are ``(context, name)``. ``name`` is a type tag or a setting in schema
spelling. ``context`` is where the error sits: ``"fluid zone"`` for a fluid
zone's type, ``"solver"`` for a setting of the solver block, ``"model"`` for
a setting of a model, ``"transforms"`` for a setting of a model's
``transforms``, ``"motion"`` for its ``motion``, ``"translating"`` for a
translating ``motion``, ``"rotating zone"`` for a rotating fluid zone,
``"rotation"`` for any rotation block, ``"outflow"`` for an outflow boundary
condition, ``"farfield"`` for a farfield boundary condition, ``"target"`` for a
farfield's ``target``, ``"target alpha"`` for its ``alpha``, ``"output
settings"`` for a setting of any ``output_settings``, and ``"*"`` for
anywhere. In the text, ``{owner}`` is the name of the entry holding the error
and ``{key}`` the setting as decks spell it.
"""

from .callables import PLACEMENT
from .motion_schemas import ROTATION_SETTINGS, TRANSLATION_SETTINGS

_ORIGIN = (
    "'{key}' is not a setting here: the centre of rotation is 'origin', as on "
    "every rotation."
)


def _frame(kind: str, settings: str) -> str:
    return (
        f"Fluid zone '{{owner}}' has type '{kind} frame', which is not a "
        "fluid-zone type. A frame is the motion of its mesh, given in the "
        "model's 'transforms': "
        f'"transforms": {{"motion": {{"type": "{kind}", {settings}, '
        '"moving mesh": False}}. A frame covers the whole mesh.'
    )


_ROTATING_ZONE = (
    '"fluid zones": {"<name>": {"type": "rotating", "cell zones": [...], '
    f"{ROTATION_SETTINGS}}}}}"
)

_STILL_MESH = (
    "'{key}' is not a setting of a rotating fluid zone: it is a frame on a still "
    "mesh. A mesh that moves through the run is given in the model's "
    '\'transforms\', as "transforms": {"motion": {"type": "rotating", '
    f'{ROTATION_SETTINGS}, "moving mesh": True}}}}.'
)

_FULL_RATE = (
    "'{key}' is not a setting of a rotating fluid zone: it is a frame on a still "
    "mesh and turns at its full 'omega' from the first step. A mesh that starts "
    "from rest is given in the model's 'transforms', as \"transforms\": "
    f'{{"motion": {{"type": "rotating", {ROTATION_SETTINGS}, "moving mesh": True, '
    '"ramp time": ...}}.'
)


def _moving(kind: str, settings: str) -> str:
    return (
        f"Fluid zone '{{owner}}' has type '{kind}', which is not a fluid-zone "
        "type. A mesh's motion is given in the model's 'transforms': "
        f'"transforms": {{"motion": {{"type": "{kind}", {settings}, '
        "\"moving mesh\": ...}}, with 'moving mesh' True for a mesh that moves "
        "through the run, or False for a frame."
    )


RETIRED_TYPES = {
    ("fluid zone", "translating"): _moving("translating", TRANSLATION_SETTINGS),
    ("fluid zone", "rotating frame"): (
        "Fluid zone '{owner}' has type 'rotating frame', which is not a "
        "fluid-zone type. A rotating frame on part of the mesh is a 'rotating' "
        f"fluid zone on its cell zones: {_ROTATING_ZONE}. A rotating frame on the "
        "whole mesh is given in the model's 'transforms': "
        f'"transforms": {{"motion": {{"type": "rotating", {ROTATION_SETTINGS}, '
        '"moving mesh": False}}.'
    ),
    ("fluid zone", "translating frame"): _frame("translating", TRANSLATION_SETTINGS),
}

RETIRED_SETTINGS = {
    ("solver", "create_overset_halos"): (
        "'create overset halos' is not a setting of the solver: it is a setting "
        "of each overset interface, as "
        '"boundary conditions": {"<name>": {"type": "interface", "kind": '
        '"overset", "create overset halos": ...}}. Every overset interface on '
        "one model gives the same value."
    ),
    ("model", "motion"): (
        "Model '{owner}' sets 'motion', which is not a setting of a model: a "
        "mesh's motion is given in its 'transforms', as "
        '"transforms": {"motion": {...}}. Set \'moving mesh\': True in it for a '
        "mesh that moves through the run."
    ),
    ("transforms", "transform_matrix"): (
        "'transform matrix' is not a setting of 'transforms': the placement is "
        "'transform', a 4x4 matrix or a function "
        f"{PLACEMENT.definition('place')}. 'transform' acts after 'scale', so a "
        "matrix's translation is in metres. If the deck also sets 'scale', give "
        "the translation in metres, or for a non-uniform scale use "
        "S @ M @ inv(S), with S = np.diag([*scale, 1])."
    ),
    ("output settings", "forces"): (
        "'forces' is not a setting of 'output settings': forces are reported "
        'from \'report\', as "report": {"forces": {"<name>": {"zones": [...]}}}.'
    ),
    ("motion", "move_mesh"): (
        "'move mesh' is not a setting of 'motion': the flag is 'moving mesh'."
    ),
    ("motion", "zones"): (
        "'zones' is not a setting of 'motion': 'motion' covers the whole mesh. "
        "A rotating frame on part of the mesh is a 'rotating' fluid zone on its "
        f"cell zones: {_ROTATING_ZONE}. A region that moves through the run is a "
        "mesh of its own, joined to the rest by a sliding or overset interface."
    ),
    ("motion", "ramp"): (
        "'ramp' is not a setting of 'motion': a mesh that starts from rest is "
        "given 'ramp time', the seconds to reach its full rate, with 'moving "
        "mesh': True."
    ),
    ("*", "translation_function"): (
        "'translation function' is not a setting: a control deck carries no "
        "functions. Give 'velocity', the full value in m/s, and 'ramp time', "
        "the seconds to reach it from rest."
    ),
    ("*", "centre"): _ORIGIN,
    ("rotation", "center"): _ORIGIN,
    ("translating", "rotate_nodes"): (
        "'rotate nodes' is not a setting of translating 'motion': a translating "
        "mesh always moves its nodes."
    ),
    ("outflow", "total_temperature_ratio"): (
        "Outflow '{owner}' sets '{key}', which is not a setting of an outflow: "
        "the solver sets no temperature at an outlet. Total temperature is set "
        "on an inflow."
    ),
    ("farfield", "driver"): (
        "Farfield '{owner}' sets 'driver', which is not a setting of a farfield: "
        "a farfield is trimmed to a lift or drag coefficient by its 'target', as "
        '"target": {"alpha": {"cl": 0.5}}.'
    ),
    ("target", "beta"): (
        "'beta' cannot be trimmed yet: a target turns 'alpha' only, as "
        '"target": {"alpha": {"cl": 0.5}}.'
    ),
    ("target", "mach"): (
        "'mach' cannot be trimmed yet: a target turns 'alpha' only, as "
        '"target": {"alpha": {"cl": 0.5}}.'
    ),
    **{
        ("target alpha", name): (
            "'{key}' cannot be a target yet: a target trims 'cl' or 'cd', as "
            '"alpha": {"cl": 0.5}.'
        )
        for name in ("cy", "lift", "drag", "side_force", "cm", "column")
    },
    ("*", "zones"): (
        "'zones' is not a setting here: 'zones' names boundary faces. A fluid "
        "zone selects its cells with 'cell zones', the mesh cell zone numbers, or "
        "'definition', a closed-surface VTP file. A rotating region of the mesh "
        f"is a 'rotating' fluid zone: {_ROTATING_ZONE}."
    ),
    ("rotating zone", "moving_mesh"): _STILL_MESH,
    ("rotating zone", "move_mesh"): _STILL_MESH,
    ("rotating zone", "ramp"): _FULL_RATE,
    ("rotating zone", "ramp_time"): _FULL_RATE,
    ("rotating zone", "definition"): (
        "'definition' is not a setting of a rotating fluid zone: it selects its "
        "cells with 'cell zones', the mesh cell zone numbers, as "
        '"cell zones": [...].'
    ),
}
