"""Fluid zones, mesh-deformation zones and mesh placement (patterns Z1 to Z14)."""

from __future__ import annotations

import fnmatch
import re
from typing import Callable, Optional

import numpy as np

from zutil.validate.schemas.callables import (
    LEAF_AREA_DENSITY,
    MESH_DEFORMATION,
    MOMENTUM_SOURCE,
    PLACEMENT,
    CallableContract,
)
from zutil.validate.schemas.motion_schemas import (
    ROTATION_SETTINGS,
    TRANSLATION_SETTINGS,
)

from ..model import Conversion, Expr, Leaf, V2Path, Value, norm_key, norm_path
from ._functions import port_function
from ._labels import bc_labels

OWNS = ["fz *", "tr *", "scale", "mesh transform matrix", "mesh transform function"]
ORDER = 50

TR_DEFAULT_TYPE = "rbf"
FRAME_TYPES = ("rotating", "translating")
ROTATION_KEYS = ("axis", "origin", "omega")
SELECTION_KEYS = ("zone", "def", "definition")
MOTION_HAS_NO_ZONES = (
    "v1 moved the whole mesh whatever 'zone' listed, so a moving mesh's "
    "'motion' takes no 'zones'."
)
ROTATING_MOTION = f'{{"type": "rotating", {ROTATION_SETTINGS}, "moving mesh": True}}'
MOTION_SHAPES = (
    f"{ROTATING_MOTION} or "
    f'{{"type": "translating", {TRANSLATION_SETTINGS}, "moving mesh": True, '
    '"ramp time": ...}'
)
V1_PLACEMENT_NOTE = (
    "v1 never placed the mesh with 'mesh transform function': it looked the "
    "function up under an empty key, only to pick partition cells, and before "
    "'scale'. v2 applies 'transforms > transform' to every node, after 'scale'."
)
SCALE_ORDER_NOTE = (
    "v2 applies 'transforms > scale' before 'transforms > transform', and v1 "
    "applied the matrix first, so the matrix is rewritten as "
    "scale * matrix * scale^-1 to keep the mesh where v1 placed it."
)
V1_TRANSLATION_CALL = 'func(**{"time": t, "time step": dt})'
V1_AFFINE_CALL = 'func(**{"dt": dt, "real time": t})'
V1_ZONE_RAMP = (
    "v1 never read 'ramp' on a fluid zone: the zone turned at its full "
    "'omega' from the first step, as a v2 rotating fluid zone does"
)
CONTROLLER_TYPES = ("tsr curve", "fixed", "schedule")

DISC_SECTIONS = {
    "discretisation": ("type", "number of elements"),
    "model": (
        "type",
        "kind",
        "aerofoil positions",
        "blade chord",
        "blade twist",
        "number of sections",
        "number of blades",
        "power",
        "power model",
        "power curve",
        "power coefficient",
        "thrust coefficient",
        "thrust coefficient curve",
        "tip loss correction",
        "tip loss correction radius",
    ),
    "controller": (
        "type",
        "pitch",
        "omega",
        "tip speed ratio",
        "tip speed ratio curve",
    ),
    "geometry": ("centre", "inner radius", "outer radius", "normal", "up"),
}
DISC_FLAT = (
    "name",
    "rotation direction",
    "reference plane",
    "reference point",
    "verbose",
    "update frequency",
)

MESH_DEFORMATION_KEYS = (
    "basis",
    "base point fraction",
    "power",
    "fixed zones",
    "n nearest",
    "deformation distance",
    "blending stiffness",
)


def convert(c: Conversion) -> None:
    for key in c.v1.children():
        folded = norm_key(key)
        if fnmatch.fnmatchcase(folded, "fz *") or fnmatch.fnmatchcase(folded, "tr *"):
            _Zone(c, key).convert()
    _placement(c)
    _frames(c)


def motion_path(c: Conversion) -> V2Path:
    """Where the motion of the model's mesh is written."""
    return ("model", c.model, "transforms", "motion")


def has_overset_boundary(c: Conversion) -> bool:
    """Whether the v1 deck has an overset boundary, the only v1 interface that moves."""
    return any(
        str(c.v1.value(label, "type", default="")).lower() == "overset"
        for label in bc_labels(c)
    )


def fluid_zone_path(c: Conversion, key: str) -> V2Path:
    return ("model", c.model, "fluid zones", key)


def _covers_whole_mesh(c: Conversion, zone: Leaf) -> bool:
    mesh = c.mesh
    return (
        mesh is not None
        and bool(mesh.cell_zones)
        and set(mesh.cell_zones) <= set(zone.value)
    )


def rotating_fluid_zone(c: Conversion, key: str) -> bool:
    """Whether v1 zone ``key`` on a still mesh is a v2 rotating fluid zone.

    A rotating zone is one unless the mesh has an overset boundary, whose mesh
    turns as a whole, or a mesh was read and its 'zone' list alone covers every
    cell zone; either makes it the whole-mesh frame.
    """
    z = _Zone(c, key)
    if z.kind() != "rotating" or has_overset_boundary(c):
        return False
    zone = c.v1.leaf(key, "zone")
    defined = c.v1.has(key, "def") or c.v1.has(key, "definition")
    return zone is None or defined or not _covers_whole_mesh(c, zone)


def _cell_zones_known(c: Conversion) -> str:
    if c.mesh is not None and c.mesh.cell_zones is not None:
        return f"The mesh has cell zones {c.mesh.cell_zones}."
    return (
        "update_input reads the cell zones from the mesh given with --mesh, "
        "and none was read."
    )


def _text(leaf: Leaf) -> str:
    text = leaf.source if leaf.source is not None else repr(leaf.value)
    return text if re.fullmatch(r"[\w.]+", text) else f"({text})"


def _function_name(leaf: Leaf) -> str:
    name = getattr(leaf.value, "__name__", "")
    if name.isidentifier():
        return name
    return leaf.source or repr(leaf.value)


def moving_zones(c: Conversion) -> list[str]:
    """The labels of the v1 rotating and translating fluid zones, in deck order."""
    return [
        key
        for key in c.v1.children()
        if fnmatch.fnmatchcase(norm_key(key), "fz *")
        and norm_key(c.v1.value(key, "type", default="")) in FRAME_TYPES
    ]


def convert_motion(c: Conversion, key: str, moving_mesh: Leaf) -> None:
    """Write v1 moving zone ``key`` as the motion of the model's moving mesh."""
    _placement(c)
    z = _Zone(c, key, motion_path(c))
    kind = z.kind()
    c.move(z.leaf("type"), z.path("type"), value=Value(kind))
    c.move(moving_mesh, z.path("moving mesh"), respelled=True)
    selection = [lf for lf in (z.leaf(k) for k in SELECTION_KEYS) if lf is not None]
    for leaf in selection:
        c.drop(leaf, "a moving mesh moves the whole mesh")
    c.note(MOTION_HAS_NO_ZONES, selection[0].path if selection else None)
    if kind == "rotating":
        _rotation(z)
        return
    _velocity(z)
    function = z.leaf("translation function")
    if function is not None:
        name = _function_name(function)
        c.decide(
            [function],
            z.path("ramp time"),
            f"A v2 control deck carries no functions. '{name}' was the translation "
            f"function of fluid zone '{key}'. Write the motion in 'transforms > "
            "motion' as 'velocity', its full value in m/s, and, if the body starts "
            "from rest with a linear rise, 'ramp time', the seconds it takes to "
            "reach that velocity. Any other profile cannot be expressed.",
            details=(
                f"v1 called {name} as {V1_TRANSLATION_CALL} every step and moved the "
                "mesh at the velocity it returned. The draft writes the v1 vector "
                "as 'velocity'."
            ),
        )


def _rotation(z: _Zone) -> None:
    for rel in ROTATION_KEYS:
        z.move((rel,))
    for spelling in ("centre", "center"):
        z.move((spelling,), ("origin",))


def _frames(c: Conversion) -> None:
    keys = [key for key in moving_zones(c) if not c.claimed(c.v1.leaf(key, "type"))]
    decided = motion_path(c) + ("moving mesh",) in c.v2
    motions = [k for k in keys if decided or not rotating_fluid_zone(c, k)]
    if motions and len(keys) > 1:
        _several_frames(c, keys)
    elif motions:
        _frame(_Zone(c, motions[0], motion_path(c)))
    else:
        for key in keys:
            _rotating_zone(_Zone(c, key))


def _rotating_zone(z: _Zone) -> None:
    c = z.c
    c.move(
        z.leaf("type"),
        z.path("type"),
        value=Value("rotating"),
        reason="a v1 rotating zone on a still mesh is a rotating fluid zone",
    )
    _rotation(z)
    zone = z.leaf("zone")
    defined = [lf for lf in (z.leaf("def"), z.leaf("definition")) if lf is not None]
    if zone is not None and not defined:
        z.move(("zone",), ("cell zones",))
    else:
        given = (
            "rotates the cells its definition selects"
            if defined
            else "gives neither 'zone' nor 'def'"
        )
        c.decide(
            [lf for lf in (zone, *defined) if lf is not None],
            z.path("cell zones"),
            f"Fluid zone {z.key} {given}, and a v2 rotating fluid zone selects "
            "its cells by 'cell zones' only. Which cell zones does it rotate?",
            details=(
                f"{_cell_zones_known(c)} The region that turns must be meshed as "
                "cell zones of its own."
            ),
        )
    for leaf in c.v1.leaves_under(z.key, "ramp"):
        if not c.claimed(leaf):
            c.drop(leaf, V1_ZONE_RAMP)


def _several_frames(c: Conversion, keys: list[str]) -> None:
    leaves = [lf for key in keys for lf in c.v1.leaves_under(key) if not c.claimed(lf)]
    settings = "; ".join(
        f"{key}: "
        + ", ".join(
            f"{' > '.join(lf.keys[1:])} {lf.value!r}" for lf in c.v1.leaves_under(key)
        )
        for key in keys
    )
    c.decide(
        leaves,
        motion_path(c),
        f"v2 gives a mesh one 'motion', and fluid zones {', '.join(keys)} each "
        f"move a frame on mesh {c.model}. Which is the motion of the mesh? Split "
        "the mesh into one model per frame to keep them all.",
        details=f"The v1 zones are {settings}.",
    )


class _Zone:
    def __init__(self, c: Conversion, key: str, base: Optional[V2Path] = None) -> None:
        self.c = c
        self.key = key
        self.base: V2Path = base if base is not None else fluid_zone_path(c, key)

    def leaf(self, *rel: str) -> Optional[Leaf]:
        found = self.c.v1.leaf(self.key, *rel)
        if found is None or self.c.claimed(found):
            return None
        return found

    def path(self, *rel: str) -> V2Path:
        return self.base + rel

    def move(self, rel: tuple[str, ...], to: Optional[tuple[str, ...]] = None) -> None:
        leaf = self.leaf(*rel)
        if leaf is None:
            return
        target = to if to is not None else norm_path(*rel)
        self.c.move(leaf, self.path(*target), respelled=to is not None)

    def port(
        self,
        rel: tuple[str, ...],
        to: tuple[str, ...],
        contract: CallableContract,
    ) -> None:
        leaf = self.leaf(*rel)
        if leaf is not None:
            port_function(
                self.c,
                leaf,
                self.path(*to),
                contract,
                respelled=to != norm_path(*rel),
            )

    def selection(self) -> None:
        zone = self.leaf("zone")
        defined = self.leaf("def") or self.leaf("definition")
        if zone is not None and defined is not None and zone.value == []:
            self.c.drop(
                zone, "an empty zone list selects no cells beside the definition"
            )
        self.move(("zone",), ("zones",))
        self.move(("def",), ("definition",))
        self.move(("definition",))

    def dropped_as_whole_mesh(self, zone: Leaf, covered_by: str) -> bool:
        if not _covers_whole_mesh(self.c, zone):
            return False
        self.c.drop(zone, "the zones are the whole mesh")
        self.c.note(
            f"Fluid zone {self.key} covers the whole mesh, and so does {covered_by}.",
            zone.path,
        )
        return True

    def source_selection(self) -> None:
        c = self.c
        zone = self.leaf("zone")
        selectors = ("def", "field") if self.kind() == "canopy" else ("def",)
        chosen = next((k for k in selectors if c.v1.has(self.key, k)), None)
        self.move(("def",), ("definition",))
        if zone is None or chosen is not None:
            self.move(("definition",))
        if zone is None:
            return
        if chosen is not None:
            c.drop(zone, f"v1 selected the cells by '{chosen}' and ignored 'zone'")
            return
        unread = self.leaf("definition")
        if self.kind() == "disc":
            ignored = (
                f" v1 read no 'definition', so {unread.value!r} selected no cells."
                if unread is not None
                else ""
            )
            c.decide(
                [lf for lf in (zone, unread) if lf is not None],
                self.path("definition"),
                f"Disc {self.key} acts on cell zones {zone.value}, and a v2 disc "
                "selects its cells with its geometry and 'definition'. Give a "
                "'definition' VTP that selects the disc region.",
                details=(
                    f"{_cell_zones_known(c)} A 'definition' is a closed-surface "
                    f"VTP file enclosing the disc.{ignored}"
                ),
            )
            return
        if unread is not None:
            c.drop(unread, "v1 read no 'definition', so it selected no cells")
        if self.dropped_as_whole_mesh(zone, "a source without 'cell zones'"):
            return
        self.move(("zone",), ("cell zones",))

    def kind(self) -> str:
        default = (
            TR_DEFAULT_TYPE if fnmatch.fnmatchcase(norm_key(self.key), "tr *") else ""
        )
        return norm_key(self.c.v1.value(self.key, "type", default=default))

    def convert(self) -> None:
        handler = _HANDLERS.get(self.kind())
        if handler is not None:
            handler(self)


def _frame(z: _Zone) -> None:
    c = z.c
    kind = z.kind()
    c.move(
        z.leaf("type"),
        z.path("type"),
        value=Value(kind),
        reason=f"a v1 {kind!r} zone on a still mesh is a frame, 'moving mesh' False",
    )
    if z.path("moving mesh") not in c.v2:
        c.set(z.path("moving mesh"), Value(False))
    _frame_zones(z)
    if kind == "rotating":
        _rotation(z)
        return
    function = z.leaf("translation function")
    if function is None:
        _velocity(z)
        return
    name = _function_name(function)
    vector = [lf for lf in (z.leaf("vector"), z.leaf("mach")) if lf is not None]
    c.decide(
        [function, *vector],
        z.path("velocity"),
        f"A v2 control deck carries no functions. '{name}' was the translation "
        f"function of fluid zone '{z.key}'. A frame has a fixed velocity: give "
        "'velocity' in 'transforms > motion' in m/s.",
        details=(
            f"v1 called {name} as {V1_TRANSLATION_CALL} every step and moved the "
            "frame at the velocity it returned. The v1 vector was "
            f"{c.v1.value(z.key, 'vector')!r}."
        ),
    )


def _frame_zones(z: _Zone) -> None:
    c = z.c
    zone = z.leaf("zone")
    defined = z.leaf("def") or z.leaf("definition")
    if zone is None and defined is None:
        return
    if defined is None:
        if z.dropped_as_whole_mesh(zone, "a frame in 'transforms > motion'"):
            return
        selected = f"cell zones {zone.value}"
    else:
        selected = "the cells its definition selects"
    known = _cell_zones_known(c)
    verb, region = (
        ("rotates", "rotating")
        if z.kind() == "rotating"
        else ("translates", "translating")
    )
    c.decide(
        [lf for lf in (zone, defined) if lf is not None],
        z.path("zones"),
        f"Fluid zone {z.key} {verb} {selected} only, and 'transforms > motion' "
        "covers the whole mesh. Should the frame cover the whole mesh, or should "
        f"the {region} region be a mesh of its own?",
        options=(
            "cover the whole mesh",
            "give the region its own mesh, joined by a sliding or overset interface",
        ),
        details=(
            f"{known} To cover the whole mesh, remove 'zones'. Otherwise split "
            "the region into a mesh of its own, with its own model, joined to "
            "the rest by a sliding or overset interface."
        ),
    )


def _mach_velocity(c: Conversion, vector: Leaf, mach: Leaf) -> Optional[Expr]:
    reference = c.v1.value("reference")
    material = c.v1.value("material")
    if not isinstance(reference, str) or not isinstance(material, str):
        return None
    temperature = c.v1.leaf(reference, "temperature")
    gamma = c.v1.leaf(material, "gamma")
    gas_constant = c.v1.leaf(material, "gas constant")
    if temperature is None or gamma is None or gas_constant is None:
        return None
    sound_speed = (gamma.value * gas_constant.value * temperature.value) ** 0.5
    length = float(np.linalg.norm(vector.value))
    if length == 0.0:
        return Expr("[0.0, 0.0, 0.0]", [0.0, 0.0, 0.0])
    speed = mach.value * sound_speed / length
    unit = "x" if np.isclose(length, 1.0, rtol=0.0, atol=1e-12) else f"x / {length!r}"
    text = (
        f"[{unit} * {_text(mach)} * ({_text(gamma)} * {_text(gas_constant)} * "
        f"{_text(temperature)}) ** 0.5 for x in {_text(vector)}]"
    )
    return Expr(text, [x * speed for x in vector.value])


def _velocity(z: _Zone) -> None:
    c = z.c
    vector, mach = z.leaf("vector"), z.leaf("mach")
    if vector is not None and mach is not None:
        velocity = _mach_velocity(c, vector, mach)
        if velocity is not None:
            c.fold(
                [vector, mach],
                z.path("velocity"),
                velocity,
                respelled=True,
                reason="mach scales the unit vector by the speed of sound",
            )
            c.note(
                f"{z.key} velocity is mach times the speed of sound of the "
                f"reference IC, along vector",
                vector.path,
            )
        else:
            c.decide(
                [vector, mach],
                z.path("velocity"),
                f"What is the dimensional velocity of translating zone {z.key}?",
                details=(
                    "v1 scaled the unit vector by mach times the speed of sound "
                    "of the reference IC. The reference IC temperature, gamma "
                    "and gas constant are not all in the deck, so the product "
                    "cannot be written."
                ),
            )
    else:
        z.move(("vector",), ("velocity",))


def _canopy(z: _Zone) -> None:
    z.move(("type",))
    defined, field = z.leaf("def"), z.leaf("field")
    if defined is not None and field is not None:
        z.c.decide(
            [defined, field],
            z.path("definition"),
            f"Canopy {z.key} gives both 'def' and 'field', and a v2 canopy selects "
            "its cells with one of 'cell zones', 'definition' and 'field'. Which "
            "selects the cells?",
            options=("definition", "field"),
            details=(
                f"'def' is {_function_name(defined)} and 'field' is "
                f"{_function_name(field)}. Write the chosen one, as 'definition' "
                "or 'field', in place of this key."
            ),
        )
    z.source_selection()
    z.port(("func",), ("function",), LEAF_AREA_DENSITY)
    for key in ("field", "cd", "beta p", "beta d", "ceps 4", "ceps 5"):
        z.move((key,))


def _momentum(z: _Zone) -> None:
    z.move(("type",))
    z.source_selection()
    z.port(("func",), ("func",), MOMENTUM_SOURCE)


def _porous(z: _Zone) -> None:
    z.move(("type",))
    z.source_selection()
    z.move(("alpha",))
    z.move(("c2",))


def _abl(z: _Zone) -> None:
    c = z.c
    leaves = [lf for lf in c.v1.leaves_under(z.key) if not c.claimed(lf)]
    c.decide(
        leaves,
        z.path("type"),
        f"What does v1 abl zone {z.key} become in v2?",
        details=(
            "A v1 abl zone took the momentum-source fields. The v2 abl zone is a "
            "different model, with roughness length, friction velocity, "
            "reference height and wind direction, and applies to the whole "
            "mesh."
        ),
    )


def _affine(z: _Zone) -> None:
    c = z.c
    leaves = [lf for lf in c.v1.leaves_under(z.key) if not c.claimed(lf)]
    function = z.leaf("transform function")
    name = _function_name(function) if function is not None else "func"
    c.decide(
        leaves,
        z.path("type"),
        f"v2 has no affine transform zone. What replaces {z.key}?",
        details=(
            f"v1 called {name} as {V1_AFFINE_CALL} every step and read back a "
            "4 x 4 numpy array, which moved the cells of the zone. Prescribed "
            "motion is declarative in v2. A rigid motion of the whole mesh is the "
            f"'motion' in the model's 'transforms', {MOTION_SHAPES}. A deformation "
            "is a mesh deformation transform whose func is "
            f"{MESH_DEFORMATION.definition(name)}. A fixed placement is the "
            "model's 'transforms > transform', a 4 x 4 matrix or a function "
            f"{PLACEMENT.definition('place')}."
        ),
    )


def _disc(z: _Zone) -> None:
    c = z.c
    z.move(("type",))
    z.source_selection()
    for key in DISC_FLAT:
        z.move((key,))
    status = z.leaf("status")
    if status is not None:
        if norm_key(status.value) == "on":
            c.drop(status, "v2 has no status; a disc is always on")
        else:
            c.decide(
                [status],
                z.path("status"),
                f"Disc {z.key} is switched off in v1. v2 has no status key. "
                "Should the zone be removed?",
                details="v2 discs are always active.",
            )
    for section, keys in DISC_SECTIONS.items():
        for key in keys:
            if (section, key) == ("controller", "type"):
                _controller_type(z)
            else:
                z.move((section, key))
    _aerofoils(z)


def _controller_type(z: _Zone) -> None:
    leaf = z.leaf("controller", "type")
    if leaf is None:
        return
    if norm_key(leaf.value) == "none":
        z.c.decide(
            [leaf],
            z.path("controller", "type"),
            f"Disc {z.key} has controller type 'none'. Which v2 controller does it use?",
            options=CONTROLLER_TYPES,
            details="v2 has no controller type 'none'.",
        )
    else:
        z.move(("controller", "type"))


def _aerofoils(z: _Zone) -> None:
    prefix = (z.key, "model", "aerofoils")
    for leaf in z.c.v1.leaves_under(*prefix):
        if z.c.claimed(leaf):
            continue
        rest = leaf.keys[len(prefix) :]
        target = (rest[0],) + norm_path(*rest[1:])
        z.c.move(leaf, z.path("model", "aerofoils", *target))


def _mesh_deformation(z: _Zone) -> None:
    c = z.c
    kind = z.kind()
    z.selection()
    type_leaf = z.leaf("type")
    if kind == "rbf":
        tol = z.leaf("tol")
        c.decide(
            [lf for lf in (type_leaf, tol) if lf is not None],
            z.path("type"),
            f"v2 has no greedy RBF for {z.key}. Use the full RBF transform?",
            options=("rbf transform",),
            details=(
                "The full RBF is 'rbf transform' with base point fraction 1.0, "
                "which the greedy RBF approximates. The draft writes base point "
                "fraction 1.0 and drops the greedy tolerance."
            ),
        )
        if z.leaf("base point fraction") is None:
            c.set(z.path("base point fraction"), Value(1.0))
    else:
        v2_type = "idw transform" if kind == "idw" else "rbf transform"
        c.move(
            type_leaf,
            z.path("type"),
            value=Value(v2_type),
            respelled=True,
            reason=f"v1 type {kind!r} is {v2_type!r} in v2",
        )
    z.move(("alpha",), ("support radius",))
    z.port(("func",), ("func",), MESH_DEFORMATION)
    for key in MESH_DEFORMATION_KEYS:
        z.move((key,))


def _unclaimed(c: Conversion, key: str) -> Optional[Leaf]:
    leaf = c.v1.leaf(key)
    return None if leaf is None or c.claimed(leaf) else leaf


def _scaled_matrix(scale: Optional[Leaf], matrix: Leaf) -> Optional[list]:
    """v1's scale @ matrix as a matrix acting after the scale, or None if they commute."""
    if scale is None:
        return None
    try:
        s = np.array([*scale.value, 1.0], dtype=float)
        m = np.array(matrix.value, dtype=float)
    except (TypeError, ValueError):
        return None
    if s.shape != (4,) or m.shape != (4, 4) or not np.all(s > 0):
        return None
    rewritten = m * s[:, None] / s[None, :]
    if np.array_equal(rewritten, m):
        return None
    return [[float(f"{x:.15g}") for x in row] for row in rewritten]


def _placement(c: Conversion) -> None:
    base = ("model", c.model, "transforms")
    scale = _unclaimed(c, "scale")
    if scale is not None:
        c.move(scale, base + ("scale",))
    matrix = _unclaimed(c, "mesh transform matrix")
    function = _unclaimed(c, "mesh transform function")
    if function is None:
        if matrix is None:
            return
        rewritten = _scaled_matrix(scale, matrix)
        if rewritten is None:
            c.move(matrix, base + ("transform",), respelled=True)
        else:
            c.move(
                matrix,
                base + ("transform",),
                value=Value(rewritten),
                respelled=True,
                reason="v2 applies scale before transform",
            )
            c.note(SCALE_ORDER_NOTE, matrix.path)
        return
    name = _function_name(function)
    if matrix is None:
        port_function(c, function, base + ("transform",), PLACEMENT, respelled=True)
    else:
        c.decide(
            [matrix, function],
            base + ("transform",),
            "v1 sets both 'mesh transform matrix' and 'mesh transform function'. "
            f"v2 places mesh {c.model} with one 'transform'. Which places it?",
            options=("the matrix", name),
            details=(
                f"{V1_PLACEMENT_NOTE} The function must be "
                f"{PLACEMENT.definition(name)}."
            ),
        )
    c.note(V1_PLACEMENT_NOTE, function.path)


_HANDLERS: dict[str, Callable[[_Zone], None]] = {
    "canopy": _canopy,
    "momentum": _momentum,
    "porous": _porous,
    "abl": _abl,
    "affine transform": _affine,
    "disc": _disc,
    "rbf": _mesh_deformation,
    "rbf multiscale": _mesh_deformation,
    "idw": _mesh_deformation,
}
