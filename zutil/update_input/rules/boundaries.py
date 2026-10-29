"""Boundary conditions, structural models, the sponge layer and overset mapping.

v1 ``BC_N`` blocks become ``model > <label> > boundary conditions > BC_N``.
The deck-wide ``modal model``, ``fsi`` and ``sponge layer damping`` blocks and
the overset ``mapping`` settings move onto the boundary conditions they act on.
Patterns B1 to B20 of the "Deck patterns" section of
``developer_docs/v1-to-v2-conversion.md``.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ._labels import bc_labels, ic_label, move_ic_name
from ..model import Conversion, Expr, Leaf, Source, V2Path, Value, norm_key

OWNS = [
    "bc *",
    "modal model",
    "fsi",
    "sponge layer damping",
    "mapping",
    "create overset halos",
]
ORDER = 40

OVERSET_CODE = 10

WALL_KINDS = {"noslip": "no slip", "wallfunction": "wall function"}
WALL_TYPES = ("wall", "immersed wall")
TRANSFORM_TYPES = {"rbf multiscale": "rbf_multiscale"}

MODAL_KEYS = {
    "file format": "file format",
    "alpha": "support radius",
    "fluid force scaling": "fluid force scaling",
    "recalculate rbfs alpha fraction": "recalculate rbfs alpha fraction",
    "nastran casename": "nastran casename",
    "mode mapping max distance": "mode mapping max distance",
    "modal damping": "modal damping",
    "mode list": "mode list",
    "mode mapping power": "mode mapping power",
    "mode mapping pureidw": "mode mapping pure idw",
    "use n nearest neighbours": "mode mapping n nearest",
    "scale": "scale",
    "newmark alpha": "newmark alpha",
    "newmark delta": "newmark delta",
    "update frequency": "update frequency",
}
MODAL_RBF_KEYS = ("basis", "base point fraction")
MODAL_IDW_KEYS = (
    "power",
    "n nearest",
    "deformation distance",
    "blending stiffness",
    "fixed zones",
)
MODAL_REQUIRED_DEFAULTS = {
    "file format": "nastran",
    "fluid force scaling": 1.0,
    "mode list": [],
}
MODAL_RBF_REQUIRED_DEFAULTS = {"base point fraction": 0.1}
MODAL_IDW_REQUIRED_DEFAULTS = {"power": [3.0, 5.0, 10.0]}

FSI_KEYS = {
    "zone": "zone",
    "fixed zones": "fixed zones",
    "alpha": "support radius",
    "basis": "basis",
    "base point fraction": "base point fraction",
    "tol": "tol",
    "max error": "max error",
    "n nearest": "n nearest",
    "deformation distance": "deformation distance",
    "blending stiffness": "blending stiffness",
    "power": "power",
}


def convert(c: Conversion) -> None:
    labels = [lb for lb in bc_labels(c) if c.v1.leaf(lb) is None]
    zones = {label: _convert_bc(c, label) for label in labels}
    _pair_periodic(c, labels)
    _note_numbering_gap(c, labels)
    _sponge(c, labels)
    walls = {
        lb: z for lb, z in zones.items() if z is not None and _type(c, lb) in WALL_TYPES
    }
    taken: set[str] = set()
    for mm in c.v1.children("modal model"):
        _modal_model(c, mm, walls, taken)
    if c.v1.has("fsi"):
        _generic_fsi(c, walls, taken)
    _mapping(c, labels)
    _overset_halos(c, labels)


def _bc_path(c: Conversion, label: str, *keys: str) -> V2Path:
    return ("model", c.model, "boundary conditions", label) + keys


def _type(c: Conversion, label: str) -> Any:
    value = c.v1.value(label, "type")
    return norm_key(value) if isinstance(value, str) else value


def _label_name(leaf: Leaf, depth: int) -> str:
    return " > ".join(str(k) for k in leaf.keys[:depth])


def _text(leaf: Leaf) -> str:
    if leaf.source is not None:
        return leaf.source
    if callable(leaf.value) and hasattr(leaf.value, "__name__"):
        return leaf.value.__name__
    return repr(leaf.value)


def _block_expr(leaves: list[Leaf], depth: int, value: Any) -> Expr:
    tree: dict = {}
    for leaf in leaves:
        node = tree
        for key in leaf.keys[depth:-1]:
            node = node.setdefault(key, {})
        node[leaf.keys[-1]] = leaf

    def literal(node: dict) -> str:
        items = [
            f"{key!r}: {literal(v) if isinstance(v, dict) else _text(v)}"
            for key, v in node.items()
        ]
        return "{" + ", ".join(items) + "}"

    return Expr(literal(tree), value)


def _park(
    c: Conversion,
    keys: tuple[str, ...],
    question: str,
    options: tuple[str, ...] = (),
    details: str = "",
) -> None:
    """DECIDE on a whole v1 block, kept as written under the unconverted keys."""
    leaves = c.v1.leaves_under(*keys)
    if not leaves:
        return
    depth = len(keys)
    name = _label_name(leaves[0], depth)
    if len(leaves) == 1 and len(leaves[0].keys) == depth:
        value = Source(leaves[0])
    else:
        value = _block_expr(leaves, depth, c.v1.value(*keys))
    c.park(leaves, name, question, options, details, value)


def _zone_ints(value: Any) -> Optional[list[int]]:
    try:
        return sorted(int(z) for z in value)
    except (TypeError, ValueError):
        return None


def _zones(c: Conversion, label: str) -> Optional[list[int]]:
    zone = c.v1.leaf(label, "zone")
    ref = c.v1.leaf(label, "ref")
    path = _bc_path(c, label, "zones")
    if ref is None:
        if zone is not None:
            c.move(zone, path)
            return _zone_ints(zone.value)
        return None
    code = ref.value
    if zone is not None:
        listed = _zone_ints(zone.value) or []
        if c.mesh is None:
            c.move(zone, path)
            c.drop(ref, f"the zone list is kept; face type {code} is not read")
            c.note(
                f"{label} has both 'zone' and 'ref: {code}'. v1 also applied it to "
                f"every zone of face type {code}; run with --mesh to check that "
                "those zones are in the list.",
                ref.path,
            )
            return listed
        typed = sorted(c.mesh.zones_by_type.get(code, []))
        if set(typed) <= set(listed):
            c.move(zone, path)
            c.drop(ref, f"every zone of face type {code} is already in the zone list")
            return listed
        union = sorted(set(listed) | set(typed))
        c.decide(
            [zone, ref],
            path,
            f"Which zones does {label} cover?",
            (str(listed), str(union)),
            f"v1 applied {label} to its zone list {listed} and to every zone of "
            f"face type {code}, which in {c.mesh.path} is {typed}.",
        )
        return None
    if code == OVERSET_CODE:
        c.decide(
            [ref],
            path,
            f"Which zones does the overset boundary condition {label} cover?",
            details=(
                f"The legacy schema gives overset faces the code {OVERSET_CODE}, but "
                "the mesh stores them as 0x14, so the zones cannot be read from the "
                "mesh."
            ),
        )
        return None
    if c.mesh is None:
        c.decide(
            [ref],
            path,
            f"Which zones does {label} cover?",
            details=(
                f"v1 applied {label} to every face of type {code}. Run update_input "
                "with --mesh to read those zones from the mesh, or list them here."
            ),
        )
        return None
    typed = sorted(c.mesh.zones_by_type.get(code, []))
    if not typed:
        c.decide(
            [ref],
            path,
            f"Which zones does {label} cover?",
            details=f"{c.mesh.path} has no boundary faces of type {code}.",
        )
        return None
    c.move(
        ref,
        path,
        Value(typed),
        respelled=True,
        reason=f"the zones of face type {code} in {c.mesh.path}",
    )
    return typed


def _derived_reference(c: Conversion, name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    ic = c.v1.value(name)
    if not isinstance(ic, dict):
        return None
    reference = c.v1.value(name, "reference")
    return reference if isinstance(reference, str) else None


def _condition_to_reference(c: Conversion, label: str) -> None:
    leaf = c.v1.leaf(label, "condition")
    if leaf is None:
        return
    path = _bc_path(c, label, "reference")
    reference = _derived_reference(c, leaf.value)
    if reference is None:
        move_ic_name(c, leaf, path)
        return
    c.move(
        leaf,
        path,
        Value(ic_label(c, reference) or reference),
        respelled=True,
        reason=f"{leaf.value} is derived from {reference}; its ratios go on {label}",
    )


def _move(c: Conversion, label: str, v1_keys: tuple, v2_keys: tuple) -> None:
    leaf = c.v1.leaf(label, *v1_keys)
    if leaf is not None:
        c.move(leaf, _bc_path(c, label, *v2_keys))


def _convert_bc(c: Conversion, label: str) -> Optional[list[int]]:
    zones = _zones(c, label)
    kind = _type(c, label)
    type_leaf = c.v1.leaf(label, "type")
    if type_leaf is None:
        return zones
    if kind == "wall":
        c.move(type_leaf, _bc_path(c, label, "type"))
        _wall(c, label)
    elif kind == "immersed wall":
        c.decide(
            [type_leaf],
            _bc_path(c, label, "type"),
            f"v2 has no 'immersed wall' boundary type: what should {label} be?",
            ("wall",),
            "zCFD no longer supports immersed boundaries. Mesh the body and make "
            f"{label} a wall, or remove the boundary condition.",
        )
        _wall(c, label)
    elif kind in ("farfield", "symmetry"):
        c.move(type_leaf, _bc_path(c, label, "type"))
        _move(c, label, ("kind",), ("kind",))
        condition = c.v1.leaf(label, "condition")
        if condition is not None:
            move_ic_name(c, condition, _bc_path(c, label, "condition"))
    elif kind == "inflow":
        c.move(type_leaf, _bc_path(c, label, "type"))
        _inflow_kind(c, label)
        _condition_to_reference(c, label)
    elif kind == "outflow":
        c.move(type_leaf, _bc_path(c, label, "type"))
        _move(c, label, ("kind",), ("kind",))
        _move(c, label, ("reference radius",), ("reference radius",))
        _condition_to_reference(c, label)
    elif kind == "overset":
        _interface(c, label, type_leaf, "overset")
    elif kind == "periodic":
        _interface(c, label, type_leaf, "periodic")
        _periodic_transform(c, label)
    elif kind == "wallsrc":
        c.decide(
            [type_leaf],
            _bc_path(c, label, "type"),
            f"v2 has no 'wallsrc' boundary type: what should {label} be?",
            details=(
                "v1 'wallsrc' marked a wall-source boundary, which config version 2 "
                "does not support. Choose another type or remove the boundary "
                "condition."
            ),
        )
    return zones


def _interface(c: Conversion, label: str, type_leaf: Leaf, kind: str) -> None:
    c.move(
        type_leaf,
        _bc_path(c, label, "type"),
        Value("interface"),
        respelled=True,
        reason=f"v2 makes {kind} a kind of interface",
    )
    c.set(_bc_path(c, label, "kind"), Value(kind))


def _inflow_kind(c: Conversion, label: str) -> None:
    leaf = c.v1.leaf(label, "kind")
    if leaf is None:
        return
    path = _bc_path(c, label, "kind")
    if leaf.value == "default" and not c.v1.incompressible:
        c.move(
            leaf,
            path,
            Value("pressure"),
            respelled=True,
            reason="v1 treated every compressible inflow that was not massflow as pressure",
        )
        return
    c.move(leaf, path)


def _wall(c: Conversion, label: str) -> None:
    leaf = c.v1.leaf(label, "kind")
    if leaf is not None:
        path = _bc_path(c, label, "kind")
        if leaf.value in WALL_KINDS:
            c.move(leaf, path, Value(WALL_KINDS[leaf.value]), respelled=True)
        else:
            c.move(leaf, path)
    c.move_tree((label, "roughness"), _bc_path(c, label, "roughness"))
    _move(c, label, ("temperature", "field"), ("temperature", "field"))
    scalar = c.v1.leaf(label, "temperature", "scalar")
    if scalar is not None:
        c.move(
            scalar,
            _bc_path(c, label, "temperature"),
            reason="v2 takes the wall temperature as a number",
        )
    normal = c.v1.leaf(label, "immersed normal type")
    if normal is not None:
        c.drop(normal, "zCFD no longer supports immersed boundaries")
    _wall_velocity(c, label)


def _wall_velocity(c: Conversion, label: str) -> None:
    velocity = _bc_path(c, label, "velocity")
    if c.v1.has(label, "v", "linear"):
        c.set(velocity + ("type",), Value("linear"))
        _move(c, label, ("v", "linear", "vector"), ("velocity", "vector"))
        _move(c, label, ("v", "linear", "mach"), ("velocity", "mach"))
    elif c.v1.has(label, "v", "rotating"):
        c.set(velocity + ("type",), Value("rotating"))
        _move(c, label, ("v", "rotating", "axis"), ("velocity", "axis"))
        _move(c, label, ("v", "rotating", "origin"), ("velocity", "origin"))
        _move(c, label, ("v", "rotating", "omega"), ("velocity", "omega"))


def _periodic_transform(c: Conversion, label: str) -> None:
    transform = _bc_path(c, label, "transform")
    for kind, keys in (
        ("linear", ("vector",)),
        ("rotated", ("theta", "axis", "origin")),
    ):
        if c.v1.has(label, "kind", kind):
            c.set(transform + ("type",), Value(kind))
            for key in keys:
                _move(c, label, ("kind", kind, key), ("transform", key))


def _periodic_model(c: Conversion, label: str):
    from zutil.validate.schemas.boundary_conditions_schemas import (
        LinearPeriodicTransform,
        RotatedPeriodicTransform,
    )

    try:
        if c.v1.has(label, "kind", "linear"):
            return LinearPeriodicTransform(
                vector=c.v1.value(label, "kind", "linear", "vector")
            )
        if c.v1.has(label, "kind", "rotated"):
            return RotatedPeriodicTransform(
                **{
                    key: c.v1.value(label, "kind", "rotated", key)
                    for key in ("theta", "axis", "origin")
                }
            )
    except (TypeError, ValueError):
        return None
    return None


def _inverse(c: Conversion, a: str, b: str) -> bool:
    ta, tb = _periodic_model(c, a), _periodic_model(c, b)
    if ta is None or tb is None or type(ta) is not type(tb):
        return False
    return ta.inverse_mismatch(tb, a, b) is None


def _pair_periodic(c: Conversion, labels: list[str]) -> None:
    periodic = [lb for lb in labels if _type(c, lb) == "periodic"]
    if len(periodic) == 2 and _inverse(c, *periodic):
        a, b = periodic
        c.set(_bc_path(c, a, "connection", "boundary"), Value(b))
        c.set(_bc_path(c, b, "connection", "boundary"), Value(a))
        return
    for label in periodic:
        others = tuple(lb for lb in periodic if lb != label)
        c.decide(
            [],
            _bc_path(c, label, "connection", "boundary"),
            f"Which boundary condition forms the other side of the periodic pair with {label}?",
            others,
            (
                "v2 needs each side of a periodic pair to name the other. The deck "
                f"has {len(periodic)} periodic boundary conditions"
                + (
                    ", whose transforms are not inverses."
                    if len(periodic) == 2
                    else "."
                )
            ),
        )


def _note_numbering_gap(c: Conversion, labels: list[str]) -> None:
    numbered = {}
    for label in labels:
        match = re.fullmatch(r"bc (\d+)", norm_key(label))
        if match:
            numbered[int(match.group(1))] = label
    missing = 1
    while missing in numbered:
        missing += 1
    ignored = [numbered[n] for n in sorted(numbered) if n > missing]
    if ignored:
        c.note(
            f"v1 read BC_1, BC_2, ... and stopped at the first missing number, "
            f"BC_{missing}, so it ignored {', '.join(ignored)}. v2 reads every "
            "label, so they now apply."
        )


def _sponge(c: Conversion, labels: list[str]) -> None:
    leaves = c.v1.leaves_under("sponge layer damping")
    if not leaves:
        return
    farfields = [lb for lb in labels if _type(c, lb) == "farfield"]
    if not farfields:
        _park(
            c,
            ("sponge layer damping",),
            "Which farfield boundary condition takes the sponge layer?",
            details=(
                "v2 sets the sponge layer on a farfield boundary condition, and this "
                "deck has none."
            ),
        )
        return
    for leaf in leaves:
        tail = ("sponge layer damping",) + leaf.path[1:]
        c.move(leaf, _bc_path(c, farfields[0], *tail))
        for label in farfields[1:]:
            c.also(leaf, _bc_path(c, label, *tail))
    c.note(
        "v1 sets one sponge layer for the whole deck; it is written on every "
        f"farfield boundary condition: {', '.join(farfields)}."
    )


def _target_wall(
    c: Conversion,
    keys: tuple[str, ...],
    name: str,
    walls: dict[str, list[int]],
    taken: set[str],
) -> Optional[str]:
    zone = _zone_ints(c.v1.value(*keys, "zone"))
    matches = [lb for lb, z in walls.items() if z == zone and lb not in taken]
    if len(matches) == 1:
        taken.add(matches[0])
        return matches[0]
    covering = [
        lb
        for lb, z in walls.items()
        if zone and z and set(zone) < set(z) and lb not in taken
    ]
    if not matches and len(covering) == 1:
        wall = covering[0]
        rest = sorted(set(walls[wall]) - set(zone))
        _park(
            c,
            keys,
            f"Split {wall} into two walls: the structure {name} on zones {zone}, "
            f"and a plain wall on zones {rest}. What should the new boundary "
            "condition be called?",
            details=(
                f"{name} couples to zones {zone}, which are part of {wall} "
                f"(zones {walls[wall]}). They are two regions: give zones {zone} their "
                f"own wall boundary condition with 'fsi', and keep {wall} on zones "
                f"{rest}."
            ),
        )
        return None
    _park(
        c,
        keys,
        f"Which wall boundary condition does {name} couple to?",
        tuple(matches),
        (
            f"v2 writes the structural model as 'fsi' on the wall boundary condition "
            f"whose zones equal its 'zone' {zone}; "
            + ("several match." if matches else "none does.")
        ),
    )
    return None


def _transform_type(
    c: Conversion, leaf: Optional[Leaf], path: V2Path, name: str, default: str
) -> str:
    if leaf is None:
        value = TRANSFORM_TYPES.get(default, default)
        c.set(path, Value(value))
        c.note(
            f"{name} did not set 'transform type'; the v1 default {default!r} is written."
        )
        return default
    value = leaf.value
    if value in TRANSFORM_TYPES:
        c.move(leaf, path, Value(TRANSFORM_TYPES[value]), respelled=True)
    else:
        c.move(leaf, path)
    return value


def _modal_model(
    c: Conversion, mm: str, walls: dict[str, list[int]], taken: set[str]
) -> None:
    keys = ("modal model", mm)
    target = _target_wall(c, keys, mm, walls, taken)
    if target is None:
        return
    fsi = _bc_path(c, target, "fsi")
    zone = c.v1.leaf(*keys, "zone")
    if zone is not None:
        c.drop(zone, f"{mm} is written as 'fsi' on {target}, whose zones it names")
    leaf = c.v1.leaf(*keys, "transform type")
    if leaf is not None and leaf.value == "rbf":
        c.decide(
            [leaf],
            fsi + ("transform type",),
            f"v2 has no greedy RBF: which transform does {mm} use?",
            ("rbf_multiscale",),
            (
                "'rbf_multiscale' with 'base point fraction': 1.0 is the full RBF "
                "that the greedy one approximates."
            ),
        )
        transform = "rbf"
    else:
        transform = _transform_type(
            c, leaf, fsi + ("transform type",), mm, "rbf multiscale"
        )
    mapping = dict(MODAL_KEYS)
    defaults = dict(MODAL_REQUIRED_DEFAULTS)
    if transform == "idw":
        mapping.update({k: k for k in MODAL_IDW_KEYS})
        defaults.update(MODAL_IDW_REQUIRED_DEFAULTS)
    else:
        mapping.update({k: k for k in MODAL_RBF_KEYS})
        defaults.update(MODAL_RBF_REQUIRED_DEFAULTS)
    for v1_key, v2_key in mapping.items():
        leaf = c.v1.leaf(*keys, v1_key)
        if leaf is None:
            if v1_key in defaults:
                c.set(fsi + (v2_key,), Value(defaults[v1_key]))
                c.note(
                    f"{mm} did not set '{v1_key}'; v2 requires it, so the v1 default "
                    f"{defaults[v1_key]!r} is written on {target}."
                )
            continue
        c.move(leaf, fsi + (v2_key,), respelled=v1_key != v2_key)
    if c.v1.has(*keys, "forcing"):
        state = fsi + ("harmonic initial state",)
        force = c.v1.leaf(*keys, "forcing", "force")
        if force is not None and callable(force.value):
            name = getattr(force.value, "__name__", "the function")
            c.decide(
                [force],
                state + ("force",),
                f"v2 takes a constant force vector: what is {mm}'s force in N?",
                details=(
                    f"v1 evaluated {name} once, at time 0, and read its 'force' "
                    "key. Give that value as three numbers."
                ),
            )
        c.move_tree(keys + ("forcing",), state)
        c.note(
            f"{mm} 'forcing' is 'harmonic initial state' on {target}: it sets "
            "each mode's starting displacement and velocity, and applies no "
            "force during the run.",
            keys + ("forcing",),
        )


def _generic_fsi(c: Conversion, walls: dict[str, list[int]], taken: set[str]) -> None:
    target = _target_wall(c, ("fsi",), "fsi", walls, taken)
    if target is None:
        return
    fsi = _bc_path(c, target, "fsi")
    leaf = c.v1.leaf("fsi", "type")
    if leaf is not None:
        if leaf.value == "generic_fsi":
            c.move(leaf, fsi + ("type",))
        elif isinstance(leaf.value, str) and "generic" in leaf.value:
            c.move(leaf, fsi + ("type",), Value("generic_fsi"), respelled=True)
        else:
            c.decide(
                [leaf],
                fsi + ("type",),
                f"v1 coupled only an fsi 'type' naming 'generic'; is {leaf.value!r} generic FSI?",
                ("generic_fsi",),
                f"With 'type': {leaf.value!r} the v1 solver ran with no structural coupling.",
            )
    _transform_type(
        c,
        c.v1.leaf("fsi", "transform type"),
        fsi + ("transform type",),
        "fsi",
        "rbf multiscale",
    )
    for v1_key, v2_key in FSI_KEYS.items():
        leaf = c.v1.leaf("fsi", v1_key)
        if leaf is not None:
            c.move(leaf, fsi + (v2_key,), respelled=v1_key != v2_key)
    for leaf in c.v1.leaves_under("fsi", "user variables"):
        c.move(leaf, fsi + ("user variables",) + tuple(leaf.keys[2:]))


def _overset_halos(c: Conversion, labels: list[str]) -> None:
    leaf = c.v1.leaf("create overset halos")
    overset = [lb for lb in labels if _type(c, lb) == "overset"]
    if leaf is None:
        if overset:
            c.note(
                "create overset halos is not set: v1 defaulted it to False and v2 "
                "defaults it to True, so this deck now runs with overset halos"
            )
        return
    if not overset:
        c.drop(leaf, "it acts only on an overset interface, and this deck has none")
        return
    c.move(leaf, _bc_path(c, overset[0], "create overset halos"))
    for label in overset[1:]:
        c.also(leaf, _bc_path(c, label, "create overset halos"))


def _mapping(c: Conversion, labels: list[str]) -> None:
    layers = c.v1.leaf("mapping", "number of additional overset active layers")
    if layers is not None:
        overset = [lb for lb in labels if _type(c, lb) == "overset"]
        if layers.value == 0:
            c.drop(layers, "0 is the default")
        elif not overset:
            _park(
                c,
                ("mapping", "number of additional overset active layers"),
                "Which overset interface takes the additional active layers?",
                details="v2 sets 'additional active layers' on an overset interface, and this deck has none.",
            )
        else:
            c.move(layers, _bc_path(c, overset[0], "additional active layers"))
            for label in overset[1:]:
                c.also(layers, _bc_path(c, label, "additional active layers"))
    order = c.v1.leaf("mapping", "idw from cells order")
    if order is not None and order.value == 0:
        c.drop(order, "0 is the default")
