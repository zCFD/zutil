"""Output settings, reports and solver-wide settings (patterns O1 to O9, G1 to G9)."""

from __future__ import annotations

from typing import Optional

from zutil.update_input.model import (
    Conversion,
    Leaf,
    Source,
    V2Path,
    Value,
)
from zutil.update_input.rules._functions import port_function
from zutil.validate.schemas.callables import FORCE_TRANSFORM

OWNS = [
    "write output",
    "report",
    "units",
    "partitioner",
    "safe",
    "linear solver options",
    "mesh quality remediation angle threshold",
    "reset face area",
    "append mesh name",
    "zone id to name",
    "variable name alias",
]
ORDER = 60

SOLVER: V2Path = ("solver",)
OUTPUT: V2Path = SOLVER + ("output settings",)
SOLUTION: V2Path = OUTPUT + ("solution",)
REPORT: V2Path = OUTPUT + ("report",)
SETTINGS: V2Path = SOLVER + ("solver settings",)
LINEAR: V2Path = SETTINGS + ("linear solver options",)

FREQUENCY_STREAMS = (
    "volume data",
    "surface data",
    "volume interpolate",
    "fwh interpolate",
    "fwh wall data",
    "checkpoint",
)

WRITE_OUTPUT: dict[tuple[str, ...], V2Path] = {
    ("format",): SOLUTION + ("format",),
    ("surface variables",): SOLUTION + ("surface variables",),
    ("volume variables",): SOLUTION + ("volume variables",),
    ("scripts",): SOLUTION + ("scripts",),
    ("fwh wall data",): SOLUTION + ("fwh wall data",),
    ("fwh interpolate",): SOLUTION + ("fwh interpolate",),
    ("frequency",): SOLUTION + ("frequency",),
    ("compute average and rms",): OUTPUT + ("compute average and rms",),
    ("average start time cycle",): OUTPUT + ("average start time cycle",),
}
for _stream in FREQUENCY_STREAMS:
    for _name in (_stream, _stream + " start"):
        WRITE_OUTPUT[("frequency", _name)] = SOLUTION + ("frequency", _name)

REPORT_DIRECT: dict[tuple[str, ...], V2Path] = {
    ("frequency",): REPORT + ("frequency",),
    ("scale residuals by volume",): REPORT + ("scale residuals by volume",),
}

MONITOR_KEYS = {"name": "name", "point": "point", "variables": "variables"}

FORCE_KEYS = {
    "name": "name",
    "zone": "zones",
    "reference area": "reference area",
    "reference length": "reference length",
    "reference point": "reference point",
    "reference pressure": "reference pressure",
    "dimensional": "dimensional",
}

LINEAR_BLOCKS = ("flow", "turbulence", "rbf")

VARIABLE_LISTS = (("surface variables",), ("volume variables",))
IMMERSED_VARIABLES = (
    "immersed boundary vector",
    "immersed boundary normal",
    "immersed boundary yplus",
    "immersed boundary friction velocity",
)
IMMERSED_REASON = "zCFD no longer supports immersed boundaries"


def convert(c: Conversion) -> None:
    _units(c)
    _solver_settings(c)
    _write_output(c)
    _report(c)
    _linear_solver_options(c)


def _units(c: Conversion) -> None:
    leaf = c.v1.leaf("units")
    if leaf is not None:
        c.drop(leaf, "the solver is always SI")


def _solver_settings(c: Conversion) -> None:
    for key in ("partitioner", "safe"):
        leaf = c.v1.leaf(key)
        if leaf is None:
            continue
        if key == "partitioner" and c.v1.value(key) == "metis":
            c.drop(leaf, "metis is a development partitioner; scotch is the default")
            c.note(_METIS_TO_SCOTCH, leaf.path)
        else:
            c.move(leaf, SETTINGS + (key,))
    _remediation_threshold(c)


_METIS_TO_SCOTCH = (
    "partitioner 'metis' is a development setting, available only in research "
    "versions of zCFD: the converted deck partitions with scotch, the default."
)


def _remediation_threshold(c: Conversion) -> None:
    key = "mesh quality remediation angle threshold"
    leaf = c.v1.leaf(key)
    if leaf is None:
        return
    value = leaf.value
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if numeric and value <= 0:
        c.drop(
            leaf,
            "v1 applied mesh quality remediation only for a threshold above 0, "
            "so this deck ran without it",
        )
        return
    if c.v1.incompressible:
        c.decide(
            [leaf],
            SETTINGS + (key,),
            "v1 applied mesh quality remediation to this incompressible deck; "
            "v2 offers it only for the compressible solver. Remove the key and "
            "run without remediation, or check the mesh first?",
            details=(
                f"The v1 deck set {' > '.join(leaf.keys)} = {value!r}. Delete the "
                "key once you have accepted running without it."
            ),
        )
        return
    c.move(leaf, SETTINGS + (key,))


def _write_output(c: Conversion) -> None:
    for leaf in c.v1.leaves_under("write output"):
        rel = leaf.path[1:]
        if rel == ("immersed boundary surface",):
            c.drop(leaf, IMMERSED_REASON)
            continue
        target = WRITE_OUTPUT.get(rel)
        if target is None:
            continue
        if rel in VARIABLE_LISTS and isinstance(leaf.value, list):
            kept = [v for v in leaf.value if v not in IMMERSED_VARIABLES]
            if len(kept) < len(leaf.value):
                c.move(
                    leaf, target, Value(kept), respelled=True, reason=IMMERSED_REASON
                )
                continue
        c.move(leaf, target)


def _report(c: Conversion) -> None:
    for leaf in c.v1.leaves_under("report"):
        rel = leaf.path[1:]
        if rel in REPORT_DIRECT:
            c.move(leaf, REPORT_DIRECT[rel])
        elif len(rel) == 3 and rel[0] == "monitor" and rel[2] in MONITOR_KEYS:
            c.move(leaf, REPORT + ("monitor", leaf.keys[2], MONITOR_KEYS[rel[2]]))
        elif len(rel) == 3 and rel[0] == "forces" and rel[2] == "transform":
            target = REPORT + ("forces", leaf.keys[2], "transform")
            port_function(c, leaf, target, FORCE_TRANSFORM)
        elif len(rel) == 3 and rel[0] == "forces" and rel[2] in FORCE_KEYS:
            target = REPORT + ("forces", leaf.keys[2], FORCE_KEYS[rel[2]])
            c.move(leaf, target, respelled=rel[2] == "zone")
    _mass_flow(c)


def _folded_name(name: str) -> str:
    return name.replace(" ", "_").lower()


def _mass_flow(c: Conversion) -> None:
    labels = c.v1.children("report", "massflow")
    entries: list[tuple[str, Optional[Leaf], Leaf]] = []
    for label in labels:
        zone = c.v1.leaf("report", "massflow", label, "zone")
        if zone is not None:
            name = c.v1.leaf("report", "massflow", label, "name")
            entries.append((label, name, zone))
    usable = [
        (label, name.value)
        for label, name, _ in entries
        if name is not None and isinstance(name.value, str) and name.value.strip()
    ]
    counts: dict[str, list[str]] = {}
    for label, text in usable:
        counts.setdefault(_folded_name(text), []).append(label)
    base = REPORT + ("mass flow",)
    for label, name, zone in entries:
        shared = None
        if name is not None and isinstance(name.value, str) and name.value.strip():
            shared = counts[_folded_name(name.value)]
        if name is not None and shared is not None and len(shared) == 1:
            c.fold(
                [name, zone],
                base + (name.value, "zones"),
                Source(zone),
                respelled=True,
                reason="the report name becomes the entry's key",
            )
            continue
        c.move(zone, base + (label, "zones"), respelled=True)
        if name is None:
            continue
        if shared is not None:
            c.note(
                f"mass flow reports {', '.join(shared)} share the name "
                f"'{name.value}'; their labels are kept as keys",
                name.path,
            )
            c.drop(name, "the name repeats another report's; the label is the key")
        else:
            c.drop(name, "the name is not text; the label is the key")


def _linear_solver_options(c: Conversion) -> None:
    for leaf in c.v1.leaves_under("linear solver options"):
        rel = leaf.path[1:]
        if rel == ("double precision",):
            c.move(leaf, LINEAR + ("double precision",))
        elif len(rel) == 2 and rel[0] in LINEAR_BLOCKS:
            target = LINEAR + (rel[0], leaf.keys[2])
            if isinstance(leaf.value, str):
                c.move(leaf, target)
            else:
                c.move(leaf, target, Value(str(leaf.value)), respelled=True)
