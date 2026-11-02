"""Equations, numerical scheme and fluid properties (patterns E1 to E12)."""

from __future__ import annotations

from ..model import INCOMPRESSIBLE_EQUATIONS, Conversion, Leaf, Value, norm_key

OWNS = [
    "equations",
    "euler",
    "viscous",
    "rans",
    "les",
    "incomp viscous",
    "incomp rans",
    "incomp les",
    "material",
    "air",
]
ORDER = 20

SOLVER = ("solver",)
EQ = SOLVER + ("equations",)
NS = SOLVER + ("numerical scheme",)
FP = SOLVER + ("fluid properties",)
SCHEME = SOLVER + ("convergence control", "scheme")

COMPRESSIBLE = ("euler", "viscous", "rans", "les")

SCHEME_KEYS = (
    "order",
    "limiter",
    "reconstruction",
    "first order cycles",
    "freeze limiter cycle",
    "permit backward difference during reconstruction",
)
COMPRESSIBLE_SCHEME_KEYS = (
    "entropy fix",
    "linear gradients",
    "roe low dissipation sensor",
    "roe dissipation sensor minimum",
    "roe entropy fix coefficient",
)
V1_DEFAULTS_DROPPED_ON_INCOMPRESSIBLE = {
    "entropy fix": 0.0,
    "roe entropy fix coefficient": 0.001,
}
INCOMPRESSIBLE_EQUATION_KEYS = (
    "correction limiter",
    "gradient limiter type",
    "gradient limiter",
    "max second order non orthogonal angle",
    "central blend",
)
TURBULENCE_KEYS = (
    "model",
    "betastar",
    "limit mut",
    "limit gradient k",
    "qcr",
    "rotation correction",
    "production",
    "separation correction",
    "cdes kw",
    "cdes keps",
    "cdes",
    "cd1",
    "cd2",
    "cw",
    "a1",
    "ca1",
    "ca2",
    "ce1",
    "ce2",
    "cthetat",
    "sigmagamma",
    "sigmathetat",
)
LIMITER_RESPELLINGS = {
    "barth jespersen": (
        "symmetric ratio",
        "v1 'barth jespersen' is the one-dimensional limiter v2 calls 'symmetric "
        "ratio'; v2 'barth jespersen' is the cell-based limiter",
    ),
    "vanalbada": ("van albada", "v2 spells the limiter 'van albada'"),
}
FLUID_KEYS = ("gamma", "gas constant", "sutherlands const", "prandtl no")


def convert(c: Conversion) -> None:
    equation = c.v1.leaf("equations")
    name = norm_key(equation.value) if equation is not None else None
    if name in COMPRESSIBLE + INCOMPRESSIBLE_EQUATIONS:
        _equation_set(c, equation, name)
        _named_block(c, name)
    _unselected_blocks(c, name)
    _material(c, incompressible=c.v1.incompressible)


def _equation_set(c: Conversion, leaf: Leaf, name: str) -> None:
    if c.v1.incompressible:
        c.move(
            leaf,
            EQ + ("type",),
            Value(name.removeprefix("incomp ")),
            respelled=True,
            reason="the incompressible solver is a solver setting in config version 2",
        )
        c.set(SOLVER + ("solver settings", "type"), Value("incompressible"))
    else:
        c.move(leaf, EQ + ("type",))
        c.set(SOLVER + ("solver settings", "type"), Value("compressible"))


def _unselected_blocks(c: Conversion, selected: str | None) -> None:
    for block in COMPRESSIBLE + INCOMPRESSIBLE_EQUATIONS:
        if block == selected:
            continue
        for leaf in c.v1.leaves_under(block):
            if not c.claimed(leaf):
                c.drop(
                    leaf,
                    f"block of the '{block}' equation set, which this deck does not select",
                )


def _named_block(c: Conversion, name: str) -> None:
    incompressible = c.v1.incompressible
    kind = c.v1.equation_set
    precondition_on = bool(c.v1.value(name, "precondition", default=False))
    for leaf in c.v1.leaves_under(name):
        rel = leaf.path[1:]
        if rel == ("turbulence",) or rel[:1] == ("turbulence",):
            _turbulence(c, leaf, rel[1:], kind, incompressible)
        else:
            _block_key(c, leaf, rel, incompressible, precondition_on)


def _block_key(
    c: Conversion,
    leaf: Leaf,
    rel: tuple[str, ...],
    incompressible: bool,
    precondition_on: bool,
) -> None:
    key = rel[0] if len(rel) == 1 else None
    if rel == ("precondition",):
        c.move(leaf, EQ + ("precondition",))
    elif rel == ("preconditioner", "minimum mach number"):
        _preconditioner(c, leaf, incompressible, precondition_on)
    elif key == "limiter":
        _limiter(c, leaf)
    elif key in SCHEME_KEYS:
        c.move(leaf, NS + (key,))
    elif key == "inviscid flux scheme":
        _flux_scheme(c, leaf)
    elif key == "leastsq gradients":
        _least_squares(c, leaf)
    elif key in COMPRESSIBLE_SCHEME_KEYS:
        _compressible_scheme_key(c, leaf, key, incompressible)
    elif incompressible and key == "relax":
        c.move(leaf, SCHEME + ("implicit relax",))
    elif incompressible and key == "correction":
        _correction(c, leaf)
    elif incompressible and key in INCOMPRESSIBLE_EQUATION_KEYS:
        c.move(leaf, EQ + (key,))


def _preconditioner(
    c: Conversion, leaf: Leaf, incompressible: bool, precondition_on: bool
) -> None:
    if not precondition_on:
        c.drop(leaf, "read only when precondition is on")
    elif not incompressible:
        c.move(leaf, NS + ("preconditioner", "minimum mach number"))


def _limiter(c: Conversion, leaf: Leaf) -> None:
    respelling = LIMITER_RESPELLINGS.get(str(leaf.value).lower())
    if respelling is None:
        c.move(leaf, NS + ("limiter",))
        return
    name, reason = respelling
    c.move(leaf, NS + ("limiter",), Value(name), respelled=True, reason=reason)


def _flux_scheme(c: Conversion, leaf: Leaf) -> None:
    path = NS + ("inviscid flux scheme",)
    if str(leaf.value).lower() == "roe low diss":
        c.move(
            leaf,
            path,
            Value("roe low-diss"),
            respelled=True,
            reason="v2 spells the low-dissipation Roe scheme 'roe low-diss'",
        )
    else:
        c.move(leaf, path)


def _least_squares(c: Conversion, leaf: Leaf) -> None:
    if leaf.value:
        c.move(
            leaf,
            NS + ("gradient",),
            Value("least-squares"),
            respelled=True,
            reason="v2 names the gradient method",
        )
    else:
        c.drop(leaf, "green-gauss is the default gradient")


def _compressible_scheme_key(
    c: Conversion, leaf: Leaf, key: str, incompressible: bool
) -> None:
    if not incompressible:
        c.move(leaf, NS + (key,))
        return
    default = V1_DEFAULTS_DROPPED_ON_INCOMPRESSIBLE.get(key)
    if default is not None and leaf.value == default:
        c.drop(
            leaf,
            f"the incompressible solver has no {key}, and the v1 value is the v1 default",
        )


def _correction(c: Conversion, leaf: Leaf) -> None:
    path = EQ + ("correction",)
    if leaf.value == "over relaxed":
        c.move(
            leaf,
            path,
            Value("over_relaxed"),
            respelled=True,
            reason="v2 spells 'over relaxed' with an underscore",
        )
    else:
        c.move(leaf, path)


def _turbulence(
    c: Conversion,
    leaf: Leaf,
    rel: tuple[str, ...],
    kind: str,
    incompressible: bool,
) -> None:
    key = rel[0] if len(rel) == 1 else None
    path = EQ + ("turbulence",)
    if kind == "rans":
        if key in TURBULENCE_KEYS:
            c.move(leaf, path + (key,))
        elif key == "les":
            _hybrid(c, leaf)
        elif incompressible and key == "relax":
            c.move(leaf, SCHEME + ("turbulence relax",))
    elif kind == "les":
        if key == "les":
            _les_model(c, leaf)
        elif incompressible and key == "relax":
            c.move(leaf, path + ("relax",))


def _hybrid(c: Conversion, leaf: Leaf) -> None:
    if str(leaf.value).lower() == "none":
        c.drop(leaf, "no hybrid RANS-LES is the default")
    else:
        c.move(leaf, EQ + ("turbulence", "hybrid"))


def _les_model(c: Conversion, leaf: Leaf) -> None:
    path = EQ + ("turbulence", "les")
    text = str(leaf.value)
    if text == text.lower():
        c.move(leaf, path)
    else:
        c.move(
            leaf,
            path,
            Value(text.lower()),
            respelled=True,
            reason="v2 spells LES model names in lower case",
        )


def _material(c: Conversion, incompressible: bool) -> None:
    material = c.v1.leaf("material")
    if material is not None:
        name = norm_key(material.value)
        if name == "air":
            c.move(
                material,
                FP + ("material",),
                Value("ideal gas"),
                respelled=True,
                reason="v1 air is the ideal-gas model, which v2 names ideal gas",
            )
        else:
            c.move(material, FP + ("material",))
    else:
        name = "air"
    euler = c.v1.equation_set == "euler"
    for leaf in c.v1.leaves_under(name):
        if c.claimed(leaf) or len(leaf.path) != 2:
            continue
        key = leaf.path[1]
        if key in FLUID_KEYS:
            c.move(leaf, FP + (key,))
        elif key == "turbulent prandtl no":
            if euler:
                c.drop(leaf, "inviscid: no turbulent Prandtl number")
            else:
                c.move(leaf, FP + (key,))
        elif key == "gravity":
            _gravity(c, leaf, incompressible)


def _gravity(c: Conversion, leaf: Leaf, incompressible: bool) -> None:
    if incompressible:
        c.move(leaf, FP + ("gravity",))
        return
    c.decide(
        [leaf],
        FP + ("gravity",),
        "v2 applies gravity to the incompressible solver only",
        details=(
            "This deck is compressible, so the v1 gravity vector has no effect "
            "in v2. Delete it, or move the deck to the incompressible solver."
        ),
    )
