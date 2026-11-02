import math


def bend(x, y, z):
    """Rotate the surface about the z axis."""
    c, s = math.cos(0.1), math.sin(0.1)
    return {"v1": c * x - s * y, "v2": s * x + c * y, "v3": z}


parameters = {
    "TR_1": {
        "type": "rbf multiscale",
        "zone": [5],
        "func": bend,
        "alpha": 2.0,
        "basis": "c2",
        "base point fraction": 0.1,
    },
}
