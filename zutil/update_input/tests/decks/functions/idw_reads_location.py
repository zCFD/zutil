location = (0.0, 0.0, 1.0)


def shift(x, y, z):
    return (x + location[0], y, z)


parameters = {
    "TR_1": {"type": "idw", "zone": [5], "func": shift, "alpha": 2.0},
}
