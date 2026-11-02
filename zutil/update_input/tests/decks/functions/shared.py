def bend(x, y, z):
    return (x, y, z)


parameters = {
    "TR_1": {"type": "idw", "zone": [5], "func": bend, "alpha": 2.0},
    "TR_2": {"type": "idw", "zone": [6], "func": bend, "alpha": 2.0},
}
