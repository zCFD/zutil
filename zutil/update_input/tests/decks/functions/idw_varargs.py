def nudge(*args, **kwargs):
    x, y, z = args
    return (x, y, z + 0.01)


parameters = {
    "TR_1": {"type": "idw", "zone": [5], "func": nudge, "alpha": 2.0},
}
