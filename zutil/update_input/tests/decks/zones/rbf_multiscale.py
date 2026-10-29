def deform(x, y, z):
    return [0.0, 0.0, 0.0]


parameters = {
    "TR_1": {
        "type": "rbf multiscale",
        "zone": [5],
        "func": deform,
        "alpha": 2.0,
        "basis": "c2",
        "base point fraction": 0.1,
    },
}
