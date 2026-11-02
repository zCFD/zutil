def deform(x, y, z):
    return [0.0, 0.0, 0.0]


parameters = {
    "TR_1": {
        "type": "rbf",
        "zone": [5],
        "func": deform,
        "alpha": 2.0,
        "basis": "c2",
        "tol": 1.0e-4,
    },
}
