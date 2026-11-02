def lift(px, py, pz):
    return (px, py, pz + 0.01)


parameters = {
    "TR_1": {
        "type": "idw",
        "zone": [5],
        "func": lift,
        "alpha": 2.0,
        "power": [3.0, 5.0, 10.0],
    },
}
