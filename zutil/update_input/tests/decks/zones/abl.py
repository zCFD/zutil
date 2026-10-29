def drive(x, y, z):
    return [1.0, 0.0, 0.0]


parameters = {
    "FZ_1": {"type": "abl", "zone": [2], "func": drive, "kind": "func"},
}
