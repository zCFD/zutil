def drive(*, locations):
    return [(1.0, 0.0, 0.0) for _ in locations]


parameters = {
    "FZ_1": {"type": "momentum", "zone": [0], "func": drive},
}
