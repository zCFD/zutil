scale = 2.0


def turn(*, vector):
    return (vector[0] * scale, vector[1], vector[2])


parameters = {
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 1.0, "time step": 0.01, "order": 2},
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 10,
        "cycles": 5,
    },
    "equations": "euler",
    "euler": {"order": "second", "inviscid flux scheme": "Roe"},
    "material": "air",
    "air": {"gamma": 1.4, "gas constant": 287.0},
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "Reynolds No": 1.0e6,
    },
    "report": {
        "frequency": 10,
        "forces": {"FR_1": {"name": "body", "zone": [1], "transform": turn}},
    },
    "write output": {"format": "vtk", "volume variables": ["p"]},
}
