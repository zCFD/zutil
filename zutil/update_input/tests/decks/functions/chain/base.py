def bend(x, y, z):
    return (x, y, z)


parameters = {
    "units": "SI",
    "reference": "IC_1",
    "restart": False,
    "time marching": {
        "unsteady": {"total time": 0.2, "time step": 1.0e-3},
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 10,
        "cycles": 5,
    },
    "equations": "euler",
    "euler": {"order": "first"},
    "material": "air",
    "air": {"gamma": 1.4, "gas constant": 1.0},
    "IC_1": {
        "temperature": 1.0,
        "pressure": 1.0,
        "V": {"vector": [0.75, 0.0, 0.0]},
        "Reynolds No": 1.0,
    },
    "report": {"frequency": 10},
    "BC_1": {"zone": [1], "type": "symmetry"},
    "write output": {"format": "vtk", "volume variables": ["p"]},
}
