R = 1.0


def bend(x, y, z):
    return {"v1": x, "v2": y, "v3": z}


def drive(cell_centres):
    return [(1.0, 0.0, 0.0) for _ in cell_centres]


def shock(**kwargs):
    location = kwargs["location"]

    pressure = 1.0
    if location[0] > 0.3:
        pressure = 0.1
    return {"pressure": pressure, "temperature": pressure / R}


def wind_axes(x, y, z):
    return (x, y, z)


parameters = {
    "units": "SI",
    "reference": "IC_1",
    "initial": {"name": "IC_1", "func": shock},
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
    "air": {"gamma": 1.4, "gas constant": R},
    "IC_1": {
        "temperature": 1.0,
        "pressure": 1.0,
        "V": {"vector": [0.75, 0.0, 0.0]},
        "Reynolds No": 1.0,
    },
    "FZ_2": {"type": "momentum", "def": "source.vtp", "func": drive},
    "TR_1": {"type": "idw", "zone": [5], "func": bend, "alpha": 2.0},
    "report": {
        "frequency": 10,
        "forces": {"FR_1": {"name": "body", "zone": [1], "transform": wind_axes}},
    },
    "BC_1": {"zone": [1], "type": "symmetry"},
    "write output": {"format": "vtk", "volume variables": ["p"]},
}
