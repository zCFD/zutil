dt = 1.0e-3
omega = [100.0, 120.0]

parameters = {
    "units": "SI",
    "restart": False,
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 100 * dt, "time step": dt, "order": "second"},
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 15,
        "cycles": 15,
    },
    "equations": "euler",
    "euler": {"order": "second", "limiter": "vanalbada"},
    "mapping": {"number of additional overset active layers": 1},
    "material": "air",
    "air": {"gamma": 1.4, "gas constant": 287.0},
    "IC_1": {
        "temperature": 288.0,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "viscosity": 1.8e-5,
    },
    "write output": {
        "format": "vtk",
        "surface variables": ["p"],
        "volume variables": ["V"],
    },
    "report": {"frequency": 10},
}
