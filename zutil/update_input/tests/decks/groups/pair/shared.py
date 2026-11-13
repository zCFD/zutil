parameters = {
    "units": "SI",
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 1.0, "time step": 0.01, "order": 2, "start": 0},
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 10,
        "cycles": 5,
    },
    "mapping": {"number of additional overset active layers": 2},
    "equations": "euler",
    "euler": {"order": "second", "inviscid flux scheme": "Roe"},
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": 287.0,
        "Sutherlands const": 110.4,
        "Prandtl No": 0.72,
        "Turbulent Prandtl No": 0.9,
    },
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "viscosity": 1.8e-05,
    },
    "report": {"frequency": 10},
    "write output": {
        "format": "vtk",
        "surface variables": ["p"],
        "volume variables": ["p"],
    },
}
