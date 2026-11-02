parameters = {
    "units": "SI",
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 1.0, "time step": 0.01, "order": 2, "start": 0},
        "cycles": 5,
    },
    "equations": "incomp viscous",
    "incomp viscous": {"order": "second", "precondition": True, "relax": 0.5},
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
        "Reynolds No": 1.0e6,
        "Reference Length": 1.0,
    },
    "report": {"frequency": 10},
    "write output": {"format": "vtk", "volume variables": ["p"]},
}
