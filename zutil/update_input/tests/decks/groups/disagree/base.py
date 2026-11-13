parameters = {
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": total_time, "time step": 0.01, "order": 2},
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 10,
        "cycles": 5,
    },
    "equations": "euler",
    "euler": {"order": "second", "inviscid flux scheme": "Roe"},
    "material": "air",
    "air": {"gamma": 1.4, "gas constant": 287.0},
    "IC_1": {
        "temperature": temperature,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "Reynolds No": 1.0e6,
    },
    "report": {"frequency": 10},
    "write output": {"format": "vtk", "volume variables": ["p"]},
}
