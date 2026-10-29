parameters = {
    "units": "SI",
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 3.6, "time step": 0.001, "order": 1, "start": 0},
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 30,
        "cycles": 5,
    },
    "Sponge Layer Damping": {
        "Distance": 20.0,
        "Damping Factor": 0.001 * 1.0 * (1.0 + 0.23) * 331.3 / 0.2,
        "Condition": "IC_1",
    },
    "mapping": {"idw from cells order": 0},
    "equations": "euler",
    "euler": {
        "order": "second",
        "precondition": False,
        "inviscid flux scheme": "Roe",
    },
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": 287.0,
        "Sutherlands const": 110.4,
        "Prandtl No": 0.72,
        "Turbulent Prandtl No": 0.9,
    },
    "IC_1": {
        "temperature": 277.7777 + 15,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 1.0e-10},
        "Reynolds No": 150,
        "Reference Length": 1.0,
        "turbulence intensity": 1.0e-4,
        "eddy viscosity ratio": 0.1,
    },
    "BC_1": {"zone": [11, 13], "type": "symmetry"},
    "BC_2": {"zone": [12], "type": "farfield", "condition": "IC_1", "kind": "riemann"},
    "report": {
        "frequency": 10,
        "monitor": {
            "MR_1": {"name": "probe", "point": [-140, 1.5, 0], "variables": ["p"]}
        },
    },
    "write output": {
        "format": "vtk",
        "surface variables": ["overset", "V", "p", "T", "rho"],
        "volume variables": ["overset", "V", "p", "T", "rho", "m", "sponge"],
    },
}
