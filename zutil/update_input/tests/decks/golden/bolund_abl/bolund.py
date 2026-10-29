import zutil

parameters = {
    # units for dimensional quantities
    "units": "SI",
    # reference state
    "reference": "IC_1",
    # restart from previous solution
    "restart": False,
    "partitioner": "metis",
    # time marching properties
    "time marching": {
        "unsteady": {
            "total time": 1.0,
            "time step": 1.0,
            "order": "second",
            "start": 0,
        },
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl ramp factor": {"growth": 1.1, "initial": 1.0},
        "cfl": 50.0,
        "cycles": 500,
    },
    "equations": "RANS",
    "RANS": {
        "order": "second",
        "limiter": "vanalbada",
        "inviscid flux scheme": "Roe",
        "precondition": True,
        "turbulence": {"model": "sst", "betastar": 0.03},
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
        "temperature": 273.15 + 15.0,
        "pressure": 101584.0,
        "V": {"vector": zutil.vector_from_wind_dir(270.0, 12.065)},
        "viscosity": 1.73e-5,
        "turbulence intensity": 0.01,
        "eddy viscosity ratio": 0.1,
        "ambient turbulence intensity": 1.0e-20,
        "ambient eddy viscosity ratio": 1.0e-20,
        "profile": {"field": "bolund.vtp", "use wall distance": True},
    },
    "BC_1": {"zone": [34], "type": "symmetry"},
    "BC_2": {
        "zone": [0],
        "type": "wall",
        "kind": "wallfunction",
        "roughness": {"type": "length", "field": "bolund_roughness.vtp"},
    },
    "BC_3": {
        "zone": [2, 3, 4, 5],
        "type": "farfield",
        "condition": "IC_1",
        "kind": "riemann",
    },
    "write output": {
        "format": "vtk",
        "surface variables": [
            "V",
            "p",
            "T",
            "rho",
            "pressureforce",
            "frictionforce",
            "roughness",
            "yplus",
            "cp",
            "ut",
        ],
        "volume variables": [
            "V",
            "p",
            "T",
            "rho",
            "var_6",
            "ti",
            "eddy",
            "walldist",
            "cp",
        ],
    },
    "report": {"frequency": 10},
}
