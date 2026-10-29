Re = 400.0
v_dim = 1.0

parameters = {
    "config version": 3,
    "solver": {
        "equations": {"type": "viscous"},
        "numerical scheme": {"order": "second"},
        "fluid properties": {
            "material": "ideal gas",
            "gamma": 1.4,
            "gas constant": 287.0,
            "sutherlands const": 110.4,
            "prandtl no": 0.72,
            "turbulent prandtl no": 0.9,
        },
        "reference conditions": {
            "IC_1": {
                "temperature": 273.15,
                "pressure": 101325.0,
                "V": {"vector": [v_dim, 0.0, 0.0]},
                "Reynolds No": Re,
                "Reference Length": 1.0,
            },
            "IC_4": {
                "temperature": 273.15,
                "pressure": 101325.0,
                "V": {"vector": [0.0, 0.0, 0.0]},
                "Reynolds No": Re,
            },
        },
        "output settings": {
            "report": {"frequency": 1},
            "solution": {
                "format": "vtk",
                "surface variables": ["V", "p", "T", "rho", "cp", "mu"],
                "volume variables": ["V", "p", "T", "rho", "mu"],
            },
        },
        "solver settings": {"type": "incompressible"},
        "initialisation": {"reference": "IC_1", "initial conditions": "IC_4"},
        "time settings": {"type": "steady"},
        "convergence control": {
            "cycles": 500,
            "scheme": {"name": "coupled", "implicit relax": 0.9},
        },
    },
    "model": {
        "background": {
            "mesh": "128_wall.h5",
            "boundary conditions": {
                "BC_1": {
                    "zones": [14],
                    "type": "wall",
                    "kind": "no slip",
                    "velocity": {"type": "linear", "vector": [v_dim, 0.0, 0.0]},
                },
                "BC_2": {
                    "zones": [12],
                    "type": "wall",
                    "kind": "no slip",
                },
                "BC_3": {
                    "zones": [11, 13],
                    "type": "symmetry",
                },
            },
        }
    },
}
