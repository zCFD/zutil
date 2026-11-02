import numpy as np

dt = 0.001
alpha = 3.0
Re = 200.0
cyl_r = 0.5
omega = 2.0 * np.pi / (360.0 * dt)
u_inf = (omega * cyl_r) / alpha

parameters = {
    "units": "SI",
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 2.4, "time step": dt, "order": 2, "start": 0},
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 40,
        "cfl ramp factor": {"initial": 0.1, "growth": 1.05},
        "cycles": 10,
    },
    "equations": "euler",
    "euler": {"order": "second", "precondition": True, "inviscid flux scheme": "Roe"},
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": 287.0,
        "Sutherlands const": 110.4,
        "Prandtl No": 0.72,
        "Turbulent Prandtl No": 0.9,
    },
    "IC_1": {
        "temperature": 277.7777,
        "pressure": 101325.0,
        "V": {"vector": [u_inf, 0.0, 0.0]},
        "Reynolds No": Re,
        "Reference Length": 1.0,
        "turbulence intensity": 1.0e-4,
        "eddy viscosity ratio": 0.1,
    },
    "BC_1": {"zone": [5, 6], "type": "symmetry"},
    "BC_2": {
        "zone": [1, 2, 3, 4],
        "type": "farfield",
        "condition": "IC_1",
        "kind": "riemann",
    },
    "report": {
        "frequency": 1,
        "monitor": {
            "MR_1": {"name": "probe", "point": [1.07, 0.25, 0.313], "variables": ["cp"]}
        },
    },
    "write output": {
        "format": "vtk",
        "surface variables": ["V", "p", "T", "rho", "cp"],
        "volume variables": [
            "V",
            "p",
            "T",
            "rho",
            "m",
            "cp",
            "overset",
            "cell_velocity",
        ],
        "frequency": 5,
    },
}
