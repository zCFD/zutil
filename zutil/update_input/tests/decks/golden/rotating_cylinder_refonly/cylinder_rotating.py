import numpy as np
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytrans

dt = 0.001
alpha = 3.0
Re = 200.0
cyl_r = 0.5
# 1 deg per dt
omega = 2.0 * np.pi / (360.0 * dt)
u_inf = (omega * cyl_r) / alpha


def transform():
    rot_matrix = pyrot.matrix_from_axis_angle([0, 1, 0, -90 * np.pi / 180])
    A2B = np.identity(4)
    A2B = pytrans.rotate_transform(A2B, rot_matrix)
    return A2B


parameters = {
    # units for dimensional quantities
    "units": "SI",
    # reference state
    "reference": "IC_1",
    #'restart' : True,
    "time marching": {
        # "unsteady": {"total time": 1.2, "time step": 0.002, "order": 2, "start": 0},
        "unsteady": {"total time": 4.8, "time step": dt, "order": 2, "start": 0},
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 100,
        "cfl ramp factor": {"initial": 0.1, "growth": 1.05},
        "cycles": 5,
    },
    "mesh transform matrix": transform(),
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
    "FZ_1": {
        "type": "rotating",
        "zone": [3],
        "axis": [0, 0, 1],
        "origin": [0, 0, 0],
        "omega": omega,
    },
    "BC_1": {"ref": 7, "type": "symmetry"},
    "BC_2": {"ref": 3, "type": "wall", "kind": "slip"},
    "BC_3": {"ref": 9, "type": "farfield", "condition": "IC_1", "kind": "riemann"},
    "report": {
        "frequency": 5,
        "monitor": {
            "MR_1": {"name": "probe", "point": [0.25, 1.07, 0.313], "variables": ["cp"]}
        },
    },
    "write output": {
        "format": "vtk",
        "surface variables": ["V", "p", "T", "rho", "cp"],
        "volume variables": ["V", "p", "T", "rho", "m", "cp"],
        "frequency": {
            "volume data": 100,
            "surface data": 100,
        },
    },
}
