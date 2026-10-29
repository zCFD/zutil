import numpy as np
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr


dt = 0.001
alpha = 3.0
Re = 200.0
cyl_r = 0.5
omega = 2.0 * np.pi / (360.0 * dt)
u_inf = (omega * cyl_r) / alpha

print("alpha {} omega {} U {}".format(alpha, omega, u_inf))


def my_mesh_transform(**kwargs):
    x = list(kwargs["coord"])
    x[0] += 0

    return {"coord": tuple(x)}


def matrix_func(**kwargs):
    time = kwargs["real time"]
    if dt == 0:
        step = 0
    else:
        step = time / dt

    step -= 1
    x_centre = step * 0.25 - 0.0

    B2C = np.identity(4)
    b = np.array([0.0, 0.0, 1.0, 0.5 * np.pi / 180.0])
    rot_mtx_b = pyrot.matrix_from_axis_angle(b)
    B2C = pytr.rotate_transform(B2C, rot_mtx_b)

    return B2C


parameters = {
    "units": "SI",
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 2 * dt, "time step": dt, "order": 2, "start": 0},
        "cfl ramp factor": {"initial": 0.1, "growth": 1.05},
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 40,
        "cycles": 10,
        "moving mesh": True,
    },
    "scale": [1, 1, 0.2],
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
    "FZ_1": {
        "type": "rotating",
        "zone": [9],
        "axis": [0, 0, 1],
        "origin": [0, 0, 0],
        "omega": omega,
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
    "BC_1": {"zone": [11, 12], "type": "symmetry"},
    "BC_2": {"zone": [13], "type": "overset"},
    "BC_3": {"zone": [14], "type": "wall", "kind": "slip"},
    "report": {
        "frequency": 1,
        "monitor": {
            "MR_1": {"name": "probe", "point": [1.07, 0.25, 0.313], "variables": ["cp"]}
        },
        "forces": {"FR_1": {"name": "cyl", "zone": [10]}},
    },
    "write output": {
        "fwh wall data": True,
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
            "cellzone",
        ],
        "frequency": {
            "volume data": 5,
            "surface data": 5,
            "checkpoint": 1000,
            "fwh wall data": 1,
            "fwh wall data start": 100,
        },
    },
}
