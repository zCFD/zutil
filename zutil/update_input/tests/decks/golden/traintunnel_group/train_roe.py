import math
import zutil
import zutil.transform


def linear_accel(**kwargs):
    t = kwargs["time"]
    dt = kwargs["time step"]
    u = [-69.4, 0, 0]
    if t < 1.5:
        u[0] = -(t / 1.5) * 69.4
    return {"velocity": tuple(u)}


parameters = {
    "units": "SI",
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 3.6, "time step": 0.001, "order": 1, "start": 0},
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 30,
        "cycles": 5,
        "moving mesh": True,
    },
    "mapping": {"idw from cells order": 0},
    "mesh transform matrix": zutil.transform.translate(
        [0.0, 0.0, 0.0], [110.0, 0.0, 0.0]
    ),
    "equations": "euler",
    "euler": {"order": "second", "precondition": False, "inviscid flux scheme": "Roe"},
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": 287.0,
        "Sutherlands const": 110.4,
        "Prandtl No": 0.72,
        "Turbulent Prandtl No": 0.9,
    },
    "FZ_1": {
        "type": "translating",
        "zone": [6],
        "vector": [-1, 0, 0],
        "mach": 69.4 / math.sqrt(1.4 * 287 * (277.7777 + 15)),
        "translation function": linear_accel,
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
    "BC_1": {"zone": [10], "type": "symmetry"},
    "BC_2": {"zone": [9], "type": "overset"},
    "BC_3": {"zone": [8], "type": "wall", "kind": "slip"},
    "report": {
        "frequency": 10,
    },
    "write output": {
        "format": "vtk",
        "surface variables": ["V", "p", "T", "rho"],
        "volume variables": ["V", "p", "T", "rho", "m"],
    },
}
