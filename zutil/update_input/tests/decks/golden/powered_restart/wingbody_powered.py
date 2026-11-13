import zutil
import math

inflow = [1.0, 0.0, 0.0]
inflow = zutil.rotate_vector(inflow, -1.0 * 2.0, 0.0)
sound = math.sqrt(1.4 * 287.0 * 220.0)


def body_axes(x, y, z):
    v = [x, y, z]
    v = zutil.rotate_vector(v, 2.0, 0.0)
    return {"v1": v[0], "v2": v[1], "v3": v[2]}


parameters = {
    "scale": [0.0254, 0.0254, 0.0254],
    "units": "SI",
    "reference": "IC_1",
    "initial": {"name": "IC_1"},
    "restart": True,
    "restart casename": "wingbody_unpowered",
    "restart ignore history": True,
    "safe": False,
    "partitioner": "metis",
    "time marching": {
        "unsteady": {
            "total time": 1.0,
            "time step": 1.0,
            "order": "second",
            "start": 2000,
        },
        "scheme": {"name": "implicit euler", "stage": 1},
        "cfl": 20.0,
        "cfl transport": 10.0,
        "cfl ramp factor": {"initial": 0.5, "growth": 1.004},
        "cycles": 100,
    },
    "linear solver options": {"double precision": True},
    "equations": "rans",
    "rans": {
        "order": "second",
        "limiter": "vanalbada",
        "precondition": False,
        "inviscid flux scheme": "Roe",
        "first order cycles": 10,
        "turbulence": {"model": "sst", "qcr": 2000},
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
        "temperature": 220.0,
        "pressure": 20000.0,
        "V": {"Mach": 0.8, "vector": inflow},
        "viscosity": 1.4e-05,
        "turbulence intensity": 0.01,
        "eddy viscosity ratio": 0.1,
        "ambient turbulence intensity": 0.01,
        "ambient eddy viscosity ratio": 0.1,
    },
    "IC_2": {"reference": "IC_1", "mass flow rate": 0.5},
    "IC_3": {
        "reference": "IC_1",
        "mass flow rate": 0.5,
        "total temperature ratio": 380.0 / 220.0,
    },
    "IC_4": {"reference": "IC_1", "mass flow rate": 20.0 + 120.0},
    "IC_5": {
        "reference": "IC_1",
        "total pressure ratio": 65000.0 / 24000.0,
        "total temperature ratio": 280.0 / 220.0,
    },
    "IC_6": {
        "reference": "IC_1",
        "total pressure ratio": 43000.0 / 24000.0,
        "total temperature ratio": 690.0 / 220.0,
    },
    "BC_1": {"zone": [0, 9], "type": "symmetry"},
    "BC_2": {
        "zone": list([5, 6, 7]) + list([8, 12]),
        "type": "wall",
        "kind": "noslip",
    },
    "BC_3": {
        "zone": [1, 2, 3, 4],
        "type": "farfield",
        "condition": "IC_1",
        "kind": "riemann",
    },
    "BC_4": {"zone": [], "type": "outflow", "kind": "massflow", "condition": "IC_2"},
    "BC_5": {"zone": [], "type": "inflow", "kind": "massflow", "condition": "IC_3"},
    "BC_6": {"zone": [10], "type": "outflow", "kind": "massflow", "condition": "IC_4"},
    "BC_7": {"zone": [11], "type": "inflow", "kind": "default", "condition": "IC_5"},
    "BC_8": {"zone": [13], "type": "inflow", "kind": "default", "condition": "IC_6"},
    "write output": {
        "format": "vtk",
        "surface variables": ["p", "T", "rho", "cp", "cf", "yplus", "mach", "V"],
        "volume variables": ["V", "p", "T", "rho", "eddy", "mach"],
        "frequency": 100,
    },
    "report": {
        "frequency": 5,
        "forces": {
            "FR_1": {
                "name": "all_walls",
                "zone": list([5, 6, 7]) + list([8, 12]),
                "transform": body_axes,
                "reference area": 60.0,
                "reference point": [15.0, 0.0, 3.0],
                "reference length": 4.0,
                "reference pressure": 24000.0,
            },
            "FR_2": {
                "name": "airframe",
                "zone": list([5, 6, 7]) + list([]),
                "transform": body_axes,
                "reference area": 60.0,
                "reference point": [15.0, 0.0, 3.0],
                "reference length": 4.0,
                "reference pressure": 24000.0,
            },
            "FR_3": {
                "name": "nacelle",
                "zone": list([8, 12]) + list([10]) + list([11]) + list([13]),
                "transform": body_axes,
                "reference area": 60.0,
                "reference point": [15.0, 0.0, 3.0],
                "reference length": 4.0,
                "reference pressure": 24000.0,
            },
            "FR_4": {
                "name": "spare",
                "zone": list([]),
                "transform": body_axes,
                "reference area": 1.0,
                "reference point": [0.0, 0.0, 0.0],
                "reference length": 1.0,
                "reference pressure": 24000.0,
            },
        },
        "massflow": {
            "MF_1": {"name": "intake_duct", "zone": []},
            "MF_2": {"name": "exhaust_duct", "zone": []},
            "MF_3": {"name": "fan_face", "zone": [10]},
            "MF_4": {"name": "cold_nozzle", "zone": [11]},
            "MF_5": {"name": "hot_nozzle", "zone": [13]},
        },
    },
}
