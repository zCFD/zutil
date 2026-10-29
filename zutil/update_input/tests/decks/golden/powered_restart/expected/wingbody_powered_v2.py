import zutil
import math

inflow = [1.0, 0.0, 0.0]
inflow = zutil.rotate_vector(inflow, -1.0 * 2.0, 0.0)
sound = math.sqrt(1.4 * 287.0 * 220.0)


def body_axes(*, vector):
    x, y, z = vector
    v = [x, y, z]
    v = zutil.rotate_vector(v, 2.0, 0.0)
    return {"v1": v[0], "v2": v[1], "v3": v[2]}


parameters = {
    "config version": 2,
    "solver": {
        "reference conditions": {
            "IC_1": {
                "temperature": 220.0,
                "pressure": 20000.0,
                "v": {"mach": 0.8, "vector": inflow},
                "viscosity": 1.4e-05,
                "turbulence intensity": 0.01,
                "eddy viscosity ratio": 0.1,
                "ambient turbulence intensity": 0.01,
                "ambient eddy viscosity ratio": 0.1,
            }
        },
        "initialisation": {
            "reference": "IC_1",
            "initial conditions": "IC_1",
            "restart": True,
            "restart casename": "wingbody_unpowered",
            "restart ignore history": True,
        },
        "equations": {
            "type": "rans",
            "precondition": False,
            "turbulence": {"model": "sst", "qcr": 2000},
        },
        "solver settings": {
            "type": "compressible",
            "safe": False,
            "linear solver options": {"double precision": True},
        },
        "numerical scheme": {
            "order": "second",
            "limiter": "van albada",
            "inviscid flux scheme": "Roe",
            "first order cycles": 10,
        },
        "fluid properties": {
            "material": "ideal gas",
            "gamma": 1.4,
            "gas constant": 287.0,
            "sutherlands const": 110.4,
            "prandtl no": 0.72,
            "turbulent prandtl no": 0.9,
        },
        "time settings": {"type": "steady"},
        "convergence control": {
            "cycles": 100,
            "scheme": {"name": "implicit euler", "stage": 1},
            "cfl": {
                "cfl": 20.0,
                "cfl turbulence": 10.0,
                "cfl ramp": [{"type": "exponential", "factor": 1.004, "initial": 0.5}],
            },
        },
        "output settings": {
            "solution": {
                "format": "vtk",
                "surface variables": [
                    "p",
                    "T",
                    "rho",
                    "cp",
                    "cf",
                    "yplus",
                    "mach",
                    "V",
                ],
                "volume variables": ["V", "p", "T", "rho", "eddy", "mach"],
                "frequency": 100,
            },
            "report": {
                "frequency": 5,
                "forces": {
                    "FR_1": {
                        "name": "all_walls",
                        "zones": list([5, 6, 7]) + list([8, 12]),
                        "transform": body_axes,
                        "reference area": 60.0,
                        "reference point": [15.0, 0.0, 3.0],
                        "reference length": 4.0,
                        "reference pressure": 24000.0,
                    },
                    "FR_2": {
                        "name": "airframe",
                        "zones": list([5, 6, 7]) + list([]),
                        "transform": body_axes,
                        "reference area": 60.0,
                        "reference point": [15.0, 0.0, 3.0],
                        "reference length": 4.0,
                        "reference pressure": 24000.0,
                    },
                    "FR_3": {
                        "name": "nacelle",
                        "zones": list([8, 12]) + list([10]) + list([11]) + list([13]),
                        "transform": body_axes,
                        "reference area": 60.0,
                        "reference point": [15.0, 0.0, 3.0],
                        "reference length": 4.0,
                        "reference pressure": 24000.0,
                    },
                    "FR_4": {
                        "name": "spare",
                        "zones": list([]),
                        "transform": body_axes,
                        "reference area": 1.0,
                        "reference point": [0.0, 0.0, 0.0],
                        "reference length": 1.0,
                        "reference pressure": 24000.0,
                    },
                },
                "mass flow": {
                    "intake_duct": {"zones": []},
                    "exhaust_duct": {"zones": []},
                    "fan_face": {"zones": [10]},
                    "cold_nozzle": {"zones": [11]},
                    "hot_nozzle": {"zones": [13]},
                },
            },
        },
    },
    "model": {
        "wingbody": {
            "mesh": "wingbody.h5",
            "boundary conditions": {
                "BC_4": {
                    "mass flow rate": 0.5,
                    "zones": [],
                    "type": "outflow",
                    "kind": "massflow",
                    "reference": "IC_1",
                },
                "BC_5": {
                    "mass flow rate": 0.5,
                    "total temperature ratio": 380.0 / 220.0,
                    "zones": [],
                    "type": "inflow",
                    "kind": "massflow",
                    "reference": "IC_1",
                },
                "BC_6": {
                    "mass flow rate": 20.0 + 120.0,
                    "zones": [10],
                    "type": "outflow",
                    "kind": "massflow",
                    "reference": "IC_1",
                },
                "BC_7": {
                    "total pressure ratio": 65000.0 / 24000.0,
                    "total temperature ratio": 280.0 / 220.0,
                    "zones": [11],
                    "type": "inflow",
                    "kind": "pressure",
                    "reference": "IC_1",
                },
                "BC_8": {
                    "total pressure ratio": 43000.0 / 24000.0,
                    "total temperature ratio": 690.0 / 220.0,
                    "zones": [13],
                    "type": "inflow",
                    "kind": "pressure",
                    "reference": "IC_1",
                },
                "BC_1": {"zones": [0, 9], "type": "symmetry"},
                "BC_2": {
                    "zones": list([5, 6, 7]) + list([8, 12]),
                    "type": "wall",
                    "kind": "no slip",
                },
                "BC_3": {
                    "zones": [1, 2, 3, 4],
                    "type": "farfield",
                    "kind": "riemann",
                    "condition": "IC_1",
                },
            },
            "transforms": {"scale": [0.0254, 0.0254, 0.0254]},
        }
    },
}
