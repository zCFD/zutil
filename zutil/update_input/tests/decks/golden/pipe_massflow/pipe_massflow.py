import zutil
import math

inlet = [3]
outlet = [2]
wall = [0, 1]

massflow = 10.0
inlet_area = 0.78539

temp = 293.15
press = 101325.0
visc = 1.825e-5
sound = math.sqrt(1.4 * 287.0 * temp)
dens = press / (temp * 287.0)
speed = massflow / (dens * inlet_area)
mach = speed / sound
inlet_mass_flow_ratio = massflow / (dens * sound * mach)
inflow = [0.0, 0.0, 1.0]


def my_transform(x, y, z):
    v = [x, y, z]
    v = zutil.rotate_vector(v, 0.0, 0.0)
    return {"v1": v[0], "v2": v[1], "v3": v[2]}


parameters = {
    "units": "SI",
    "reference": "IC_1",
    "initial": {"name": "IC_1"},
    "restart": False,
    "safe": False,
    "partitioner": "metis",
    "time marching": {
        "unsteady": {
            "total time": 1.0,
            "time step": 1.0,
            "order": "second",
            "start": 2000,
        },
        "scheme": {"name": "runge kutta", "stage": 5, "kind": "local timestepping"},
        "multigrid cycles": 20000,
        "multigrid": 3,
        "cfl": 2.0,
        "cfl transport": 1.0,
        "cfl coarse": 1.0,
        "cfl ramp factor": {"initial": 0.001, "growth": 1.05},
        "cycles": 20000,
    },
    "equations": "rans",
    "rans": {
        "order": "second",
        "limiter": "vanalbada",
        "precondition": True,
        "turbulence": {"model": "sst"},
    },
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": 287.0,
        "sutherlands const": 110.4,
        "prandtl no": 0.72,
        "turbulent prandtl no": 0.9,
    },
    "ic_1": {
        "temperature": temp,
        "pressure": press,
        "v": {"mach": mach, "vector": inflow},
        "viscosity": visc,
        "reference length": 1.0,
        "turbulence intensity": 0.04,
        "eddy viscosity ratio": 50.0,
        "ambient turbulence intensity": 0.04,
        "ambient eddy viscosity ratio": 50.0,
    },
    "ic_2": {
        "reference": "IC_1",
        "static pressure ratio": 1.0,
        "total temperature ratio": 1.0,
    },
    "ic_3": {
        "reference": "IC_1",
        "mass flow rate": massflow,
        "total temperature ratio": 1.0,
    },
    "ic_4": {
        "reference": "IC_1",
        "total pressure ratio": 1.01,
        "total temperature ratio": 1.0,
    },
    "bc_1": {"zone": wall, "type": "wall", "kind": "noslip"},
    "bc_2": {"zone": inlet, "type": "inflow", "condition": "IC_3", "kind": "massflow"},
    "bc_3": {
        "zone": outlet,
        "type": "outflow",
        "kind": "pressure",
        "condition": "IC_2",
    },
    # 'bc_2': {'zone': inlet,
    #          'type': 'farfield',
    #          'condition': 'IC_1',
    #          'kind': 'riemann'},
    # 'bc_3': {'zone': outlet,
    #          'type': 'farfield',
    #          'kind': 'riemann',
    #          'condition': 'IC_1'},
    "write output": {
        "format": "vtk",
        "surface variables": [
            "p",
            "T",
            "rho",
            "yplus",
            "cf",
            "cp",
            "pressureforce",
            "frictionforce",
            "V",
        ],
        "volume variables": [
            "V",
            "p",
            "T",
            "rho",
            "eddy",
            "turbulencekineticenergy",
            "turbulenceeddyfrequency",
        ],
    },
    "report": {
        "frequency": 10,
        "massflow": {"mf_1": {"name": "out", "zone": [2]}},
        "monitor": {
            "mr_1": {"name": "pt_1", "point": [0.0, 0.0, 5.0], "variables": ["V", "p"]},
            "mr_2": {
                "name": "pt_2",
                "point": [0.0, 0.0, 10.0],
                "variables": ["V", "p"],
            },
            "mr_3": {
                "name": "pt_3",
                "point": [0.0, 0.0, 15.0],
                "variables": ["V", "p"],
            },
            "mr_4": {
                "name": "pt_4",
                "point": [0.0, 0.0, 20.0],
                "variables": ["V", "p"],
            },
            "mr_5": {
                "name": "pt_5",
                "point": [0.0, 0.0, 25.0],
                "variables": ["V", "p"],
            },
            "mr_6": {
                "name": "pt_6",
                "point": [0.0, 0.0, 30.0],
                "variables": ["V", "p"],
            },
            "mr_7": {
                "name": "pt_7",
                "point": [0.0, 0.0, 35.0],
                "variables": ["V", "p"],
            },
            "mr_8": {
                "name": "pt_8",
                "point": [0.0, 0.0, 40.0],
                "variables": ["V", "p"],
            },
            "mr_9": {
                "name": "pt_9",
                "point": [0.0, 0.0, 45.0],
                "variables": ["V", "p"],
            },
            "mr_10": {
                "name": "pt_10",
                "point": [0.0, 0.0, 50.0],
                "variables": ["V", "p"],
            },
        },
    },
}
