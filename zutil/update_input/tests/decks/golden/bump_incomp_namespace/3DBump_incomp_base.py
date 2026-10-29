import zutil
import math

inlet = [4]
outlet = [5]
wall_solid = [7]
wall_symmetry = [6, 8]
sides = [2, 3]
top = [9]

# https://turbmodels.larc.nasa.gov/bump3d.html
temp = zutil.to_kelvin(540.0)
mach = 0.2
press = 101325.0  # Pa
dens = press / (temp * 287.0)
sound = math.sqrt(1.4 * 287.0 * temp)
speed = sound * mach
inlet_area = 1.0 * 5.0  # units - let's make them metres
massflow = dens * speed * inlet_area
inflow = [speed, 0.0, 0.0]
eddy = 0.01
TI = 3.8e-2
eddy_ambient = 1e-20
TI_ambient = 1e-20

parameters = {
    "units": "SI",
    "scale": [1.0, 1.0, 1.0],
    "reference": "IC_1",
    "initial": {"name": "IC_1"},
    "restart": False,
    "safe": False,
    "partitioner": "metis",
    "mesh quality remediation angle threshold": 70.0,
    "time marching": {
        "unsteady": {
            "total time": 1.0,
            "time step": 1.0,
            "order": "second",
        },
        "cycles": 2000,
    },
    "equations": "incomp RANS",
    "incomp RANS": {
        "order": "second",
        "limiter": "barth jespersen",
        "precondition": True,
        "inviscid flux scheme": "Roe",
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
        "v": {"vector": inflow},
        "reynolds no": 3e6,
        "reference length": 1.0,
        "turbulence intensity": TI,
        "eddy viscosity ratio": eddy,
        "ambient turbulence intensity": TI_ambient,
        "ambient eddy viscosity ratio": eddy_ambient,
    },
    "ic_2": {
        "reference": "IC_1",
        "static pressure ratio": 1.0,
        "total temperature ratio": 1.0,
    },
    "ic_3": {
        "reference": "IC_1",
        "mass flow rate": massflow,
        "total temperature ratio": 1.008,
    },
    "ic_4": {
        "reference": "IC_1",
        "total pressure ratio": 1.0,
        "total temperature ratio": 1.0,
    },
    "bc_1": {"zone": wall_solid, "type": "wall", "kind": "wallfunction"},
    "bc_2": {"zone": inlet, "type": "inflow", "condition": "IC_3", "kind": "massflow"},
    "bc_3": {
        "zone": outlet,
        "type": "outflow",
        "kind": "pressure",
        "condition": "IC_2",
    },
    "bc_4": {"zone": sides + top, "type": "symmetry"},
    "bc_5": {"zone": wall_symmetry, "type": "symmetry"},
    "write output": {
        "format": "vtk",
        "frequency": {
            "volume data": 1000,
            "surface data": 1000,
        },
        "surface variables": [
            "p",
            "T",
            "cp",
            "pressureforce",
            "V",
            "yplus",
        ],
        "volume variables": [
            "V",
            "p",
            "T",
            "turbulencekineticenergy",
            "eddy",
        ],
    },
    "report": {
        "frequency": 1,
        "massflow": {
            "mf_1": {"name": "in", "zone": inlet},
            "mf_2": {"name": "out", "zone": outlet},
            "mf_3": {"name": "wall", "zone": wall_solid + wall_symmetry},
            "mf_4": {"name": "top", "zone": top},
            "mf_5": {"name": "sides", "zone": sides},
            "mf_6": {
                "name": "total",
                "zone": inlet + outlet + wall_solid + wall_symmetry + top + sides,
            },
        },
        "forces": {
            "FR_1": {
                "name": "wall",
                "zone": wall_solid,
                #'transform' : my_transform,
                "reference area": 1.5,
                "reference length": 1.0,
                "reference point": [0.0, 0.0, 0.0],
                "reference pressure": press,
            },
        },
    },
}
