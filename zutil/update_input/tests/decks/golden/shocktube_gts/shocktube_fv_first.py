# Zone: 12 XMIN
# Zone: 13 XMAX
# Zone: 14 YMAX
# Zone: 15 YMIN
# Zone: 16 ZMIN
# Zone: 17 ZMAX

xmin = [12]
xmax = [13]
sym = [14, 15, 16, 17]

R = 1.0
x0 = 0.3

# Upstream
rl = 1.0
ul = 0.75
pl = 1.0
tl = pl / (R * rl)

# Downstream
rr = 0.125
ur = 0.0
pr = 0.1
tr = pr / (R * rr)


def shock(**kwargs):
    location = kwargs["location"]

    velocity = [ul, 0.0, 0.0]
    pressure = pl
    density = rl
    temperature = tl

    if location[0] > x0:
        velocity = [ur, 0.0, 0.0]
        pressure = pr
        density = rr
        temperature = tr

    return {
        "pressure": pressure,
        "temperature": temperature,
        "velocity": tuple(velocity),
    }


parameters = {
    # units for dimensional quantities
    "units": "SI",
    # reference state
    "reference": "IC_1",
    "initial": {"name": "IC_1", "func": shock},
    # restart from previous solution
    "restart": False,
    "partitioner": "metis",
    # time marching properties
    "time marching": {
        "unsteady": {"total time": 0.2, "time step": 1.0e-5},
        "scheme": {
            "name": "runge kutta",
            "stage": "rk third order tvd",
            "kind": "global timestepping",
        },
    },
    "equations": "euler",
    "euler": {"order": "first", "entropy fix": 0.05},
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": R,
        "Sutherlands const": 110.4,
        "Prandtl No": 0.71,
        "Turbulent Prandtl No": 0.9,
    },
    "IC_1": {
        "temperature": tl,
        "pressure": pl,
        "V": {"vector": [ul, 0.0, 0.0]},
        "Reynolds No": 1.0,
        "Reference Length": 1.0,
        "turbulence intensity": 0.01,
        "eddy viscosity ratio": 0.1,
    },
    "IC_2": {
        "temperature": tr,
        "pressure": pr,
        "V": {"vector": [ur, 0.0, 0.0]},
        "Reynolds No": 1.0,
        "Reference Length": 1.0,
        "turbulence intensity": 0.01,
        "eddy viscosity ratio": 0.1,
    },
    "BC_1": {"zone": sym, "type": "symmetry"},
    "BC_2": {"zone": xmin, "type": "farfield", "kind": "riemann", "condition": "IC_1"},
    "BC_3": {"zone": xmax, "type": "farfield", "kind": "riemann", "condition": "IC_2"},
    "write output": {
        "format": "vtk",
        "surface variables": ["V", "p", "T", "rho"],
        "volume variables": ["V", "p", "T", "rho"],
    },
    "report": {"frequency": 1000},
}

############################
#
# Variable list
#
# var_1 to var_n
# p,pressure
# T, temperature
#
############################
