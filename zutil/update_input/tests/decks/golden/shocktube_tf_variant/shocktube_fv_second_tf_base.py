import zutil
import numpy as np
import zutil.transform

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


def tr_matrix():
    return zutil.transform.rotate(rot_vec, rot_ang_deg)


def my_velocity(v):
    vec = np.asarray(v + [0])
    mat = tr_matrix()
    v2 = np.dot(mat, vec)
    return [v2[0], v2[1], v2[2]]


def shock(**kwargs):
    location = kwargs["location"]
    velocity = my_velocity([ul, 0.0, 0.0])
    pressure = pl
    density = rl
    temperature = tl

    vec = list(location) + [1.0]
    mat = tr_matrix()
    new_loc = np.dot(mat.T, vec)
    if new_loc[0] > x0:
        velocity = my_velocity([ur, 0.0, 0.0])
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
    "mesh transform matrix": tr_matrix(),
    # reference state
    "reference": "IC_1",
    "initial": {"name": "IC_1", "func": shock},
    # restart from previous solution
    "restart": False,
    "partitioner": "metis",
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
        "V": {"vector": my_velocity([ul, 0.0, 0.0])},
        "Reynolds No": 1000.0,
        "Reference Length": 1.0,
        "turbulence intensity": 0.01,
        "eddy viscosity ratio": 0.1,
    },
    "IC_2": {
        "temperature": tr,
        "pressure": pr,
        "V": {"vector": my_velocity([ur, 0.0, 0.0])},
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
