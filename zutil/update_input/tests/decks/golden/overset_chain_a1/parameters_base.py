import zutil
import math


frequency = 1000 / 60.0  # 1000 RPM

# the timestep is determined by the fastest rotation speed
dt = (1.0 / frequency) / (720.0 / 1.0)  # half a degree per timestep
print("dt = ", dt)


tt = 20 * dt  # 1 rotation

print("tt = ", tt)

omega_ref = 2.0 * math.pi * frequency

# ISA 1976 conditions
temp = 288.15  # K
press = 101325  # Pa

rad = 1.0  # radius in m
swept_area = math.pi * rad**2

# Dimensionalise using fastest tip speed
U_tip = omega_ref * rad
mach_tip = U_tip / math.sqrt(1.4 * 287.6 * temp)
mu = 1.81e-5

speed = 15.0  # m/3

cycles = 10

if steady:
    tt = 1.0
    dt = 1.0
    cycles = 200

parameters = {
    "units": "SI",
    "restart": False,
    "initial": "IC_2",
    "reference": "IC_1",
    "time marching": {
        "unsteady": {
            "total time": tt,
            "time step": dt,
            "order": "second",
            "start": 100,
        },
        "scheme": {"name": "implicit euler", "stage": 1},
        # "scheme": {"name": "runge kutta", "stage": 5},
        "cfl": 250,
        "cfl ramp factor": {"initial": 1, "growth": 1.1},
        "cycles": cycles,
    },
    "mapping": {
        "number of additional overset active layers": 2,
    },
    "equations": "viscous",
    "viscous": {
        "order": "second",
        "inviscid flux scheme": "Roe",
        "limiter": "van albada",
        "leastsq gradients": True,
        "reconstruction": "umuscl",
    },
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": 287.0,
        "Sutherlands const": 110.4,
        "Prandtl No": 0.72,
        "Turbulent Prandtl No": 0.9,
    },
    "IC_1": {  # this is just for non-dimen, taken from tip conditions
        "temperature": temp,
        "pressure": press,
        "V": {"vector": zutil.vector_from_angle(-angle, 0.0), "Mach": mach_tip},
        "viscosity": mu,
    },
    # Initial conditions
    "IC_2": {
        "temperature": temp,
        "pressure": press,
        "V": {"vector": zutil.vector_from_angle(-angle, 0.0), "Mach": 0.1},
        "viscosity": mu,
    },
    "Sponge Layer Damping": {
        "Distance": 5.0,
        "Damping Factor": 5.0,
        "condition": "IC_2",
    },
    "write output": {
        "format": "vtk",
        "surface variables": [
            "cf",
            "cp",
            "p",
            "T",
            "rho",
            "V",
            "immersed boundary vector",
            "immersed boundary normal",
        ],
        "volume variables": [
            "walldist",
            "overset",
            "cell_velocity",
            "V",
            "cp",
            "rho",
            "Qcriterion",
            "sponge",
        ],
    },
    "report": {
        "frequency": 10,
    },
}
