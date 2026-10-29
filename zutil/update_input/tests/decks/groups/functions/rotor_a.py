import math

import zutil


def tilt(*, vector):
    x, y, z = vector
    c, s = math.cos(0.1), math.sin(0.1)
    return (c * x - s * y, s * x + c * y, z)


parameters = zutil.get_parameters_from_file("base.py")

parameters["report"]["forces"]["FR_2"] = {
    "name": "rotor",
    "zone": [7],
    "transform": tilt,
}
parameters["BC_1"] = {"zone": [7], "type": "overset"}
parameters["write output"]["volume variables"] = ["p", "overset"]
