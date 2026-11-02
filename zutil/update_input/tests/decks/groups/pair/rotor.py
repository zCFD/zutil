import math

import zutil

omega = 2.0 * math.pi * 10.0

parameters = zutil.get_parameters_from_file("shared.py")

parameters["time marching"]["moving mesh"] = True
parameters["FZ_1"] = {
    "type": "rotating",
    "zone": [0],
    "axis": [0.0, 0.0, 1.0],
    "origin": [0.0, 0.0, 0.0],
    "omega": omega,
}
parameters["BC_1"] = {"zone": [7], "type": "overset"}
parameters["BC_2"] = {"zone": [8], "type": "wall", "kind": "slip"}
parameters["write output"]["volume variables"] = ["p", "overset"]
parameters["report"]["forces"] = {"FR_1": {"name": "rotor", "zone": [8]}}
