import zutil

parameters = zutil.get_parameters_from_file("base.py")

parameters["time marching"]["cycles"] = 9
parameters["BC_1"] = {"zone": [7], "type": "overset"}
parameters["BC_2"] = {"zone": [8], "type": "wall", "kind": "noslip"}
