import zutil

total_time = 2.0
temperature = 300.0

parameters = zutil.get_parameters_from_file("base.py", namespace=globals())

parameters["BC_1"] = {"zone": [7], "type": "overset"}
