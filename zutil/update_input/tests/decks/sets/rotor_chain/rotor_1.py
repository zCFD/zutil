import zutil

rotor_number = 1

parameters = zutil.get_parameters_from_file("rotor_base.py", namespace=globals())

parameters["report"]["forces"] = {
    "FR_1": {"name": "rotor", "zone": [1], "transform": rotor_transform}
}
parameters["time marching"]["moving mesh"] = True
parameters["FZ_1"] = {
    "type": "rotating",
    "zone": [0],
    "axis": [0.0, 0.0, 1.0],
    "origin": [0.0, 0.0, 0.0],
    "omega": rotor_omega,
}
parameters["BC_1"] = {"zone": [1], "type": "wall", "kind": "slip"}
parameters["BC_2"] = {"zone": [2], "type": "overset"}
