import zutil

parameters = zutil.get_parameters_from_file("base.py")

parameters["BC_1"] = {
    "zone": [1],
    "type": "inflow",
    "condition": "IC_1",
    "kind": "default",
}
parameters["BC_2"] = {
    "zone": [2],
    "type": "outflow",
    "condition": "IC_1",
    "kind": "pressure",
}
