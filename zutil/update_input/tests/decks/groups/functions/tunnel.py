import zutil

parameters = zutil.get_parameters_from_file("base.py")

parameters["BC_1"] = {
    "zone": [1],
    "type": "farfield",
    "kind": "riemann",
    "condition": "IC_1",
}
