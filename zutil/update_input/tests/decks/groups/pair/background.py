import zutil

parameters = zutil.get_parameters_from_file("shared.py")

parameters["BC_1"] = {
    "zone": [1, 2],
    "type": "farfield",
    "kind": "riemann",
    "condition": "IC_1",
}
parameters["BC_2"] = {"zone": [3], "type": "symmetry"}
