import zutil

farfield_zones = [1, 2, 3, 4, 5, 6]

parameters = zutil.get_parameters_from_file("parameters_base.py", namespace=globals())

parameters["BC_1"] = {
    "zone": farfield_zones,
    "type": "farfield",
    "kind": "riemann",
    "condition": "IC_2",
}
