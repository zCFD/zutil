import zutil

total_time = 1.0
temperature = 288.15

parameters = zutil.get_parameters_from_file("base.py", namespace=globals())

parameters["IC_2"] = {
    "temperature": 300.0,
    "pressure": 101325.0,
    "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.1},
    "Reynolds No": 1.0e6,
}
parameters["BC_1"] = {
    "zone": [1],
    "type": "farfield",
    "kind": "riemann",
    "condition": "IC_2",
}
