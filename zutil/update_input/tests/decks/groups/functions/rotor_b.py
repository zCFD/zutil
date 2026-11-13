import zutil

offset = [1.0, 0.0, 0.0]


def place(*, vector):
    return tuple(a + b for a, b in zip(vector, offset))


parameters = zutil.get_parameters_from_file("base.py")

parameters["report"]["forces"]["FR_2"] = {
    "name": "rotor",
    "zone": [7],
    "transform": place,
}
parameters["BC_1"] = {"zone": [7], "type": "overset"}
