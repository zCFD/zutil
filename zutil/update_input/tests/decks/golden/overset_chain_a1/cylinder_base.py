import zutil
import math
import zutil.transform


def transform_matrix():
    A2B = zutil.transform.rotate(rotation_vector, angle)
    B2C = zutil.transform.translate([0.0, 0.0, 0.0], translation, A2B)

    return B2C


origin = [0.0, 0.0, 0.0]
origin = [o + t for o, t in zip(origin, translation)]

frequency = 1000.0 / 60  # 1000 rpm
omega = 2.0 * math.pi * frequency

if reverse:
    omega = -omega

# load paramter namespace from file
parameters = zutil.get_parameters_from_file("parameters_base.py", namespace=globals())

parameters["mesh transform matrix"] = transform_matrix()

parameters["BC_1"] = {
    "zone": [7],
    "type": "overset",
}

parameters["FZ_1"] = {
    "type": "rotating",
    "zone": [0],
    "axis": zutil.vector_from_angle(-angle, 0.0),
    "origin": origin,
    "omega": omega,
}
