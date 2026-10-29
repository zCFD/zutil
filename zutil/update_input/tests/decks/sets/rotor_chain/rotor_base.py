import zutil

parameters = zutil.get_parameters_from_file("base.py", namespace=globals())

rotor_index = rotor_number - 1
rotor_omega = omega[rotor_index]


def rotor_transform(*, vector):
    return vector
