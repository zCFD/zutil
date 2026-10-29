import zutil

translation = [0.0, 1.2, -1.2]
angle = 25.0
rotation_vector = [0.0, 1.0, 0.0]
rotate_mesh = True
reverse = True
steady = False


if rotate_mesh:
    transform_angle = angle
else:
    transform_angle = 0.0

parameters = zutil.get_parameters_from_file("cylinder_base.py", namespace=globals())
parameters["time marching"]["moving mesh"] = True
