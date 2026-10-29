import zutil

translation = [0.0, -1.2, -1.2]
angle = 0.0
rotation_vector = [0.0, 1.0, 1.0]
rotate_mesh = False
reverse = False
steady = True


if rotate_mesh:
    transform_angle = angle
else:
    transform_angle = 0.0

parameters = zutil.get_parameters_from_file("cylinder_base.py", namespace=globals())
parameters["time marching"]["unsteady"]["total time"] = 1.0
parameters["time marching"]["unsteady"]["time step"] = 1.0
