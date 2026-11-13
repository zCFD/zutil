import zutil

angle = 0.0
steady = True

parameters = zutil.get_parameters_from_file("background_base.py", namespace=globals())

parameters["IC_1"]["V"]["vector"] = zutil.vector_from_angle(-angle, 0.0)
parameters["IC_2"]["V"]["vector"] = zutil.vector_from_angle(-angle, 0.0)
